import json
import os

from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from src.prompt import *


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY .env file mein missing hai")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY .env file mein missing hai")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 15},
)

chatModel = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    api_key=GROQ_API_KEY
)
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the latest user question as a standalone search query using "
            "the conversation history when needed. Preserve the condition and "
            "meaning already being discussed. Never answer the question. If it "
            "is not a follow-up, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    chatModel,
    retriever,
    contextualize_prompt,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question.", 400

    try:
        raw_history = json.loads(request.form.get("history", "[]"))
    except json.JSONDecodeError:
        return "Invalid chat history.", 400

    if not isinstance(raw_history, list):
        return "Chat history must be a list.", 400

    chat_history = []
    for message in raw_history[-20:]:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            return "Invalid chat history message.", 400

        content = message["content"].strip()
        if content:
            message_type = HumanMessage if message["role"] == "user" else AIMessage
            chat_history.append(message_type(content=content[:4000]))

    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
