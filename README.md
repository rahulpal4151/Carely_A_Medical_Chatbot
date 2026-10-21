# Medical Chatbot

A local Flask web app that answers medical questions using retrieval-augmented
generation (RAG). It retrieves relevant passages from PDFs indexed in Pinecone
and asks a Groq-hosted language model to answer from those passages.

> This is an educational project, not a medical device. It does not replace
> professional medical advice, diagnosis, or treatment.

## Features

- Chat interface served by Flask.
- Loads PDF references, splits them into text chunks, and creates embeddings.
- Stores and retrieves document vectors with Pinecone.
- Uses conversation history to interpret follow-up questions.
- Uses maximal marginal relevance (MMR) retrieval to favor relevant, less
  repetitive passages.
- Uses a cautious medical prompt that instructs the model not to fill gaps
  with unsupported outside information.
- Sets the Groq model temperature to `0.0` for more consistent responses.

## Project structure

```text
.
├── app.py                  # Flask routes and conversational RAG chain
├── store_index.py          # PDF indexing and Pinecone upload
├── requirements.txt        # Python dependencies
├── setup.py                # Local Python package configuration
├── data/                   # Put PDFs you are permitted to use here
├── research/
│   └── trials.ipynb        # RAG experiments
├── src/
│   ├── helper.py           # PDF loading, chunking, and embedding helpers
│   └── prompt.py           # Medical-answer prompt
├── templates/
│   └── chat.html           # Chat page
└── static/
    └── style.css           # Chat page styles
```

The `.env` file and Python virtual environment are local files and should not
be committed. Medical PDFs are not included in this repository. Add only
documents you have the right to process; do not redistribute a book or PDF
unless its license or the rights holder permits it.

## Technology

- Python 3.10
- Flask
- LangChain
- Groq chat model: `openai/gpt-oss-20b`
- Hugging Face embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Pinecone vector database

## Run locally

### 1. Get the project and create an environment

```bash
git clone https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

conda create -n medibot python=3.10 -y
conda activate medibot
```

Alternatively, create and activate a virtual environment using Python's
`venv` module.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

Create a `.env` file in the project root:

```ini
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

Keep this file private. Do not commit real API keys to GitHub.

### 4. Add a source PDF and build the vector index

Place a PDF you are authorized to use in `data/`, then run:

```bash
python store_index.py
```

The script splits the PDF text into 500-character chunks with 20-character
overlap, embeds the chunks, and writes them to the Pinecone index named
`medical-chatbot`. A new index is created in the AWS `us-east-1` Pinecone
serverless region when it does not already exist. The first run may take longer
while the embedding model is downloaded.

### 5. Start the chatbot

```bash
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

## Notes

- `app.py` expects the Pinecone index to exist, so run `store_index.py` first.
- The app retrieves 3 passages from an initial set of up to 15 candidates.
- Re-running the index script uses stable IDs for unchanged chunks; avoid
  deleting the Pinecone index unless you intend to remove all its vectors.
- A model response can still be incomplete or wrong. Check important
  information with a qualified healthcare professional and reliable sources.
