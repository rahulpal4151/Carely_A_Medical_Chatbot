import hashlib
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        metadata = {"source": doc.metadata.get("source")}
        for key in ("page", "page_label"):
            if key in doc.metadata:
                metadata[key] = doc.metadata[key]
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )
    return minimal_docs


def prepare_documents_for_indexing(docs: List[Document]) -> tuple[List[Document], List[str]]:
    unique_docs: List[Document] = []
    ids: List[str] = []
    seen_ids = set()

    for doc in docs:
        identity = "\0".join(
            (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
                doc.page_content,
            )
        )
        doc_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if doc_id in seen_ids:
            continue

        seen_ids.add(doc_id)
        unique_docs.append(doc)
        ids.append(doc_id)

    return unique_docs, ids



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings