import os
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from bs4 import BeautifulSoup
import requests
from .config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Path to store Chroma DB
CHROMA_PATH = "backend/data/db"


def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return UnstructuredWordDocumentLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type.")

def load_url(url: str):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        return [{"page_content": text, "metadata": {"source": url}}]
    except Exception as e:
        return [{"page_content": f"Error: {e}", "metadata": {"source": url}}]

def process_and_store(file_path: str):
    documents = load_document(file_path)

    for doc in documents:
        doc.metadata["source"] = file_path

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    return len(chunks)

def query_rag(question: str, k=3):
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    docs = db.similarity_search(question, k=k)

    sources = [f"[{i+1}] {doc.metadata.get('source', 'Unknown')}" for i, doc in enumerate(docs)]
    content = "\n\n".join([doc.page_content for doc in docs])
    source_str = "\n\nSources:\n" + "\n".join(sources)
    return content + source_str