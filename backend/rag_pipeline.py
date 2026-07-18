import os

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_PATH, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

_embeddings = OpenAIEmbeddings()
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def load_document(file_path: str) -> list[Document]:
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type.")


def load_url(url: str) -> list[Document]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url})]


def _store(documents: list[Document], source_name: str) -> int:
    for doc in documents:
        doc.metadata["source"] = source_name

    chunks = _splitter.split_documents(documents)
    Chroma.from_documents(chunks, embedding=_embeddings, persist_directory=CHROMA_PATH)
    return len(chunks)


def process_and_store(file_path: str) -> int:
    documents = load_document(file_path)
    return _store(documents, os.path.basename(file_path))


def process_url(url: str) -> int:
    documents = load_url(url)
    return _store(documents, url)


def retrieve(question: str, k: int = 3) -> list[Document]:
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=_embeddings)
    return db.similarity_search(question, k=k)
