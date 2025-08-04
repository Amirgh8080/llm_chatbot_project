from fastapi import FastAPI, Request
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm_chat import generate_response
from rag_pipeline import process_and_store
from rag_pipeline import load_url

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: ChatRequest):
    reply = generate_response(req.prompt)
    return {"response": reply}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"backend/data/docs/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    n_chunks = process_and_store(file_path)
    return {"message": f"Stored {n_chunks} chunks from {file.filename}"}


class UrlRequest(BaseModel):
    url: str

@app.post("/upload_url")
def upload_url(req: UrlRequest):
    docs = load_url(req.url)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    return {"message": f"Stored {len(chunks)} chunks from URL"}