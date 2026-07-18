import os

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DOCS_PATH, FRONTEND_DIR
from llm_chat import stream_grounded_answer
from rag_pipeline import process_and_store, process_url

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

os.makedirs(DOCS_PATH, exist_ok=True)


class ChatRequest(BaseModel):
    prompt: str


@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(
        stream_grounded_answer(req.prompt),
        media_type="text/event-stream",
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed")

    file_path = os.path.join(DOCS_PATH, safe_name)
    size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                os.remove(file_path)
                raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")
            f.write(chunk)

    try:
        n_chunks = process_and_store(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=502, detail=f"Failed to process document: {e}")
    return {"message": f"Stored {n_chunks} chunks from {safe_name}"}


class UrlRequest(BaseModel):
    url: str


@app.post("/upload_url")
def upload_url(req: UrlRequest):
    try:
        n_chunks = process_url(req.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to ingest URL: {e}")
    return {"message": f"Stored {n_chunks} chunks from URL"}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
