import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

CHROMA_PATH = str(BACKEND_DIR / "data" / "db")
DOCS_PATH = str(BACKEND_DIR / "data" / "docs")
FRONTEND_DIR = str(REPO_ROOT / "frontend")
