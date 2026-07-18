import json

from openai import OpenAI

from config import OPENAI_API_KEY
from rag_pipeline import retrieve

_client = OpenAI(api_key=OPENAI_API_KEY)

GROUNDING_PROMPT = (
    "You are a document Q&A assistant. Answer the question using ONLY the "
    "context below. If the context does not contain the answer, respond "
    "exactly with \"I don't know based on the provided documents.\" Do not "
    "use any knowledge outside the given context."
)


def _build_sources(docs) -> list[dict]:
    sources = []
    for doc in docs:
        sources.append(
            {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:200].strip(),
            }
        )
    return sources


def stream_grounded_answer(question: str):
    """Yields Server-Sent-Events. Token events while generating, one final
    event carrying the full { answer, sources } contract."""
    try:
        docs = retrieve(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = _build_sources(docs)

        stream = _client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[
                {"role": "system", "content": f"{GROUNDING_PROMPT}\n\nContext:\n{context}"},
                {"role": "user", "content": question},
            ],
        )

        full_answer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta
                yield f"data: {json.dumps({'token': delta})}\n\n"

        yield f"data: {json.dumps({'done': True, 'answer': full_answer, 'sources': sources})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
