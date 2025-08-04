import openai
from .config import OPENAI_KEY_KEY
from .rag_pipeline import query_rag

openai.api_key = OPENAI_KEY_KEY


def generate_response(prompt: str) -> str:
    context = query_rag(prompt)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are an assistant. Use this context:\n{context}"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
