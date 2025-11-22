from typing import List, Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from rag import RAGPipeline

app = FastAPI(title="Text Searching RAG System (Vector DB)")
rag = RAGPipeline()


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class AskRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/search")
def search(request: SearchRequest) -> Dict[str, Any]:
    if request.top_k is not None:
        rag.top_k = request.top_k

    chunks = rag.search(request.query)
    return {
        "query": request.query,
        "results": [
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
                "distance": c.distance,
            }
            for c in chunks
        ],
    }


@app.post("/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    if request.top_k is not None:
        rag.top_k = request.top_k

    return rag.answer(request.query)


@app.get("/")
def root():
    return {
        "message": "RAG Vector DB API is running.",
        "endpoints": ["/search", "/ask"],
    }
