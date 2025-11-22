from typing import List, Dict, Any
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, TOP_K
from vector_store import VectorDB


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


class RAGPipeline:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.vdb = VectorDB()
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def search(self, query: str) -> List[RetrievedChunk]:
        query_emb = self.embed_text([query])[0]
        res = self.vdb.query(query_embedding=query_emb, top_k=self.top_k)

        chunks = []
        for doc, meta, dist, _id in zip(
            res["documents"],
            res["metadatas"],
            res["distances"],
            res["ids"],
        ):
            chunks.append(
                RetrievedChunk(
                    id=_id,
                    text=doc,
                    metadata=meta or {},
                    distance=dist,
                )
            )
        return chunks

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Super simple RAG: returns context + a naive answer.
        Plug in your own LLM here in place of `naive_answer`.
        """
        retrieved = self.search(query)
        context_text = "\n\n".join(
            f"[{i+1}] {chunk.text}"
            for i, chunk in enumerate(retrieved)
        )

        # Naive "answer": just echoes a template with context.
        # Replace this with a real LLM call.
        naive_answer = (
            "This is a retrieval-only demo. A real RAG system "
            "would call an LLM here with the context below.\n\n"
            "Context used:\n" + context_text
        )

        return {
            "query": query,
            "answer": naive_answer,
            "context": [
                {
                    "id": c.id,
                    "text": c.text,
                    "metadata": c.metadata,
                    "distance": c.distance,
                }
                for c in retrieved
            ],
        }
