from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb import HttpClient


from config import VECTOR_DB_DIR, COLLECTION_NAME


class VectorDB:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset(self):
        """Danger: clears the whole collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        # Flatten a bit
        return {
            "documents": result.get("documents", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "distances": result.get("distances", [[]])[0],
            "ids": result.get("ids", [[]])[0],
        }
