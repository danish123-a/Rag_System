import uuid
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from config import DATA_DIR, EMBEDDING_MODEL_NAME, MAX_CHUNK_SIZE, CHUNK_OVERLAP
from vector_store import VectorDB



def read_text_files(data_dir: Path) -> Dict[str, str]:
    files = {}
    for path in data_dir.glob("*.txt"):
        files[path.name] = path.read_text(encoding="utf-8", errors="ignore")
    return files


def chunk_text(text: str, max_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_documents() -> Dict[str, List[Dict[str, Any]]]:
    raw_files = read_text_files(DATA_DIR)
    all_docs = {}

    for filename, text in raw_files.items():
        chunks = chunk_text(text, MAX_CHUNK_SIZE, CHUNK_OVERLAP)
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"{filename}-{i}-{uuid.uuid4().hex[:8]}",
                    "text": chunk,
                    "metadata": {
                        "source_file": filename,
                        "chunk_index": i,
                    },
                }
            )
        all_docs[filename] = docs
    return all_docs


def main(reset_db: bool = True):
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vdb = VectorDB()

    if reset_db:
        print("Resetting vector DB (deleting old data)...")
        vdb.reset()

    print("Preparing documents...")
    file_docs = build_documents()

    ids = []
    texts = []
    metadatas = []

    for filename, docs in file_docs.items():
        print(f"Ingesting {filename} with {len(docs)} chunks...")
        for d in docs:
            ids.append(d["id"])
            texts.append(d["text"])
            metadatas.append(d["metadata"])

    print("Computing embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    print("Adding to vector DB...")
    vdb.add_documents(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Done. Ingested {len(ids)} chunks.")


if __name__ == "__main__":
    main()
