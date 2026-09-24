from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_chunk_docs(chunk_size: int | None = None) -> tuple[list[str], list[str], list[dict]]:
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        doc_id = path.stem
        if chunk_size is None:
            chunks = [text]
        else:
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            ids.append(f"{doc_id}_chunk_{idx:02d}")
            metadatas.append({"document_id": doc_id, "chunk_index": idx, "source_file": path.name})
    if len(texts) == 0:
        raise FileNotFoundError(f"No doc_*.txt files found in {DOCS_DIR}")
    return texts, ids, metadatas


def build_index() -> None:
    texts, ids, metadatas = load_and_chunk_docs()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(ids)} chunks from {len(set(x.split('_chunk_')[0] for x in ids))} documents into ChromaDB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()
