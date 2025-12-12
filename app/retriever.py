import os
import json
import time
from typing import List, Dict, Any, Optional
import openai
import faiss
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY không được cấu hình")
openai.api_key = OPENAI_API_KEY

PERSIST_DIR = os.getenv("PERSIST_DIR", "./persist")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# Dim for text-embedding-3-small -> 1536; we'll infer from embeddings returned
DEFAULT_DIM = 1536

class Document(BaseModel):
    id: str
    text: str

class Retriever:
    def __init__(self, persist_dir: str = PERSIST_DIR, embed_model: str = EMBED_MODEL):
        self.persist_dir = persist_dir
        self.embed_model = embed_model
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.index: Optional[faiss.Index] = None
        self.dim = DEFAULT_DIM
        os.makedirs(self.persist_dir, exist_ok=True)
        # try load existing index
        try:
            self._load_persisted_index()
        except Exception:
            # no persisted index yet
            self.index = None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = openai.Embedding.create(input=texts, model=self.embed_model)
        return [item['embedding'] for item in resp['data']]

    def index_documents(self, documents: List[Document], persist: bool = True):
        """
        Index list of Document(id, text). Overwrites current in-memory index.
        After building index, optionally persist to disk.
        """
        texts = [d.text for d in documents]
        ids = [d.id for d in documents]

        # compute embeddings in batches to be safe
        batch_size = 32
        embs_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embs = self._embed(batch_texts)
            embs_list.extend(embs)
        embs = np.array(embs_list).astype('float32')

        # normalize embeddings for cosine similarity (inner product)
        faiss.normalize_L2(embs)
        self.dim = embs.shape[1]
        self.ids = ids
        self.texts = texts
        # create IndexFlatIP (inner product) for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

        if persist:
            self._persist_index(embs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return list of evidences: {id, score, text, snippet}
        If no index, returns empty list.
        """
        if self.index is None:
            return []

        q_emb = np.array(self._embed([query])).astype('float32')
        faiss.normalize_L2(q_emb)
        top_k = min(top_k, len(self.ids))
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            text = self.texts[idx]
            results.append({
                "id": self.ids[idx],
                "score": float(score),
                "text": text,
                "snippet": text[:400].replace("\n", " ")  # truncated snippet
            })
        return results

    # ---- persistence helpers ----
    def _persist_index(self, embeddings: np.ndarray):
        """
        Persist FAISS index and metadata to self.persist_dir.
        Saves:
          - faiss.index
          - meta.json (ids, texts, dim, created_at)
          - embeddings.npy (optional)
        """
        index_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "meta.json")
        embed_path = os.path.join(self.persist_dir, "embeddings.npy")
        # write index
        faiss.write_index(self.index, index_path)
        meta = {
            "ids": self.ids,
            "texts": self.texts,
            "dim": self.dim,
            "created_at": time.time()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        # save embeddings optionally (useful for reindexing)
        try:
            np.save(embed_path, embeddings)
        except Exception:
            pass  # optional

    def _load_persisted_index(self):
        """
        Load persisted index and metadata if present. Raises FileNotFoundError if not.
        """
        index_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "meta.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("No persisted index found")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.ids = meta.get("ids", [])
        self.texts = meta.get("texts", [])
        self.dim = int(meta.get("dim", DEFAULT_DIM))

    # convenience helpers for JSONL doc store
    def save_documents_jsonl(self, docs: List[Document], path: Optional[str] = None):
        path = path or os.path.join(self.persist_dir, "docs.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for d in docs:
                json.dump({"id": d.id, "text": d.text}, f, ensure_ascii=False)
                f.write("\n")

    def load_documents_jsonl(self, path: Optional[str] = None) -> List[Document]:
        path = path or os.path.join(self.persist_dir, "docs.jsonl")
        docs = []
        if not os.path.exists(path):
            return docs
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(Document(id=obj["id"], text=obj["text"]))
        return docs
