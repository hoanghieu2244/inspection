import os
import json
import faiss
import numpy as np
from typing import List, Tuple

def save_faiss_index(index: faiss.Index, ids: List[str], texts: List[str], dir_path: str):
    """
    Lưu FAISS index và metadata (ids,texts) vào thư mục dir_path.
    """
    os.makedirs(dir_path, exist_ok=True)
    index_path = os.path.join(dir_path, "faiss.index")
    meta_path = os.path.join(dir_path, "meta.json")
    faiss.write_index(index, index_path)
    meta = {"ids": ids, "texts": texts}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return index_path, meta_path

def load_faiss_index(dir_path: str) -> Tuple[faiss.Index, List[str], List[str]]:
    """
    Load FAISS index và metadata. Raise if not found.
    """
    index_path = os.path.join(dir_path, "faiss.index")
    meta_path = os.path.join(dir_path, "meta.json")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("FAISS index or metadata not found in " + dir_path)
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta.get("ids", [])
    texts = meta.get("texts", [])
    return index, ids, texts

def save_numpy_embeddings(embeddings: np.ndarray, path: str):
    """
    Nếu bạn lưu cả embeddings (không bắt buộc), dùng numpy.save.
    """
    np.save(path, embeddings)

def load_numpy_embeddings(path: str) -> np.ndarray:
    return np.load(path)
