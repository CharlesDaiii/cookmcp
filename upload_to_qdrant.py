import os, json, uuid, re
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")  # 或 BAAI/bge-small-zh-v1.5

def embedder(model_name: str):
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())
    def encode(texts: List[str]) -> np.ndarray:
        # 归一化向量 => 适合 cosine
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")
    return encode, dim

def ensure_collection(client: QdrantClient, size: int):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION}' (dim={size})")

def text_for_embedding(rec: dict) -> str:
    parts = [rec.get("title","")]
    parts += rec.get("ingredients", [])
    parts += rec.get("steps", [])
    return "\n".join(parts)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main(jsonl_path="recipes_wikibooks.jsonl"):
    assert QDRANT_URL and QDRANT_API_KEY, "Please set QDRANT_URL and QDRANT_API_KEY"
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    encode, dim = embedder(EMBED_MODEL)
    ensure_collection(client, size=dim)

    batch, payloads = [], []
    batch_size = 96
    ids = []
    texts = []

    for rec in tqdm(load_jsonl(jsonl_path), desc="encoding"):
        raw_id = rec.get("id") or ""
        ids.append(str(uuid.uuid4()))
        payloads.append({
            "original_id": raw_id,
            "title": rec["title"],
            "source_url": rec["source_url"],
            "cuisine": rec.get("cuisine"),
            "tags": rec.get("tags", []),
            "servings": rec.get("servings"),
            "total_time_min": rec.get("total_time_min"),
            "ingredients": rec.get("ingredients"),
            "steps": rec.get("steps"),
            "license": rec.get("license"),
        })
        texts.append(text_for_embedding(rec))
        if len(texts) >= batch_size:
            vecs = encode(texts)
            points = [PointStruct(id=i, vector=v.tolist(), payload=p)
                      for i, v, p in zip(ids, vecs, payloads)]
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            ids, payloads, texts = [], [], []

    if texts:
        vecs = encode(texts)
        points = [PointStruct(id=i, vector=v.tolist(), payload=p)
                  for i, v, p in zip(ids, vecs, payloads)]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)

    print("Done upserting to Qdrant.")

if __name__ == "__main__":
    main()