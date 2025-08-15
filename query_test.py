from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np, os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
qvec = model.encode(["Chicken Gyros"], normalize_embeddings=True).astype("float32")[0].tolist()

hits = client.search(collection_name=os.getenv("QDRANT_COLLECTION","recipes"),
                     query_vector=qvec, limit=5, with_payload=True)

info = client.get_collection(os.getenv("QDRANT_COLLECTION","recipes"))
print(info)
for h in hits:
    print(round(h.score, 4), h.payload["title"], h.payload.get("source_url"))