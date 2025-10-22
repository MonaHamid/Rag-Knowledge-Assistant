"""
faiss_to_qdrant_cloud.py
------------------------
Convert FAISS-based vector index into Qdrant Cloud for persistence and metadata filtering.

✅ Works directly with Qdrant Cloud (no Docker needed)
✅ Tests connection before uploading
✅ Displays progress and sample search results
"""

import os
import numpy as np
import pickle
import faiss
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from requests.exceptions import RequestException

# -----------------------------
# 1️⃣ Load FAISS + Embeddings + Metadata
# -----------------------------
EMBED_PATH = "notebooks/data/processed/retrieval_system_embeddings.npy"
META_PATH = "notebooks/data/processed/retrieval_system_metadata.pkl"
INDEX_PATH = "notebooks/data/processed/retrieval_system_index.faiss"

if not (os.path.exists(EMBED_PATH) and os.path.exists(META_PATH) and os.path.exists(INDEX_PATH)):
    raise FileNotFoundError("❌ One or more FAISS files not found in notebooks/data/processed/. Check paths.")

print("📦 Loading FAISS data...")
embeddings = np.load(EMBED_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)
faiss_index = faiss.read_index(INDEX_PATH)

if len(metadata) != embeddings.shape[0]:
    raise ValueError("⚠️ Metadata & embeddings count mismatch!")

VECTOR_SIZE = embeddings.shape[1]
print(f"✅ Loaded {len(embeddings)} vectors of dimension {VECTOR_SIZE}")

# -----------------------------
# 2️⃣ Connect to Qdrant Cloud
# -----------------------------
QDRANT_URL = "https://6eb0b696-8d2d-49fb-8c75-d9d178cf0b62.eu-west-1-0.aws.cloud.qdrant.io"   # ← Replace with your actual Qdrant Cloud endpoint
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uP4MxoWXxr6NWVlG2Kiey6vXsQafdSSDyWz8GbPEzmw"                 # ← Replace with your actual API key
COLLECTION_NAME = "pikaplexity_docs_cloud"

print("🔗 Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Connection test
try:
    info = client.get_collections()
    print(f"✅ Connected! Existing collections: {[c.name for c in info.collections]}")
except RequestException as e:
    raise ConnectionError(f"❌ Could not connect to Qdrant Cloud. Check URL/API key.\nDetails: {e}")

# -----------------------------
# 3️⃣ Create / Recreate Collection
# -----------------------------
print(f"⚙️ Creating collection '{COLLECTION_NAME}' in Qdrant Cloud...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)
print("✅ Collection ready!")

# -----------------------------
# 4️⃣ Upload Vectors + Metadata
# -----------------------------
print("🚀 Uploading data to Qdrant Cloud...")
batch_size = 256
num_points = len(embeddings)

for start in tqdm(range(0, num_points, batch_size)):
    end = min(start + batch_size, num_points)
    batch_vectors = embeddings[start:end]
    batch_metadata = metadata[start:end]

    points = [
        PointStruct(
            id=start + i,
            vector=batch_vectors[i].tolist(),
            payload=batch_metadata[i]
        )
        for i in range(len(batch_vectors))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"🎉 Successfully uploaded {num_points} vectors to '{COLLECTION_NAME}'")

# -----------------------------
# 5️⃣ Quick Search Test
# -----------------------------
print("\n🔍 Performing test search...")
test_vector = embeddings[0]
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=test_vector,
    limit=3
)

for r in results:
    title = r.payload.get("title", "Unknown")
    domain = r.payload.get("domain", "Unknown domain")
    print(f"• {title} ({domain}) — score: {r.score:.3f}")

print("\n✅ Migration from FAISS → Qdrant Cloud complete! ☁️⚡")