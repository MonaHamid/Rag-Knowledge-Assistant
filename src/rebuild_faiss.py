import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Load chunk data ---
with open("notebooks/data/processed/hybrid_chunks.pkl", "rb") as f:
    data = pickle.load(f)

# Extract actual chunks list
chunks = data["chunks"]
print(f"Loaded {len(chunks)} chunks.")

# Extract text properly
if isinstance(chunks[0], dict) and "text" in chunks[0]:
    texts = [c["text"] for c in chunks]
else:
    texts = [str(c) for c in chunks]

# --- Embed text ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

# --- Build FAISS index ---
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("FAISS index built with", index.ntotal, "vectors.")

# --- Save index + metadata ---
os.makedirs("notebooks/data/processed", exist_ok=True)
faiss.write_index(index, "notebooks/data/processed/retrieval_system_index.faiss")

# Save metadata
metadata = [{"text": t} for t in texts]
with open("notebooks/data/processed/retrieval_system_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Saved FAISS and metadata successfully!")
