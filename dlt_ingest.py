import os
import dlt
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------
# 1️⃣ Load environment variables
# ---------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------------------------
# 2️⃣ Initialize models & clients
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "pikaplexity_docs_cloud"

# Create Qdrant collection if missing
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 384, "distance": "Cosine"},
    )

# ---------------------------
# 3️⃣ Define DLT source (Data ingestion)
# ---------------------------
@dlt.source
def knowledge_source():
    """Reads the CSV knowledge base and yields documents as records."""
    df = pd.read_csv("data/wikipedia_knowledge_base_balanced.csv")

    # Detect which column to use as text
    text_col = next((c for c in ["text", "content", "chunk", "body", "summary"] if c in df.columns), None)
    if not text_col:
        raise ValueError("No suitable text column found in the CSV!")

    # ✅ Wrap inside dlt.resource to give a proper name
    @dlt.resource(name="documents")
    def docs():
        for i, row in df.iterrows():
            yield {
                "id": int(i),
                "title": row.get("title", "Untitled"),
                "text": row.get(text_col, ""),
                "domain": row.get("domain", ""),
                "url": row.get("url", ""),
            }

    return docs()

# ---------------------------
# 4️⃣ Define DLT pipeline
# ---------------------------
pipeline = dlt.pipeline(
    pipeline_name="pikaplexity_ingest_pipeline",
    destination=dlt.destinations.filesystem(bucket_url="./data/dlt_output"),
    dataset_name="pikaplexity_data",
)

# ---------------------------
# 5️⃣ Main ETL + Upsert to Qdrant
# ---------------------------
if __name__ == "__main__":
    print("⚡ Starting DLT pipeline for Qdrant ingestion...")

    # Run DLT pipeline (back up to ./data/dlt_output)
    load_info = pipeline.run(knowledge_source())
    print(f"✅ DLT pipeline executed successfully. Loads: {load_info.loads_ids}")

    # Upload to Qdrant
    print("🚀 Uploading documents to Qdrant...")
    for doc in tqdm(knowledge_source(), desc="Uploading"):
        vector = model.encode(doc["text"]).tolist()
        payload = {
            "chunk": doc["text"],
            "title": doc["title"],
            "domain": doc.get("domain", ""),
            "url": doc.get("url", ""),
        }
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{"id": doc["id"], "vector": vector, "payload": payload}],
        )

    print("✅ Ingestion complete! Documents successfully loaded into Qdrant via DLT.")
