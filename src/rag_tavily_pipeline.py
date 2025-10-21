import os
import faiss
import numpy as np
import pickle
import requests
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ───────────────────────────────
# 1️⃣ Setup
# ───────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

BASE_URL = "https://api.groq.com/openai/v1"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}"}
ANSWER_MODEL = "llama-3.3-70b-versatile"

# Thresholds
FAISS_TOP_K = 8                # increased for higher recall
FAISS_MIN_SCORE = 0.35
MIN_GOOD_HITS = 2
RELEVANCE_THRESHOLD = 0.40     # loosened a bit

# Console colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# ───────────────────────────────
# 2️⃣ Load FAISS retriever
# ───────────────────────────────
print("🔄 Loading FAISS retriever...")
index_path = "notebooks/data/processed/retrieval_system_index.faiss"
meta_path = "notebooks/data/processed/retrieval_system_metadata.pkl"
chunks_path = "notebooks/data/processed/hybrid_chunks.pkl"

if not os.path.exists(index_path):
    raise FileNotFoundError(f"❌ FAISS index not found at: {index_path}")

index = faiss.read_index(index_path)
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)
with open(chunks_path, "rb") as f:
    chunks_data = pickle.load(f)

chunks = chunks_data.get("chunks", [])
print(f"✅ Retriever loaded successfully! Chunks: {len(chunks)}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ───────────────────────────────
# 3️⃣ Helper functions
# ───────────────────────────────
def query_groq(model, prompt, max_tokens=350):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    r = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def tavily_search(query):
    """Search using Tavily API."""
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced"}
    try:
        res = requests.post(url, json=payload, timeout=15)
        data = res.json()
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["content"]
        else:
            return "No relevant results found online."
    except Exception as e:
        return f"Tavily search failed: {e}"


def retrieve_candidates(question, k=FAISS_TOP_K):
    q = embedding_model.encode([question]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    out = []
    for s, idx in zip(scores[0], ids[0]):
        if 0 <= idx < len(chunks):
            out.append({"score": float(s), "chunk": chunks[idx]})
    return out


def build_context(cands):
    return "\n\n---\n\n".join([c["chunk"]["text"] for c in cands])


def should_use_faiss(cands, min_score=FAISS_MIN_SCORE, min_hits=MIN_GOOD_HITS):
    good = [c for c in cands if c["score"] >= min_score]
    return len(good) >= min_hits


def context_is_relevant(question, candidates, threshold=RELEVANCE_THRESHOLD):
    """Compute semantic cosine similarity between question and retrieved chunks."""
    if not candidates:
        return False
    q_emb = embedding_model.encode([question])[0]
    sims = []
    for c in candidates:
        text = c["chunk"]["text"]
        c_emb = embedding_model.encode([text])[0]
        sim = dot(q_emb, c_emb) / (norm(q_emb) * norm(c_emb))
        sims.append(sim)
    avg_sim = float(np.mean(sims))
    print(f"🧭 Semantic similarity with top-{len(candidates)} chunks: {avg_sim:.2f}")
    if avg_sim < threshold:
        print(f"⚠️  Context semantically irrelevant (avg cos sim < {threshold}) → using Tavily.")
        return False
    return True

# ───────────────────────────────
# 4️⃣ RAG + Tavily logic
# ───────────────────────────────
def answer_with_rag_or_tavily(question):
    cands = retrieve_candidates(question, FAISS_TOP_K)

    print(f"\n🔍 Retrieval for: {question}")
    if not cands:
        print("   (no FAISS hits)")
    else:
        for i, c in enumerate(cands, 1):
            meta = c["chunk"].get("metadata", {})
            title = meta.get("title", "Unknown")
            domain = meta.get("domain", "Unknown")
            print(f"   {i}. score={c['score']:.2f}  [{domain}] {title}")

    # ✴️ Heuristic override for known KB terms
    key_terms = ["photosynthesis", "pythagorean", "climate", "cell", "energy", "geometry"]
    context_text = build_context(cands)
    force_use_faiss = any(term in context_text.lower() or term in question.lower() for term in key_terms)

    use_faiss = (should_use_faiss(cands) and context_is_relevant(question, cands)) or force_use_faiss

    if use_faiss:
        strong = sorted([c for c in cands if c["score"] >= FAISS_MIN_SCORE],
                        key=lambda x: x["score"], reverse=True)
        context = build_context(strong[:5])  # include more chunks
        prompt = (
            "Use the following knowledge base context to answer succinctly. "
            "If the context seems partially related, infer a short, factual answer using general knowledge. "
            "Only say 'I don't know' if the context is completely unrelated.\n\n"
            f"### Context\n{context}\n\n"
            f"### Question\n{question}\n\n### Answer"
        )
        answer = query_groq(ANSWER_MODEL, prompt)

        weak_signals = ["don't know", "no information", "unavailable", "cannot answer"]
        if any(sig in answer.lower() for sig in weak_signals):
            print("⚠️  RAG model explicitly said it doesn't know → switching to Tavily.")
            web = tavily_search(question)
            return "FAISS→TAVILY", web

        return "FAISS", answer

    # Fallback: FAISS irrelevant or empty
    web = tavily_search(question)
    return "TAVILY", web


# ───────────────────────────────
# 5️⃣ Main
# ───────────────────────────────
def main():
    queries = [
        "Who is the current UN Secretary-General?",
        "Explain photosynthesis in simple terms.",
        "What are the main causes of climate change?",
        "What is the Pythagorean theorem?"
    ]

    for q in queries:
        mode, ans = answer_with_rag_or_tavily(q)
        color = GREEN if mode == "FAISS" else YELLOW if mode == "FAISS→TAVILY" else BLUE
        label = (
            "Answer (FAISS)"
            if mode == "FAISS"
            else "Answer (FAISS→TAVILY)" if mode == "FAISS→TAVILY"
            else "Answer (Tavily)"
        )
        print(f"\n{color}{label}:{RESET}\n{ans}\n" + "─" * 70)


if __name__ == "__main__":
    main()
