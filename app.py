import streamlit as st
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# 2️⃣ Initialize model + Qdrant
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qdrant():
    client = QdrantClient(
        url="https://6eb0b696-8d2d-49fb-8c75-d9d178cf0b62.eu-west-1-0.aws.cloud.qdrant.io",   # 🔁 Replace with your real cluster URL
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uP4MxoWXxr6NWVlG2Kiey6vXsQafdSSDyWz8GbPEzmw"             # 🔁 Replace with your real API key
    )
    return client

model = load_model()
qdrant_client = load_qdrant()
COLLECTION_NAME = "pikaplexity_docs_cloud"

# -----------------------------
# 3️⃣ Qdrant Search Helper
# -----------------------------
def qdrant_search(query, model, client, collection=COLLECTION_NAME, limit=8):
    query_vec = model.encode([query])[0].tolist()

    results = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=limit,
        with_payload=True
    )

    hits = []
    for r in results.points:
        payload = r.payload or {}
        hits.append({
            "score": r.score,
            "chunk": payload.get("chunk", ""),
            "domain": payload.get("domain") or payload.get("source") or "Unknown domain",
            "title": payload.get("title") or payload.get("topic") or payload.get("file") or "Untitled"
        })
    return hits

# -----------------------------
# 4️⃣ Hybrid RAG + Tavily Logic
# -----------------------------
def rag_or_tavily(query, model, client, threshold=0.4):
    results = qdrant_search(query, model, client)
    if not results:
        return "No results found.", [], "🧠 Qdrant"

    avg_sim = np.mean([r["score"] for r in results])
    sources = [f"{r['domain']}: {r['title']}" for r in results]
    context = " ".join([r["chunk"] for r in results])

    client_groq = Groq(api_key=GROQ_API_KEY)
    llm_answer = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are PikaPlexity ⚡, a friendly AI powered by Groq. Use only the given context. If context is irrelevant, reply with 'not found'."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    answer = llm_answer.choices[0].message.content

    # 🌐 Fallback to Tavily if weak similarity or 'not found'
    if avg_sim < threshold or "not found" in answer.lower():
        tavily_resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "include_answer": True}
        )
        if tavily_resp.status_code == 200:
            data = tavily_resp.json()
            tavily_answer = data.get("answer", "No answer found.")
            citations = data.get("results", [])
            sources = [f"[{i+1}]({r.get('url')}) {r.get('title')}" for i, r in enumerate(citations)]
            return tavily_answer, sources, "🌐 Tavily"
        else:
            return "Tavily search failed.", [], "🌐 Tavily"

    return answer, sources, "🧠 Qdrant"

# -----------------------------
# 5️⃣ Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="PikaPlexity ⚡", page_icon="⚡", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #0e0e0e;
        color: #f5f5f5;
        font-family: "Poppins", sans-serif;
    }
    .title-container {
        text-align: center;
        margin-top: 2em;
    }
    .title-container img {
        width: 120px;
        animation: float 3s ease-in-out infinite, pulseGlow 2.5s ease-in-out infinite;
        filter: drop-shadow(0 0 20px #FFD93D);
    }
    @keyframes float {
        0% { transform: translatey(0px); }
        50% { transform: translatey(-10px); }
        100% { transform: translatey(0px); }
    }
    @keyframes pulseGlow {
        0%, 100% { filter: drop-shadow(0 0 15px #FFD93D); }
        50% { filter: drop-shadow(0 0 35px #FFEA00); }
    }
    .title-container h1 {
        font-size: 3.2em;
        font-weight: 800;
        color: #FFD93D;
        text-shadow: 0 0 20px #FFD93D, 0 0 40px #FFEA00;
    }
    .title-container p {
        color: #ccc;
        font-size: 1.1em;
        margin-top: -0.5em;
    }
    .search-box input {
        font-size: 1.1em !important;
        border-radius: 50px !important;
        padding: 0.9em 1.5em !important;
        background-color: #2b2b2b !important;
        color: white !important;
        border: 2px solid #FFD93D !important;
        box-shadow: 0 0 10px #FFD93D60;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FFD93D, #FFEA00);
        color: black;
        border: none;
        border-radius: 50px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #ffea00;
        transform: scale(1.05);
        box-shadow: 0 0 15px #FFD93D;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 3em;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 6️⃣ UI (Transparent Pikachu)
# -----------------------------
st.markdown("""
<div class="title-container">
    <img src="https://cdn-icons-png.flaticon.com/512/188/188987.png" alt="Pikachu Icon">
    <h1>⚡ PikaPlexity Chatbot</h1>
    <p>Powered by Groq + Llama 3.3 70B — hybrid Qdrant & Tavily search</p>
</div>
""", unsafe_allow_html=True)

query = st.text_input("Ask Pikachu anything:", placeholder="Who is the president of the United States?", key="search", label_visibility="collapsed")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🌿 Photosynthesis"):
        query = "How does photosynthesis work?"
with col2:
    if st.button("📐 Pythagorean theorem"):
        query = "What is the Pythagorean theorem?"
with col3:
    if st.button("🌍 UN Secretary-General"):
        query = "Who is the current UN Secretary-General?"

# -----------------------------
# 7️⃣ Run RAG + Tavily
# -----------------------------
if query:
    with st.spinner("⚡ Pikachu is thinking..."):
        answer, sources, mode = rag_or_tavily(query, model, qdrant_client)

    st.markdown(f"## 💡 {mode} Answer")
    st.write(answer)

    if sources:
        st.markdown("## 📚 Sources")
        for i, s in enumerate(sources):
            st.markdown(f"{i+1}. {s}")

# -----------------------------
# 8️⃣ Footer
# -----------------------------
st.markdown(
    "<div class='footer'>⚡ Built with ❤️ by Fareeda — PikaPlexity powered by Groq & Qdrant 🧠</div>",
    unsafe_allow_html=True
)
