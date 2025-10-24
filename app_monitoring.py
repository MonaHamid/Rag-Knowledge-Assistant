# app.py — PikaPlexity RAG Chatbot with Monitoring (Streamlit + Prometheus Pushgateway)
# ---------------------------------------------------------------------------------
# Requirements (add these to requirements.txt):
# streamlit
# sentence-transformers
# qdrant-client
# groq
# requests
# python-dotenv
# numpy
# pandas
# altair
# prometheus-client
# (optional) python-json-logger
# ---------------------------------------------------------------------------------

import os
import time
import json
import datetime as dt
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client import QdrantClient
from prometheus_client import generate_latest 

# Prometheus
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, push_to_gateway

# -----------------------------
# 0️⃣ Constants & Paths
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FEEDBACK_CSV = DATA_DIR / "user_feedback.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"

COLLECTION_NAME = "pikaplexity_docs_cloud"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# -----------------------------
# 1️⃣ Secure Secrets Loading
# -----------------------------
# Use Streamlit secrets first; fall back to environment variables when running locally.
get_secret = lambda k, d=None: st.secrets.get(k, os.getenv(k, d))

GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY", "")
QDRANT_URL = get_secret("QDRANT_URL", "")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY", "")

PROM_USERNAME = get_secret("PROM_USERNAME", "")
PROM_API_KEY = get_secret("PROM_API_KEY", "")

# Optional for Grafana/Prometheus Pushgateway
PUSHGATEWAY_URL = get_secret("PUSHGATEWAY_URL", "")  # e.g. http://<host>:9091
PROM_JOB_NAME = get_secret("PROM_JOB_NAME", "pikaplexity_streamlit")
PROM_INSTANCE = get_secret("PROM_INSTANCE", os.getenv("RENDER_INSTANCE_ID", "streamlit-cloud"))

if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY in secrets; the app will not be able to call Groq.")

# -----------------------------
# 2️⃣ Clients
# -----------------------------
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL and QDRANT_API_KEY else None

# -----------------------------
# 3️⃣ Load Embedding Model (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(DEFAULT_MODEL)

model = load_model()

# -----------------------------
# 4️⃣ Prometheus Metrics Setup
# -----------------------------
registry = CollectorRegistry()

MET_QUERY_TOTAL = Counter(
    "pikaplexity_queries_total",
    "Total queries processed",
    ["mode"],
    registry=registry,
)

MET_FEEDBACK_TOTAL = Counter(
    "pikaplexity_feedback_total",
    "Feedback label counts",
    ["feedback"],
    registry=registry,
)

MET_RESPONSE_TIME = Histogram(
    "pikaplexity_response_time_seconds",
    "Time to answer a query",
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
    registry=registry,
)

MET_SIMILARITY = Histogram(
    "pikaplexity_qdrant_avg_similarity",
    "Average similarity from Qdrant hits",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry,
)

MET_LIVE_USERS = Gauge(
    "pikaplexity_live_sessions",
    "Concurrent sessions (approx)",
    registry=registry,
)

# tiny helper for pushgateway

ddef push_metrics():
    """Push metrics to Grafana Cloud remote_write endpoint via HTTP POST."""
    if not (PUSHGATEWAY_URL and PROM_API_KEY and PROM_USERNAME):
        return

    try:
        # Basic Auth: username is stack ID, password is API key
        auth_str = f"{PROM_USERNAME}:{PROM_API_KEY}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()

        # Collect metrics data from registry
        data = []
        for metric in registry.collect():
            for sample in metric.samples:
                line = f"{sample.name}{{"
                labels = ",".join([f'{k}="{v}"' for k, v in sample.labels.items()])
                line += labels + f"}} {sample.value}\n"
                data.append(line)
        metrics_data = "".join(data)

        headers = {
            "Authorization": f"Basic {b64_auth}",
            "Content-Type": "text/plain",
        }

        resp = requests.post(PUSHGATEWAY_URL, data=metrics_data, headers=headers)
        if resp.status_code != 200:
            st.session_state["prometheus_error"] = f"Grafana push failed [{resp.status_code}]: {resp.text}"
    except Exception as e:
        st.session_state["prometheus_error"] = str(e)

# -----------------------------
# 5️⃣ Qdrant Search Helper
# -----------------------------

def qdrant_search(query: str, model, client: QdrantClient, collection: str = COLLECTION_NAME, limit: int = 8):
    if not client:
        return []
    query_vec = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=limit,
        with_payload=True,
    )

    hits = []
    for r in results.points:
        payload = r.payload or {}
        hits.append({
            "score": r.score,
            "chunk": payload.get("chunk", ""),
            "domain": payload.get("domain") or "Knowledge Base",
            "title": payload.get("title") or "Untitled",
            "url": payload.get("source") or "",
        })
    return hits

# -----------------------------
# 6️⃣ Hybrid RAG + Tavily Logic
# -----------------------------

def rag_or_tavily(query: str, model, client: QdrantClient, threshold: float = 0.4):
    results = qdrant_search(query, model, client)
    if not results:
        return "No results found in Qdrant.", [], "🧠 Qdrant", None

    avg_sim = float(np.mean([r["score"] for r in results])) if results else 0.0

    # Record similarity metric
    MET_SIMILARITY.observe(avg_sim)

    context = "\n".join([r["chunk"] for r in results])[:6000]  # safety truncate

    if not client_groq:
        return "LLM client not configured.", results, "🧠 Qdrant", avg_sim

    llm_answer = client_groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PikaPlexity ⚡, a friendly AI powered by Groq. "
                    "Use only the provided context. If answer not found, reply with 'not found'."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    answer = llm_answer.choices[0].message.content

    # 🌐 Fallback to Tavily
    if avg_sim < threshold or "not found" in answer.lower():
        if not TAVILY_API_KEY:
            return answer, results, "🧠 Qdrant", avg_sim
        tavily_resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "include_answer": True},
            timeout=20,
        )
        if tavily_resp.status_code == 200:
            data = tavily_resp.json()
            tavily_answer = data.get("answer", "No answer found.")
            citations = data.get("results", [])
            sources = [f"[{i+1}]({r.get('url')}) {r.get('title')}" for i, r in enumerate(citations)]
            return tavily_answer, sources, "🌐 Tavily", avg_sim
        else:
            return "Tavily search failed.", [], "🌐 Tavily", avg_sim

    return answer, results, "🧠 Qdrant", avg_sim

# -----------------------------
# 7️⃣ Streamlit UI Styling & Layout
# -----------------------------
st.set_page_config(page_title="PikaPlexity ⚡", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #0e0e0e; color: #f5f5f5; font-family: "Poppins", sans-serif; }
    .title-container { text-align: center; margin-top: 0.5em; }
    .title-container img { width: 100px; filter: drop-shadow(0 0 20px #FFD93D); }
    .title-container h1 { font-size: 2.6em; font-weight: 800; color: #FFD93D; text-shadow: 0 0 20px #FFD93D, 0 0 40px #FFEA00; }
    .stTextInput>div>div>input { font-size: 1.05em !important; border-radius: 50px !important; padding: 0.8em 1.2em !important; background-color: #2b2b2b !important; color: white !important; border: 2px solid #FFD93D !important; box-shadow: 0 0 10px #FFD93D60; }
    .stButton>button { background: linear-gradient(90deg, #FFD93D, #FFEA00); color: black; border: none; border-radius: 50px; font-weight: 600; }
    .stButton>button:hover { background: #ffea00; transform: scale(1.03); box-shadow: 0 0 15px #FFD93D; }
    .footer { text-align: center; color: #888; margin-top: 1em; font-size: 0.9em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
<div class="title-container">
    <img src="https://cdn-icons-png.flaticon.com/512/188/188987.png" alt="Pikachu Icon">
    <h1>⚡ PikaPlexity Chatbot</h1>
    <p>Groq + Llama 3.3 70B — Hybrid Qdrant & Tavily Search</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar — Monitoring switcher
st.sidebar.title("⚡ Monitoring")
dashboard_option = st.sidebar.selectbox(
    "View:", ["Chat", "Feedback Summary", "Usage Trends", "Source Distribution", "Latency"]
)

# -----------------------------
# 8️⃣ Chat Flow (+ metrics, logging)
# -----------------------------

if dashboard_option == "Chat":
    query = st.text_input(
        "Ask Pikachu anything:",
        placeholder="Who is the current UN Secretary-General?",
        key="search",
    )

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

    if query:
        start = time.time()
        with st.spinner("⚡ Pikachu is thinking..."):
            answer, results, mode, avg_sim = rag_or_tavily(query, model, qdrant_client)
        latency = round(time.time() - start, 3)

        # Metrics
        MET_QUERY_TOTAL.labels(mode=mode).inc()
        MET_RESPONSE_TIME.observe(latency)
        push_metrics()

        # Persist interaction row
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inter_row = {
            "timestamp": now,
            "query": query,
            "mode": mode,
            "latency_s": latency,
            "avg_qdrant_sim": avg_sim if avg_sim is not None else np.nan,
        }
        if not INTERACTIONS_CSV.exists():
            pd.DataFrame([inter_row]).to_csv(INTERACTIONS_CSV, index=False)
        else:
            pd.DataFrame([inter_row]).to_csv(INTERACTIONS_CSV, mode="a", header=False, index=False)

        # Render result
        st.markdown(f"## 💡 {mode} Answer")
        st.write(answer)

        if mode == "🧠 Qdrant" and isinstance(results, list) and results:
            st.markdown("## 📚 Sources")
            for i, r in enumerate(results):
                url = r.get("url", "")
                title = r.get("title", "Untitled")
                domain = r.get("domain", "Knowledge Base")
                if isinstance(url, str) and url.startswith("http"):
                    st.markdown(f"{i+1}. [{domain} — {title}]({url})")
                else:
                    st.markdown(f"{i+1}. **{domain}** — {title}")

        elif mode == "🌐 Tavily" and isinstance(results, list) and results:
            st.markdown("## 🌎 Web Sources")
            for i, s in enumerate(results):
                st.markdown(f"{i+1}. {s}")

        # Feedback
        st.subheader("💭 Was this answer helpful?")
        feedback = st.radio(
            "Please rate the quality of the answer:",
            options=["👍 Yes, it was helpful", "👎 No, it wasn’t accurate", "🤔 Partially useful"],
            horizontal=True,
            key="feedback",
        )

        if st.button("Submit Feedback"):
            MET_FEEDBACK_TOTAL.labels(feedback=feedback).inc()
            push_metrics()

            fb_row = {
                "timestamp": now,
                "query": query,
                "mode": mode,
                "feedback": feedback,
                "latency_s": latency,
                "avg_qdrant_sim": avg_sim if avg_sim is not None else np.nan,
            }
            if not FEEDBACK_CSV.exists():
                pd.DataFrame([fb_row]).to_csv(FEEDBACK_CSV, index=False)
            else:
                pd.DataFrame([fb_row]).to_csv(FEEDBACK_CSV, mode="a", header=False, index=False)
            st.success("✅ Thank you for your feedback! Pikachu appreciates it ⚡")

# -----------------------------
# 9️⃣ Monitoring Dashboards (Streamlit)
# -----------------------------

if dashboard_option == "Feedback Summary":
    try:
        df = pd.read_csv(FEEDBACK_CSV)
        st.subheader("📊 Feedback Summary")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("feedback", sort=["👍 Yes, it was helpful", "🤔 Partially useful", "👎 No, it wasn’t accurate"]),
            y="count()",
            color="feedback",
            tooltip=["feedback", "count()"],
        )
        st.altair_chart(chart, use_container_width=True)
        st.metric("Total Feedback Received", len(df))
        st.dataframe(df.tail(10))
    except FileNotFoundError:
        st.info("No feedback yet. Try asking some questions first!")

if dashboard_option == "Usage Trends":
    try:
        df = pd.read_csv(INTERACTIONS_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # ensure dtype
        df["date"] = df["timestamp"].dt.date

        st.subheader("📈 User Activity Over Time")
        chart = alt.Chart(df).mark_line(point=True).encode(
            x="date:T",
            y="count()",
            tooltip=["date:T", "count()"],
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("⏱️ Average Latency by Day")
        by_day = df.groupby("date")["latency_s"].mean().reset_index()
        chart2 = alt.Chart(by_day).mark_bar().encode(
            x="date:T",
            y=alt.Y("latency_s:Q", title="Avg seconds"),
            tooltip=["date:T", alt.Tooltip("latency_s:Q", format=",.2f")],
        )
        st.altair_chart(chart2, use_container_width=True)
    except FileNotFoundError:
        st.info("No usage data yet.")

if dashboard_option == "Source Distribution":
    try:
        df = pd.read_csv(INTERACTIONS_CSV)
        st.subheader("🧠 Source Distribution (Qdrant vs Tavily)")
        chart = alt.Chart(df).mark_arc(innerRadius=60).encode(
            theta="count()",
            color="mode",
            tooltip=["mode", "count()"],
        )
        st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError:
        st.info("No source data yet.")

if dashboard_option == "Latency":
    try:
        df = pd.read_csv(INTERACTIONS_CSV)
        st.subheader("⏱️ Latency Histogram")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("latency_s:Q", bin=alt.Bin(maxbins=20), title="Latency (s)"),
            y="count()",
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        st.altair_chart(chart, use_container_width=True)

        st.metric("Mean Latency (s)", round(df["latency_s"].mean(), 3))
        st.metric("P95 Latency (s)", round(df["latency_s"].quantile(0.95), 3))
        st.metric("Max Latency (s)", round(df["latency_s"].max(), 3))
    except FileNotFoundError:
        st.info("No latency data yet.")

# -----------------------------
# 🔟 Footer & Prometheus status
# -----------------------------
with st.container():
    prom_msg = ""
    if PUSHGATEWAY_URL:
        prom_msg = f" | Prometheus Pushgateway: configured"
        if st.session_state.get("prometheus_error"):
            prom_msg += f" (last error: {st.session_state['prometheus_error']})"
    st.markdown(
        f"<div class='footer'>⚡ Built with ❤️ by Fareeda — PikaPlexity powered by Groq, Qdrant & Tavily 🧠{prom_msg}</div>",
        unsafe_allow_html=True,
    )
