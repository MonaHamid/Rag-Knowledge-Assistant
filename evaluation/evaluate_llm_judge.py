import os
import json
import csv
import time
import re
import requests
import faiss
import numpy as np
import pickle
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ───────────────────────────────
#  1️⃣ SETUP
# ───────────────────────────────
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Define models
ANSWER_MODEL = "llama-3.1-8b-instant"       # For generating answers
JUDGE_MODEL = "llama-3.3-70b-versatile"     # For evaluating answers

# ───────────────────────────────
#  2️⃣ LOAD YOUR RETRIEVER (FAISS)
# ───────────────────────────────
print("🔄 Loading FAISS retriever...")

BASE_DIR = pathlib.Path("/workspaces/Rag-Knowledge-Assiatant")
DATA_DIR = BASE_DIR / "notebooks" / "data" / "processed"

index_path = DATA_DIR / "retrieval_system_index.faiss"
metadata_path = DATA_DIR / "retrieval_system_metadata.pkl"

if not index_path.exists():
    raise FileNotFoundError(f"❌ FAISS index not found at: {index_path}")
if not metadata_path.exists():
    raise FileNotFoundError(f"❌ Metadata not found at: {metadata_path}")

index = faiss.read_index(str(index_path))
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Retriever loaded successfully!")


def retrieve_context(question: str, k=3) -> str:
    """Retrieve top-k context chunks for a question using FAISS index."""
    query_vector = embedding_model.encode([question]).astype("float32")
    distances, indices = index.search(query_vector, k)
    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(metadata):
            retrieved_chunks.append(metadata[idx]["text"])
    return "\n".join(retrieved_chunks)


# ───────────────────────────────
#  3️⃣ HELPER FUNCTIONS
# ───────────────────────────────
def query_groq(model, prompt, max_tokens=512):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    r = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def generate_answer(question, context, mode):
    """Generate answer using different prompting strategies."""
    if mode == "zero_shot":
        prompt = f"Q: {question}\nA:"
    elif mode == "few_shot":
        examples = (
            "Q: What is photosynthesis?\n"
            "A: Photosynthesis is the process where plants convert sunlight into energy.\n\n"
            "Q: What is DNA?\n"
            "A: DNA stores genetic information in living organisms.\n\n"
        )
        prompt = f"{examples}Q: {question}\nA:"
    elif mode == "cot":
        prompt = f"Think step by step before answering.\nQ: {question}\nA:"
    elif mode == "rag":
        prompt = f"Use the following context to answer:\n{context}\n\nQ: {question}\nA:"
    else:
        raise ValueError("Unknown prompting mode")

    return query_groq(ANSWER_MODEL, prompt)


def judge_answer(question, context, answer):
    """Judge model rates factuality, relevance, and fluency."""
    judge_prompt = f"""
You are an expert LLM evaluator.

### Question
{question}

### Context (from RAG)
{context}

### Model Answer
{answer}

Rate on a 0–1 scale:
- Factuality: correctness based on the provided context
- Relevance: how well it uses the context and stays on topic
- Fluency: clarity and readability of language

Return only JSON:
{{"factuality": 0.xx, "relevance": 0.xx, "fluency": 0.xx}}
"""
    result = query_groq(JUDGE_MODEL, judge_prompt)

    # 🔍 Extract JSON block even if surrounded by explanations
    match = re.search(r'\{.*?\}', result, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print("⚠️ Judge parsing failed. Raw output:", result)
    return {"factuality": 0, "relevance": 0, "fluency": 0}


# ───────────────────────────────
#  4️⃣ MAIN EVALUATION LOOP
# ───────────────────────────────
def main():
    test_questions_path = DATA_DIR / "test_questions.json"
    if not test_questions_path.exists():
        raise FileNotFoundError(f"❌ Test questions file not found at: {test_questions_path}")

    with open(test_questions_path) as f:
        questions = json.load(f)

    modes = ["zero_shot", "few_shot", "cot", "rag"]
    results = []

    for q in questions:
        question = q["question"]
        context = retrieve_context(question)
        print(f"\n🔍 Evaluating: {question}")

        for mode in modes:
            print(f"   ➜ {mode} ...", end="", flush=True)
            start = time.time()
            try:
                answer = generate_answer(question, context, mode)
                scores = judge_answer(question, context, answer)
                elapsed = round(time.time() - start, 2)
                results.append({
                    "question": question,
                    "mode": mode,
                    "factuality": scores["factuality"],
                    "relevance": scores["relevance"],
                    "fluency": scores["fluency"],
                    "time_s": elapsed
                })
                print(" ✅")
            except Exception as e:
                print(f" ❌ ({e})")
            time.sleep(2)  # ⏳ prevent Groq rate limit (429 errors)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_csv = "results/llm_eval_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "mode", "factuality", "relevance", "fluency", "time_s"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊 Evaluation complete! Results saved to {output_csv}\n")

    # Print summary table
    summary = {}
    for r in results:
        mode = r["mode"]
        if mode not in summary:
            summary[mode] = {"factuality": [], "relevance": [], "fluency": []}
        for k in ["factuality", "relevance", "fluency"]:
            summary[mode][k].append(r[k])

    print("🧠 Summary (averaged across all questions):")
    print("-" * 60)
    print(f"{'Prompt Strategy':<15} {'Factuality':<12} {'Relevance':<12} {'Fluency':<12}")
    print("-" * 60)
    for mode, vals in summary.items():
        f_mean = np.mean(vals["factuality"])
        r_mean = np.mean(vals["relevance"])
        fl_mean = np.mean(vals["fluency"])
        print(f"{mode:<15} {f_mean:<12.2f} {r_mean:<12.2f} {fl_mean:<12.2f}")
    print("-" * 60)

    # ───────────────────────────────
    #  5️⃣ OPTIONAL: VISUALIZATION
    # ───────────────────────────────
    try:
        df = pd.DataFrame(results)
        summary_df = df.groupby("mode")[["factuality", "relevance", "fluency"]].mean()
        summary_df.plot(kind="bar", figsize=(8, 5), rot=0)
        plt.title("LLM Evaluation: Zero-shot vs Few-shot vs CoT vs RAG")
        plt.ylabel("Score (0–1)")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Visualization skipped: {e}")


if __name__ == "__main__":
    main()
