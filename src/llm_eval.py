from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv(dotenv_path="/workspaces/Rag-Knowledge-Assiatant/.env")

# Initialize Groq client (OpenAI-compatible)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Sample evaluation questions
queries = [
    "What is physics?",
    "Define photosynthesis",
    "Explain DNA structure",
    "What is artificial intelligence?",
    "How does machine learning work?"
]

# Evaluate model responses
print("🚀 LLM Evaluation (Groq - LLaMA 3.3 70B Versatile)")
print("=" * 60)

for q in queries:
    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert explainer for a RAG-based knowledge assistant."},
            {"role": "user", "content": q}
        ],
        temperature=0.7,
        max_tokens=300
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content.strip()
    print(f"\n🧠 Question: {q}")
    print(f"💬 Answer: {answer[:400]}")
    print(f"⏱️ Response time: {elapsed:.2f} seconds")

print("\n✅ LLM evaluation complete.")
