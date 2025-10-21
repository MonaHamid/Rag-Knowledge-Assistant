from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment
load_dotenv(dotenv_path="/workspaces/Rag-Knowledge-Assiatant/.env")

# Initialize Groq client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Models to test
models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct"
]

# Prompt types
prompts = {
    "zero_shot": "You are a knowledgeable science tutor. Answer clearly and concisely.",
    "few_shot": "You are a science tutor using examples to guide your answers.",
    "cot": "You are a reasoning tutor. Think step by step before answering clearly."
}

# Few-shot examples
few_shot_examples = [
    {"role": "user", "content": "What is physics?"},
    {"role": "assistant", "content": "Physics is the study of matter, motion, and energy that governs the universe."},
    {"role": "user", "content": "What is chemistry?"},
    {"role": "assistant", "content": "Chemistry is the branch of science focused on substances, their reactions, and properties."}
]

# Test queries
queries = [
    "What is photosynthesis?",
    "Explain DNA structure",
    "How does machine learning work?",
    "What is artificial intelligence?",
    "What is quantum mechanics?"
]

print("🚀 MULTI-MODEL + PROMPT EVALUATION (Groq)")
print("=" * 75)

results = []

for model in models:
    print(f"\n🤖 Model: {model}")
    print("-" * 60)

    for style, system_prompt in prompts.items():
        print(f"\n🧠 Prompting Style: {style.upper()}")
        print("-" * 40)

        for query in queries:
            start = time.time()

            # Base message structure
            messages = [{"role": "system", "content": system_prompt}]

            if style == "few_shot":
                messages += few_shot_examples

            # CoT: Add reasoning hint
            if style == "cot":
                query = f"{query}\nThink step by step before answering."

            messages.append({"role": "user", "content": query})

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300
                )
                elapsed = time.time() - start
                answer = response.choices[0].message.content.strip()

                print(f"\nQ: {query}")
                print(f"A ({style}): {answer[:350]}...")
                print(f"⏱️ Time: {elapsed:.2f}s")

                results.append({
                    "model": model,
                    "prompt_style": style,
                    "query": query,
                    "answer": answer,
                    "time": elapsed
                })

            except Exception as e:
                print(f"⚠️ Error for {model} ({style}): {e}")

print("\n✅ Multi-model LLM evaluation complete.")