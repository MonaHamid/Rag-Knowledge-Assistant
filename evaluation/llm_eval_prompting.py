from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment
load_dotenv(dotenv_path="/workspaces/Rag-Knowledge-Assiatant/.env")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Sample queries
queries = [
    "What is photosynthesis?",
    "Explain DNA structure",
    "How does machine learning work?"
]

# Few-shot examples
few_shot_examples = [
    {"role": "user", "content": "What is physics?"},
    {"role": "assistant", "content": "Physics is the scientific study of matter, motion, energy, and the fundamental forces of nature."},
    {"role": "user", "content": "What is chemistry?"},
    {"role": "assistant", "content": "Chemistry is the branch of science concerned with the properties, composition, and behavior of substances."}
]

# Prompt templates
prompts = {
    "zero_shot": "You are a knowledgeable science tutor. Answer clearly and concisely.",
    "few_shot": "You are a science tutor using examples to guide your answers.",
    "cot": "You are a reasoning tutor. Think step by step before answering the question clearly."
}

models = ["llama-3.3-70b-versatile"]

print("🚀 LLM Evaluation: Zero-shot, Few-shot, and Chain-of-Thought (CoT)")
print("=" * 70)

for model in models:
    print(f"\n🤖 Model: {model}")
    print("-" * 60)

    for style, system_prompt in prompts.items():
        print(f"\n🧠 Prompting Style: {style.upper()}")
        print("-" * 40)

        for query in queries:
            start = time.time()
            messages = [{"role": "system", "content": system_prompt}]

            if style == "few_shot":
                messages += few_shot_examples  # include demo pairs

            messages.append({"role": "user", "content": query})

            # Add CoT instruction to user query for chain-of-thought
            if style == "cot":
                messages[-1]["content"] = f"{query}\nThink step by step before answering."

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            elapsed = time.time() - start
            answer = response.choices[0].message.content.strip()

            print(f"\nQ: {query}")
            print(f"A ({style}): {answer[:400]}...")
            print(f"⏱️ Time: {elapsed:.2f}s")

print("\n Prompting evaluation complete.")