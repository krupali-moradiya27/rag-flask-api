# generate_response.py
import groq
from retriever import retrieve_similar
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=api_key)

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "Respond only in English.",
    "hi": "उत्तर केवल हिंदी में दीजिए।",
    "ur": "جواب صرف اردو میں دیں۔"
}

def generate_response(user_query, user_health_data, language):
    if language not in SUPPORTED_LANGUAGES:
        return f"❌ Language '{language}' is not supported. Choose from: English (en), Hindi (hi), Urdu (ur)."

    # Retrieve relevant context
    retrieved_context = retrieve_similar(user_query, top_k=3)

    # Format user health data
    user_health_info = "\n".join(
        [f"{k}: {v}" for k, v in user_health_data.items() if v]
    )

    prompt = f"""
You are a helpful and professional mental health assistant.

Use ONLY the 'Retrieved Knowledge' and 'User Health Data' provided below to answer the user's query.
Be short (2–4 sentences), clear, and supportive.

{SUPPORTED_LANGUAGES[language]}

---
User Query:
{user_query}

User Health Data:
{user_health_info}

Retrieved Knowledge:
{retrieved_context}
"""

    # Call LLM
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a multilingual mental health counseling assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
        top_p=0.8,
    )

    return response.choices[0].message.content.strip()
