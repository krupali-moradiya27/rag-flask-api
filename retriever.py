import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

# Load FAISS index
index_path = os.path.join(os.path.dirname(__file__), "mental_health_index.faiss")
index = faiss.read_index(index_path)

# Load all JSON files in English, Hindi, and Urdu folders
data_folders = [
    os.path.join(os.path.dirname(__file__), "english_data"),
    os.path.join(os.path.dirname(__file__), "hindi_data"),
    os.path.join(os.path.dirname(__file__), "urdu_data"),
]

all_data = []

for folder in data_folders:
    if os.path.exists(folder):
        for file_name in os.listdir(folder):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        all_data.extend(data)
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON: {file_path}")

# Load chunk metadata
chunk_file = os.path.join(os.path.dirname(__file__), "chunk_metadata.json")
with open(chunk_file, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to retrieve top-k similar documents
def retrieve_similar(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(chunk_metadata):
            doc_index = chunk_metadata[i]["doc_index"]
            if 0 <= doc_index < len(all_data):
                item = all_data[doc_index]
                question = item.get("question")
                answer = item.get("answer")
                if question and answer:
                    results.append(f"Q: {question}\nA: {answer}")
                else:
                    results.append(chunk_metadata[i].get("text", ""))
            else:
                results.append(chunk_metadata[i].get("text", ""))
    return "\n\n".join(results)

print("✅ FAISS retriever is ready!")
