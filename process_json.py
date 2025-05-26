import os
import json
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

data_folders = ['english_data', 'hindi_data', 'urdu_data']
documents = []
chunk_metadata = []

for folder in data_folders:
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for entry in data:
                        if isinstance(entry, dict) and "question" in entry and "answer" in entry:
                            combined_text = f"Q: {entry['question']} A: {entry['answer']}"
                            documents.append(combined_text)
                except Exception as e:
                    print(f"Failed to read {file_name}: {e}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = []
for i, doc in enumerate(documents):
    split_chunks = text_splitter.split_text(doc)
    chunks.extend(split_chunks)
    chunk_metadata.extend([{"doc_index": i, "text": chunk} for chunk in split_chunks])

with open("chunk_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)

# Embedding + FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, "mental_health_index.faiss")
print("✅ FAISS index created.")