import os
import uuid
import traceback
from openai import OpenAI
from unstructured.partition.auto import partition
from tiktoken import encoding_for_model
import pickle
import faiss
import numpy as np

# === CONFIGURACIÓN ===
ROOT_DIR = "Poliza_Chat"  # <-- ajusta si es necesario
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536
client = OpenAI()

# === FUNCIONES ===

def chunk_text(text, max_tokens=1500):
    enc = encoding_for_model("text-embedding-3-small")
    words = text.split()
    chunks, current = [], []

    for word in words:
        current.append(word)
        if len(enc.encode(" ".join(current))) > max_tokens:
            chunks.append(" ".join(current[:-1]))
            current = [word]

    if current:
        chunks.append(" ".join(current))

    return chunks

def process_document(path, index, texts, metadatas):
    print(f"📄 Procesando: {path}")
    try:
        elements = partition(filename=path)
        text = "\n".join([el.text for el in elements if el.text])

        if not text.strip():
            print("⚠️ Sin texto útil.")
            return

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            embedding = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk
            ).data[0].embedding

            vector = np.array(embedding, dtype="float32").reshape(1, -1)
            index.add(vector)

            texts.append(chunk)
            metadatas.append({
                "source": path,
                "chunk_index": i
            })

    except Exception:
        print(f"❌ Error: {path}")
        print(traceback.format_exc())

# === INDEXADO ===
index = faiss.IndexFlatL2(DIMENSION)
texts, metadatas = [], []

for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith((".pdf", ".docx")):
            process_document(os.path.join(root, file), index, texts, metadatas)

# === GUARDAR FAISS + METADATA ===
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/index.pkl", "wb") as f:
    pickle.dump({"texts": texts, "metadatas": metadatas}, f)

print("✅ ¡Indexado y guardado en vector_store!")
