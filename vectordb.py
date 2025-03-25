import os
import re
import uuid
import traceback
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from supabase import create_client
from unstructured.partition.auto import partition
from tiktoken import encoding_for_model

# === CONFIGURACIÓN ===
ROOT_DIR = "Poliza_Chat"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536

# ✅ Supabase
SUPABASE_URL = "https://xewxagocwjpepdmwzzxj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhld3hhZ29jd2pwZXBkbXd6enhqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI4ODE3NDcsImV4cCI6MjA1ODQ1Nzc0N30.RC5Im_aHvlYFxsGmhzFJMjFnPtGtPjET1qawGXfzifM"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ OpenAI
client = OpenAI()
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# === FUNCIONES ===

def chunk_text(text, max_tokens=1200):
    enc = encoding_for_model(EMBEDDING_MODEL)
    paragraphs = text.split("\n")
    chunks, current_chunk = [], []

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        candidate = "\n".join(current_chunk + [paragraph])
        if len(enc.encode(candidate)) <= max_tokens:
            current_chunk.append(paragraph)
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [paragraph]

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def process_docx(path):
    print(f"📄 Procesando: {path}")
    try:
        elements = partition(filename=path)
        text = "\n".join([el.text for el in elements if el.text])

        if not text.strip():
            print("❌ Sin texto útil.")
            return []

        chunks = chunk_text(text)

        return [{
            "text": chunk,
            "source": path,
            "chunk_index": i
        } for i, chunk in enumerate(chunks)]

    except Exception as e:
        print(f"❌ Error procesando {path}: {e}")
        print(traceback.format_exc())
        return []

# === RECORRER SOLO ARCHIVOS .DOCX ===

all_docs = []

for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith(".docx"):
            path = os.path.join(root, file)
            all_docs.extend(process_docx(path))

print(f"🧠 Generando e insertando embeddings para {len(all_docs)} chunks...")

# === INSERTAR EN SUPABASE ===

for doc in all_docs:
    embedding = embedding_model.embed_query(doc["text"])
    supabase.table("documentos_vector").insert({
        "id": str(uuid.uuid4()),
        "content": doc["text"],
        "embedding": embedding,
        "source": doc["source"],
        "chunk_index": doc["chunk_index"]
    }).execute()

print("✅ ¡Carga completa a Supabase solo con .docx! 🎉")
