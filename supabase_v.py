import faiss
import pickle
import numpy as np
import psycopg2
import uuid

# === RUTAS A LOS ARCHIVOS ===
FAISS_INDEX_PATH = "vector_store/index.faiss"
PKL_PATH = "vector_store/index.pkl"

# === CARGAR EMBEDDINGS Y METADATOS ===
index = faiss.read_index(FAISS_INDEX_PATH)

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
    texts = data["texts"]
    metadatas = data["metadatas"]

# === CONEXIÓN A SUPABASE ===
conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com",
    port=6543,
    database="postgres",
    user="postgres.xewxagocwjpepdmwzzxj",
    password="Aliaga280199",  # cambia esto por la contraseña real
    sslmode="require"
)
cursor = conn.cursor()

# === INSERTAR EN LA TABLA documentos_vector ===
for i in range(index.ntotal):
    vector = index.reconstruct(i).tolist()
    texto = texts[i]
    metadata = metadatas[i]

    cursor.execute(
        """
        INSERT INTO documentos_vector (id, content, embedding, source, chunk_index)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            str(uuid.uuid4()),
            texto,
            vector,
            metadata.get("source", ""),
            metadata.get("chunk_index", 0)
        )
    )

conn.commit()
cursor.close()
conn.close()

print("✅ Migración completada con éxito a documentos_vector")