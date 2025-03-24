# Archivo: app.py (backend)
import os
import docx2txt
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict
import json
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Reemplázalo con los dominios permitidos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar API Key de OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# Cargar base de datos de FAISS
vector_store_path = "vector_store"
embeddings = OpenAIEmbeddings()

if os.path.exists(vector_store_path):
    print("Cargando base de datos vectorial existente...")
    vector_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("No se encontró base de datos vectorial. Debes generar los embeddings primero.")
    exit()

# Configurar el modelo y la búsqueda semántica
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model_name="gpt-4o", streaming=True)

# Clase para manejar conexiones WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Función para generar respuestas por partes
async def generate_response_stream(question, websocket):
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    Responde a la pregunta basada en el siguiente contexto:
    
    {context}
    
    Pregunta: {question}
    """
    
    response = ""
    async for chunk in llm.astream(prompt):
        if hasattr(chunk, 'content'):
            content = chunk.content
            response += content
            await manager.send_message(json.dumps({"type": "chunk", "content": content}), websocket)
    
    # Enviar mensaje completo al final
    await manager.send_message(json.dumps({"type": "complete", "content": response}), websocket)
    return response

# Crear la aplicación FastAPI
app = FastAPI()

# Endpoint REST tradicional
class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Pregunta no proporcionada")
    
    response = ask_chatbot(request.question)
    return {"response": response.content}

# Endpoint WebSocket para chat en tiempo real
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "question":
                question = message_data.get("content", "")
                if question:
                    await generate_response_stream(question, websocket)
                else:
                    await manager.send_message(json.dumps({"type": "error", "content": "Pregunta vacía"}), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            await manager.send_message(json.dumps({"type": "error", "content": str(e)}), websocket)
        except:
            pass
        manager.disconnect(websocket)

# Función tradicional para preguntas no streaming
def ask_chatbot(question):
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
    Responde a la pregunta basada en el siguiente contexto:
    
    {context}
    
    Pregunta: {question}
    """
    return llm.invoke(prompt)

if __name__ == "__main__":
    import uvicorn
    print("Iniciando API FastAPI con soporte WebSocket...")
    uvicorn.run(app, host="0.0.0.0", port=8000)