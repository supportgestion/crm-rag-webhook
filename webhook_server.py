import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import requests
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from supabase import create_client, Client

app = FastAPI(title="Zoho CRM to RAG Webhook (Cloud Lightweight Edition)")

# --- 1. CONFIGURATION DES CLÉS API ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. INITIALISATION DES CLIENTS CLOUD ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("crm-notes")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# --- FONCTION D'EMBEDDING VIA API (LÉGÈRE) ---
def get_embedding(text: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()


# --- 3. MODÈLES DE DONNÉES ---
class ChatModel(BaseModel):
    question: str
    n_results: Optional[int] = 3

class NoteModel(BaseModel):
    note_id: Optional[str] = None
    client: Optional[str] = "Client Inconnu"
    note: str
    date: Optional[str] = "2026-08-11"


# --- 4. ENDPOINTS ---
@app.post("/zoho-webhook")
async def recevoir_note_zoho(data: NoteModel):
    note_id = data.note_id or f"zoho_{pd.Timestamp.now().timestamp()}"
    if not data.note:
        return {"status": "error", "message": "Note vide"}

    resume = data.note[:100] + "..." if len(data.note) > 100 else data.note

    # 1. Sauvegarde dans Supabase (SQL)
    supabase.table("incidents").upsert({
        "id": str(note_id),
        "client": data.client,
        "date_note": str(data.date),
        "categorie": "Général",
        "resume_probleme": resume
    }).execute()

    # 2. Vectorisation et sauvegarde dans Pinecone via API
    vector = get_embedding(data.note)
    if isinstance(vector, list):
        index.upsert(
            vectors=[{
                "id": str(note_id),
                "values": vector,
                "metadata": {"client": data.client, "date": str(data.date), "texte": data.note}
            }]
        )
    
    return {"status": "success", "message": "Note synchronisée dans le Cloud"}


@app.post("/chat-rag")
async def chat_rag(data: ChatModel):
    if not data.question:
        return {"status": "error", "message": "Question vide"}

    # 1. Transformer la question en vecteur via API
    question_vector = get_embedding(data.question)
    if not isinstance(question_vector, list):
        return {"status": "error", "message": "Erreur lors de la vectorisation de la question"}

    # 2. Chercher dans Pinecone
    search_results = index.query(
        vector=question_vector,
        top_k=data.n_results,
        include_metadata=True
    )

    matches = search_results.get("matches", [])
    if not matches:
        return {
            "reponse": "Information non disponible dans la base de données CRM.",
            "sources": []
        }

    # 3. Formater le contexte
    contexte_elements = []
    sources = []
    for match in matches:
        meta = match.get("metadata", {})
        client = meta.get("client", "Inconnu")
        date = meta.get("date", "")
        texte = meta.get("texte", "")
        
        contexte_elements.append(f"[Client: {client} | Date: {date}]\nNote: {texte}")
        sources.append({"doc": texte, "metadata": {"client": client, "date": date}})

    contexte = "\n\n---\n\n".join(contexte_elements)

    # 4. Interroger l'IA Hugging Face
    user_prompt = f"""Tu es un assistant B2B factuel.

Consignes strictes :
1. Réponds à la question en t'appuyant uniquement sur les notes CRM ci-dessous.
2. N'invente aucune donnée. Si l'information est absente, réponds exactement : "Information non disponible dans la base de données CRM."

Notes CRM disponibles :
{contexte}

Question : {data.question}"""

    try:
        client_hf = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
        response = client_hf.chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=300,
            temperature=0.1,
        )

        return {
            "status": "success",
            "question": data.question,
            "reponse": response.choices[0].message.content,
            "sources": sources
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
