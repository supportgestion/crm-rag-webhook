import json
import sqlite3
import pandas as pd
import chromadb
from fastapi import FastAPI, Request
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI(title="Zoho CRM to RAG Webhook")

# 1. Chargement de ChromaDB
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class HFEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embedder.encode(input).tolist()
    def name(self) -> str:
        return "hf_multilingual_minilm"

chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes",
    embedding_function=HFEmbeddingFunction()
)

# 2. Endpoint Webhook pour Zoho
@app.post("/zoho-webhook")
async def recevoir_note_zoho(request: Request):
    payload = await request.json()
    
    client_name = payload.get("client", "Client Inconnu")
    date_note = payload.get("date", "2026-08-10")
    texte_note = payload.get("note", "")
    note_id = payload.get("note_id", f"zoho_{pd.Timestamp.now().timestamp()}")

    if not texte_note:
        return {"status": "error", "message": "Note vide"}

    print(f"\n📩 Note reçue de Zoho pour : {client_name}")

    prompt_json = f"""
    Analyse cette note CRM et extrait le problème principal au format JSON STRICT.
    {{"categorie": "nom court", "resume_probleme": "une phrase"}}
    Note : {texte_note}
    """
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_json}])
    
    try:
        data = json.loads(response['message']['content'])
    except Exception:
        data = {"categorie": "Général", "resume_probleme": "Analyse manuelle requise"}

    # SQL
    conn = sqlite3.connect('analytics_crm.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO incidents (client, date_note, categorie, resume_probleme)
        VALUES (?, ?, ?, ?)
    ''', (client_name, str(date_note), data['categorie'], data['resume_probleme']))
    conn.commit()
    conn.close()

    # ChromaDB
    collection.add(
        documents=[texte_note],
        metadatas=[{"client": client_name, "date": str(date_note)}],
        ids=[str(note_id)]
    )

    print("✅ Note Zoho synchronisée dans le RAG et SQLite !")
    return {"status": "success"}