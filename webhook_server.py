import json
import os
import sqlite3
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from fastapi import FastAPI, Request
import pandas as pd
import requests

app = FastAPI(title="Zoho CRM to RAG Webhook")

# API Hugging Face (Acheminement via le nouveau Router HF)
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HFEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input: Documents) -> Embeddings:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": input, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        res = response.json()

        # Formatage de sécurité pour ChromaDB
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], float):
            return [res]
        return res

    def name(self) -> str:
        return "hf_multilingual_minilm"


chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes", embedding_function=HFEmbeddingFunction()
)


# Endpoint Webhook pour Zoho
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

    # Catégorisation
    data = {
        "categorie": "Général",
        "resume_probleme": texte_note[:100] + "..."
        if len(texte_note) > 100
        else texte_note,
    }

    # SQL
    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client TEXT,
            date_note TEXT,
            categorie TEXT,
            resume_probleme TEXT
        )
    """
    )
    cursor.execute(
        """
        INSERT INTO incidents (client, date_note, categorie, resume_probleme)
        VALUES (?, ?, ?, ?)
    """,
        (client_name, str(date_note), data["categorie"], data["resume_probleme"]),
    )
    conn.commit()
    conn.close()

    # ChromaDB
    collection.add(
        documents=[texte_note],
        metadatas=[{"client": client_name, "date": str(date_note)}],
        ids=[str(note_id)],
    )

    print("✅ Note Zoho synchronisée dans le RAG et SQLite !")
    return {"status": "success"}
