import json
import os
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
import pandas as pd

app = FastAPI(title="Zoho CRM to RAG Webhook")

# Utilisation de la fonction d'embedding par défaut / ONNX ultra-léger de ChromaDB
# (Ne charge aucun gros modèle en RAM et évite les erreurs HTTP 400 HF)
default_ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes", embedding_function=default_ef
)


# Endpoint Webhook pour Zoho CRM
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

    # Résumé / Catégorisation
    data = {
        "categorie": "Général",
        "resume_probleme": texte_note[:100] + "..."
        if len(texte_note) > 100
        else texte_note,
    }

    # Stockage SQLite
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

    # Indexation Vectorielle ChromaDB
    collection.add(
        documents=[texte_note],
        metadatas=[{"client": client_name, "date": str(date_note)}],
        ids=[str(note_id)],
    )

    print("✅ Note Zoho synchronisée dans le RAG et SQLite !")
    return {"status": "success"}
