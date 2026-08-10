import json
import os
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
import pandas as pd

app = FastAPI(title="Zoho CRM to RAG Webhook")

# Embedding ONNX ultra-léger de ChromaDB
default_ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes", embedding_function=default_ef
)


# 1. Endpoint : Ingestion depuis Zoho CRM
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

    data = {
        "categorie": "Général",
        "resume_probleme": texte_note[:100] + "..."
        if len(texte_note) > 100
        else texte_note,
    }

    # SQLite
    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            client TEXT,
            date_note TEXT,
            categorie TEXT,
            resume_probleme TEXT
        )
    """
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO incidents (id, client, date_note, categorie, resume_probleme)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            str(note_id),
            client_name,
            str(date_note),
            data["categorie"],
            data["resume_probleme"],
        ),
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


# 2. Endpoint : Interroger le RAG
@app.post("/query-rag")
async def query_rag(request: Request):
    payload = await request.json()
    question = payload.get("question", "")
    n_results = payload.get("n_results", 3)

    if not question:
        return {"status": "error", "message": "Question vide"}

    results = collection.query(query_texts=[question], n_results=n_results)

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_metadatas = results.get("metadatas", [[]])[0]
    retrieved_ids = results.get("ids", [[]])[0]

    return {
        "status": "success",
        "question": question,
        "results": [
            {"id": doc_id, "doc": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                retrieved_ids, retrieved_docs, retrieved_metadatas
            )
        ],
    }


# 3. Endpoint : Modifier une note
@app.post("/update-note")
async def modifier_note(request: Request):
    payload = await request.json()
    note_id = payload.get("note_id")
    nouveau_texte = payload.get("note")
    client_name = payload.get("client", "Client Inconnu")

    if not note_id or not nouveau_texte:
        return {
            "status": "error",
            "message": "Les champs 'note_id' et 'note' sont requis",
        }

    # Mise à jour dans ChromaDB
    collection.update(
        ids=[str(note_id)],
        documents=[nouveau_texte],
        metadatas=[
            {"client": client_name, "date": str(pd.Timestamp.now().date())}
        ],
    )

    # Mise à jour dans SQLite
    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE incidents SET resume_probleme = ? WHERE id = ?",
        (nouveau_texte[:100], str(note_id)),
    )
    conn.commit()
    conn.close()

    print(f"✏️ Note {note_id} mise à jour.")
    return {
        "status": "success",
        "message": f"Note {note_id} mise à jour avec succès.",
    }


# 4. Endpoint : Supprimer une note
@app.post("/delete-note")
async def effacer_note(request: Request):
    payload = await request.json()
    note_id = payload.get("note_id")

    if not note_id:
        return {"status": "error", "message": "Le champ 'note_id' est requis"}

    # Suppression dans ChromaDB
    collection.delete(ids=[str(note_id)])

    # Suppression dans SQLite
    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents WHERE id = ?", (str(note_id),))
    conn.commit()
    conn.close()

    print(f"🗑️ Note {note_id} supprimée.")
    return {
        "status": "success",
        "message": f"Note {note_id} supprimée avec succès.",
    }
