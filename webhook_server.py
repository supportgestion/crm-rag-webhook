import json
import os
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import pandas as pd

app = FastAPI(title="Zoho CRM to RAG Webhook")

# Embedding ONNX ultra-léger de ChromaDB
default_ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes", embedding_function=default_ef
)


# Modèles de données pour Swagger (Affiche les champs d'écriture)
class QueryModel(BaseModel):
    question: str
    n_results: Optional[int] = 3


class NoteModel(BaseModel):
    note_id: Optional[str] = None
    client: Optional[str] = "Client Inconnu"
    note: str
    date: Optional[str] = "2026-08-10"


class DeleteModel(BaseModel):
    note_id: str


# 1. Ingestion depuis Zoho CRM
@app.post("/zoho-webhook")
async def recevoir_note_zoho(data: NoteModel):
    note_id = data.note_id or f"zoho_{pd.Timestamp.now().timestamp()}"

    if not data.note:
        return {"status": "error", "message": "Note vide"}

    print(f"\n📩 Note reçue de Zoho pour : {data.client}")

    resume = data.note[:100] + "..." if len(data.note) > 100 else data.note

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
        (str(note_id), data.client, str(data.date), "Général", resume),
    )
    conn.commit()
    conn.close()

    # ChromaDB
    collection.add(
        documents=[data.note],
        metadatas=[{"client": data.client, "date": str(data.date)}],
        ids=[str(note_id)],
    )

    print("✅ Note Zoho synchronisée dans le RAG et SQLite !")
    return {"status": "success"}


# 2. Interroger le RAG
@app.post("/query-rag")
async def query_rag(data: QueryModel):
    if not data.question:
        return {"status": "error", "message": "Question vide"}

    results = collection.query(query_texts=[data.question], n_results=data.n_results)

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_metadatas = results.get("metadatas", [[]])[0]
    retrieved_ids = results.get("ids", [[]])[0]

    return {
        "status": "success",
        "question": data.question,
        "results": [
            {"id": doc_id, "doc": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                retrieved_ids, retrieved_docs, retrieved_metadatas
            )
        ],
    }


# 3. Modifier une note
@app.post("/update-note")
async def modifier_note(data: NoteModel):
    if not data.note_id or not data.note:
        return {"status": "error", "message": "note_id et note sont requis"}

    collection.update(
        ids=[str(data.note_id)],
        documents=[data.note],
        metadatas=[{"client": data.client, "date": str(pd.Timestamp.now().date())}],
    )

    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE incidents SET resume_probleme = ? WHERE id = ?",
        (data.note[:100], str(data.note_id)),
    )
    conn.commit()
    conn.close()

    return {"status": "success", "message": f"Note {data.note_id} mise à jour"}


# 4. Supprimer une note
@app.post("/delete-note")
async def effacer_note(data: DeleteModel):
    collection.delete(ids=[str(data.note_id)])

    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents WHERE id = ?", (str(data.note_id),))
    conn.commit()
    conn.close()

    return {"status": "success", "message": f"Note {data.note_id} supprimée"}
