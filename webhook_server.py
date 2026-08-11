import json
import os
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from huggingface_hub import InferenceClient

app = FastAPI(title="Zoho CRM to RAG Webhook with Hugging Face")

# Embedding ONNX local ultra-léger
default_ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes", embedding_function=default_ef
)

# Token Hugging Face récupéré depuis l'environnement Render
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Modèle souverain Mistral-7B via Hugging Face Inference API
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Modèles Pydantic
class QueryModel(BaseModel):
    question: str
    n_results: Optional[int] = 3


class ChatModel(BaseModel):
    question: str
    n_results: Optional[int] = 3


class NoteModel(BaseModel):
    note_id: Optional[str] = None
    client: Optional[str] = "Client Inconnu"
    note: str
    date: Optional[str] = "2026-08-10"


class DeleteModel(BaseModel):
    note_id: str


# 1. Ingestion Zoho CRM
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


# 2. Recherche vectorielle brute (Search RAG)
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


# 3. Chatbot intelligent (Mistral 7B via Hugging Face Chat Completion)
@app.post("/chat-rag")
async def chat_rag(data: ChatModel):
    if not data.question:
        return {"status": "error", "message": "Question vide"}

    # Recherche dans ChromaDB
    results = collection.query(query_texts=[data.question], n_results=data.n_results)
    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_metadatas = results.get("metadatas", [[]])[0]

    if not retrieved_docs:
        return {
            "reponse": "Information non disponible dans la base de données CRM.",
            "sources": [],
        }

    # Formatage du contexte
    contexte_elements = []
    for doc, meta in zip(retrieved_docs, retrieved_metadatas):
        client = meta.get("client", "Inconnu")
        date = meta.get("date", "")
        contexte_elements.append(f"[Client: {client} | Date: {date}]\nNote: {doc}")

    contexte = "\n\n---\n\n".join(contexte_elements)

    # Prompt d'ancrage strict anti-hallucination
    system_prompt = f"""Tu es un assistant B2B factuel pour l'entreprise.
Consignes de sécurité :
1. Réponds à la question uniquement avec les notes CRM fournies ci-dessous.
2. Si l'information n'est pas dans le texte, réponds STRICTEMENT : "Information non disponible dans la base de données CRM."
3. N'invente aucune donnée.

Contexte CRM :
{contexte}"""

    if not HF_TOKEN:
        return {
            "status": "error",
            "message": "Variable HF_TOKEN non configurée sur Render.",
        }

    try:
        client_hf = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

        # Appel au format Chat Completion
        response = client_hf.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.question},
            ],
            max_tokens=300,
            temperature=0.1,
        )

        reponse_ia = response.choices[0].message.content

        return {
            "status": "success",
            "question": data.question,
            "reponse": reponse_ia,
            "sources": [
                {"doc": doc, "metadata": meta}
                for doc, meta in zip(retrieved_docs, retrieved_metadatas)
            ],
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# 4. Modifier une note
@app.post("/update-note")
async def modifier_note(data: NoteModel):
    if not data.note_id or not data.note:
        return {"status": "error", "message": "note_id et note requis"}

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


# 5. Supprimer une note
@app.post("/delete-note")
async def effacer_note(data: DeleteModel):
    collection.delete(ids=[str(data.note_id)])

    conn = sqlite3.connect("analytics_crm.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents WHERE id = ?", (str(data.note_id),))
    conn.commit()
    conn.close()

    return {"status": "success", "message": f"Note {data.note_id} supprimée"}
