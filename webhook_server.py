"""
Zoho CRM -> RAG (Pinecone + Supabase + Hugging Face)

Principes de ce fichier :
  1. Aucune connexion externe a l'import -> l'app demarre toujours, meme si une
     cle est manquante. Le port est binde, et /health dit ce qui ne va pas.
  2. Les erreurs remontent en vrais codes HTTP (502 / 503), jamais deguisees
     en "success". Un echec doit etre visible immediatement.
  3. Une seule source de verite pour la config, verifiee au demarrage.
"""

import os
import logging
from typing import Optional, List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("crm-rag")

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "crm-notes")

# Modele multilingue : indispensable, tes notes et tes questions sont en francais.
# Dimension 384 -> compatible avec ton index Pinecone existant.
# ATTENTION : si tu as deja indexe des notes avec all-MiniLM-L6-v2 (anglais),
# il faut les revectoriser, sinon les resultats seront incoherents.
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Modele de generation. Qwen2.5-Coder est specialise code : mauvais choix pour
# rediger en francais. Verifie la disponibilite du modele choisi sur
# https://huggingface.co/docs/inference-providers avant de deployer.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# L'ancien hote api-inference.huggingface.co est mort (DNS supprime).
# Format actuel : router.huggingface.co/hf-inference/models/{model}/pipeline/{task}
HF_EMBED_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"
    "/pipeline/feature-extraction"
)

REQUEST_TIMEOUT = 30
NO_INFO = "Information non disponible dans la base de donnees CRM."

app = FastAPI(title="Zoho CRM to RAG Webhook")


# ---------------------------------------------------------------------------
# 2. CLIENTS EN INITIALISATION PARESSEUSE
# ---------------------------------------------------------------------------
# Instancier Pinecone / Supabase au niveau du module fait planter l'import si une
# cle manque -> le process meurt avant de binder le port -> 502 cote proxy, sans
# aucun message exploitable. On differe donc la connexion au premier usage.

_pinecone_index = None
_supabase = None


def get_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise HTTPException(503, "PINECONE_API_KEY absente de l'environnement.")
        from pinecone import Pinecone

        _pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)
        log.info("Pinecone connecte sur l'index '%s'", PINECONE_INDEX)
    return _pinecone_index


def get_supabase():
    global _supabase
    if _supabase is None:
        if not (SUPABASE_URL and SUPABASE_KEY):
            raise HTTPException(503, "SUPABASE_URL ou SUPABASE_KEY absente.")
        from supabase import create_client

        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase connecte")
    return _supabase


# ---------------------------------------------------------------------------
# 3. EMBEDDINGS
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """Retourne un vecteur 1D. Leve une HTTPException explicite en cas d'echec.

    On ne retourne jamais None : un embedding manquant doit interrompre la
    requete, pas produire silencieusement un resultat vide.
    """
    if not HF_TOKEN:
        raise HTTPException(503, "HF_TOKEN absent de l'environnement.")

    try:
        r = requests.post(
            HF_EMBED_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": text},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        log.error("Reseau HF injoignable: %s", e)
        raise HTTPException(502, f"Hugging Face injoignable : {e}")

    if r.status_code != 200:
        log.error("HF %s -> %s", r.status_code, r.text[:400])
        raise HTTPException(502, f"Hugging Face a renvoye {r.status_code}: {r.text[:200]}")

    vec = r.json()

    # Selon le modele et le payload, HF renvoie soit [0.1, ...] soit [[0.1, ...]].
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = vec[0]

    if not (isinstance(vec, list) and vec and isinstance(vec[0], (int, float))):
        raise HTTPException(502, f"Format d'embedding inattendu : {str(vec)[:200]}")

    return vec


# ---------------------------------------------------------------------------
# 4. SCHEMAS
# ---------------------------------------------------------------------------

class NoteModel(BaseModel):
    note_id: Optional[str] = None
    client: Optional[str] = "Client Inconnu"
    note: str
    date: Optional[str] = None


class ChatModel(BaseModel):
    question: str
    n_results: Optional[int] = 3
    client: Optional[str] = None       # filtre optionnel sur un client precis
    min_score: Optional[float] = 0.30  # seuil anti-hallucination (cosine)


# ---------------------------------------------------------------------------
# 5. DIAGNOSTIC
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/health")
def health():
    """Repond toujours 200, meme mal configure : c'est le point d'entree du
    diagnostic. Affiche quelles cles sont presentes SANS jamais les exposer."""
    return {
        "status": "ok",
        "config": {
            "HF_TOKEN": bool(HF_TOKEN),
            "PINECONE_API_KEY": bool(PINECONE_API_KEY),
            "SUPABASE_URL": bool(SUPABASE_URL),
            "SUPABASE_KEY": bool(SUPABASE_KEY),
        },
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": MODEL_ID,
        "pinecone_index": PINECONE_INDEX,
    }


@app.get("/debug/pinecone")
def debug_pinecone():
    """Combien de vecteurs contient reellement l'index ?
    Si total_vector_count vaut 0, le RAG ne peut rien trouver : le probleme est
    a l'ingestion, pas a la recherche."""
    try:
        return get_index().describe_index_stats()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Pinecone: {e}")


@app.get("/debug/embedding")
def debug_embedding(text: str = "test de vectorisation"):
    """Teste la chaine d'embedding seule, sans Pinecone ni LLM."""
    vec = get_embedding(text)
    return {"dimension": len(vec), "apercu": vec[:5]}


# ---------------------------------------------------------------------------
# 6. INGESTION
# ---------------------------------------------------------------------------

@app.post("/zoho-webhook")
def recevoir_note_zoho(data: NoteModel):
    note = (data.note or "").strip()
    if not note:
        raise HTTPException(400, "Note vide")

    from datetime import date as _date

    note_id = data.note_id or f"zoho_{_date.today()}_{abs(hash(note)) % 10**8}"
    date_note = data.date or str(_date.today())
    resume = note[:200] + ("..." if len(note) > 200 else "")

    # Pinecone d'abord : si la vectorisation echoue, on ne veut PAS d'une ligne
    # SQL orpheline qui laisse croire que la note est interrogeable.
    vector = get_embedding(note)
    try:
        get_index().upsert(
            vectors=[{
                "id": str(note_id),
                "values": vector,
                "metadata": {
                    "client": data.client,
                    "date": date_note,
                    "texte": note,
                },
            }]
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Pinecone upsert: %s", e)
        raise HTTPException(502, f"Echec de l'indexation Pinecone : {e}")

    try:
        get_supabase().table("incidents").upsert({
            "id": str(note_id),
            "client": data.client,
            "date_note": date_note,
            "categorie": "General",
            "resume_probleme": resume,
        }).execute()
    except HTTPException:
        raise
    except Exception as e:
        # L'essentiel (le RAG) est en place : on signale sans tout faire echouer.
        log.error("Supabase upsert: %s", e)
        return {"status": "partial", "note_id": note_id,
                "message": f"Indexe dans Pinecone mais echec Supabase : {e}"}

    return {"status": "success", "note_id": note_id}


# ---------------------------------------------------------------------------
# 7. RAG
# ---------------------------------------------------------------------------

@app.post("/chat-rag")
def chat_rag(data: ChatModel):
    question = (data.question or "").strip()
    if not question:
        raise HTTPException(400, "Question vide")

    question_vector = get_embedding(question)

    kwargs = {
        "vector": question_vector,
        "top_k": data.n_results or 3,
        "include_metadata": True,
    }
    if data.client:
        kwargs["filter"] = {"client": {"$eq": data.client}}

    try:
        res = get_index().query(**kwargs)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Pinecone query: %s", e)
        raise HTTPException(502, f"Recherche Pinecone impossible : {e}")

    # Le SDK Pinecone renvoie un objet, pas un dict : on gere les deux formes
    # plutot que de supposer une version precise du client.
    matches = getattr(res, "matches", None)
    if matches is None:
        matches = res["matches"] if isinstance(res, dict) else []

    # Seuil de similarite : sans ca, Pinecone renvoie toujours les top_k plus
    # proches, meme s'ils n'ont aucun rapport avec la question. C'est la
    # premiere cause d'hallucination dans un RAG.
    retenus = []
    for m in matches:
        score = getattr(m, "score", None)
        if score is None and isinstance(m, dict):
            score = m.get("score")
        if score is None or score >= (data.min_score or 0):
            retenus.append(m)

    if not retenus:
        return {"status": "success", "question": question,
                "reponse": NO_INFO, "sources": []}

    contexte_elements, sources = [], []
    for m in retenus:
        meta = getattr(m, "metadata", None)
        if meta is None:
            meta = m.get("metadata", {}) if isinstance(m, dict) else {}
        meta = meta or {}
        client = meta.get("client", "Inconnu")
        date = meta.get("date", "")
        texte = meta.get("texte", "")
        contexte_elements.append(f"[Client: {client} | Date: {date}]\nNote: {texte}")
        sources.append({
            "doc": texte,
            "score": getattr(m, "score", None),
            "metadata": {"client": client, "date": date},
        })

    contexte = "\n\n---\n\n".join(contexte_elements)

    system_prompt = (
        "Tu es un assistant CRM factuel pour le suivi client B2B. "
        "Tu reponds exclusivement a partir des notes CRM fournies. "
        "Tu n'utilises aucune connaissance externe et tu n'extrapoles jamais. "
        f'Si les notes ne contiennent pas la reponse, tu ecris exactement : "{NO_INFO}" '
        "Tu cites le nom du client et la date des notes utilisees. "
        "Tu reponds en francais."
    )
    user_prompt = f"Notes CRM :\n\n{contexte}\n\nQuestion : {question}"

    try:
        from huggingface_hub import InferenceClient

        hf = InferenceClient(api_key=HF_TOKEN)
        completion = hf.chat_completion(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        reponse = completion.choices[0].message.content
    except Exception as e:
        # On remonte une vraie erreur : c'est ce qui permet de la diagnostiquer.
        log.error("LLM HF: %s", e)
        raise HTTPException(502, f"Generation impossible : {e}")

    return {"status": "success", "question": question,
            "reponse": reponse, "sources": sources}
