"""
Zoho CRM -> RAG + Analytique (Pinecone + Supabase + Hugging Face)

Deux chemins de donnees distincts, chacun pour ce qu'il sait faire :

  PINECONE (semantique) -> questions ouvertes
      "qu'est-ce qu'on a dit sur le format XML ?"
      La similarite vectorielle est le bon outil. Le decoupage en morceaux
      thematiques est essentiel : un vecteur unique pour 7000 caracteres est
      une moyenne floue de 20 sujets et ne matche precisement rien.

  SUPABASE (structure) -> comptages, filtres, tris, dashboards
      "le probleme le plus remonte en aout", "le dernier RDV de Jean-Pierre"
      Ce sont des agregations SQL. Les faire passer par un LLM sur des
      extraits vectoriels produit des chiffres inventes. Postgres compte, le
      LLM ne compte pas.

Le LLM n'intervient que pour deux choses : extraire du structure a
l'ingestion, et rediger une reponse a partir d'un contexte fourni. Il ne
calcule jamais.
"""

import os
import re
import json
import logging
from datetime import date as _date
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Query
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

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

try:
    MIN_SCORE_DEFAULT = float(os.environ.get("MIN_SCORE", "0.15"))
except ValueError:
    log.warning("MIN_SCORE illisible, repli sur 0.15")
    MIN_SCORE_DEFAULT = 0.15

# Liste FERMEE de categories. Sans contrainte, le LLM ecrit "Integration
# caisse" sur une note et "Probleme Popina" sur la suivante : le GROUP BY
# renvoie alors 30 categories a 1 occurrence et ne veut plus rien dire.
# Modifiable par la variable CATEGORIES dans Railway (separees par ';').
CATEGORIES_DEFAUT = [
    "Facturation & import",
    "Integration caisse",
    "Produits & fournisseurs",
    "Recettes & marges",
    "Droits & acces",
    "Parametrage etablissement",
    "Performance & donnees",
    "Formation & prise en main",
    "Demande d'evolution",
    "Autre",
]
CATEGORIES = [
    c.strip() for c in os.environ.get("CATEGORIES", ";".join(CATEGORIES_DEFAUT)).split(";")
    if c.strip()
]

HF_EMBED_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"
    "/pipeline/feature-extraction"
)

REQUEST_TIMEOUT = 30
CHUNK_MAX = 900          # caracteres par morceau vectorise
NO_INFO = "Information non disponible dans la base de donnees CRM."

app = FastAPI(title="Zoho CRM to RAG + Analytique")


# ---------------------------------------------------------------------------
# 2. CLIENTS PARESSEUX
# ---------------------------------------------------------------------------
# Instancier au niveau du module fait planter l'import si une cle manque : le
# process meurt avant de binder le port, et le proxy renvoie un 502 muet.

_pinecone_index = None
_supabase = None


def get_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise HTTPException(503, "PINECONE_API_KEY absente.")
        from pinecone import Pinecone

        _pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)
    return _pinecone_index


def get_supabase():
    global _supabase
    if _supabase is None:
        if not (SUPABASE_URL and SUPABASE_KEY):
            raise HTTPException(503, "SUPABASE_URL ou SUPABASE_KEY absente.")
        from supabase import create_client

        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


# ---------------------------------------------------------------------------
# 3. NETTOYAGE DU TEXTE ZOHO
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")


def nettoyer_note(texte: str) -> Dict[str, str]:
    """Separe l'URL du contenu et normalise les retours a la ligne.

    Zoho transmet les sauts de ligne comme les deux caracteres \\ et n, pas
    comme de vrais retours. Et l'URL du Google Doc est du bruit pur pour un
    embedding : "https docs google document" n'a aucun rapport semantique avec
    le sujet de la reunion. On la retire du texte mais on la garde en metadonnee
    pour que l'utilisateur puisse ouvrir la source.
    """
    t = texte or ""
    t = t.replace("\\n", "\n").replace("\\t", "\t")

    liens = URL_RE.findall(t)
    lien_doc = next((l for l in liens if "docs.google.com" in l), liens[0] if liens else "")

    t = URL_RE.sub(" ", t)
    t = re.sub(r"-{2,}\s*SYNTH[EÈ]SE IA \(GEMINI\)\s*-{2,}", "", t, flags=re.I)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return {"texte": t.strip(), "lien_doc": lien_doc}


def decouper(texte: str, titre: str = "") -> List[str]:
    """Coupe en morceaux d'environ CHUNK_MAX caracteres, sur les frontieres
    naturelles (paragraphes, puces) plutot qu'au milieu d'une phrase.

    Chaque morceau est prefixe du titre de la reunion : isole, un morceau
    perdrait sinon toute indication de client et de date, et le LLM ne pourrait
    pas citer sa source.
    """
    blocs = [b.strip() for b in re.split(r"\n\s*\n|\n(?=[-•*]\s)", texte) if b.strip()]
    prefixe = f"{titre}\n" if titre else ""

    morceaux, courant = [], ""
    for bloc in blocs:
        # Un bloc plus long que la limite est coupe par phrases.
        if len(bloc) > CHUNK_MAX:
            if courant:
                morceaux.append(courant)
                courant = ""
            phrases = re.split(r"(?<=[.!?])\s+", bloc)
            tampon = ""
            for p in phrases:
                if len(tampon) + len(p) + 1 > CHUNK_MAX and tampon:
                    morceaux.append(tampon)
                    tampon = p
                else:
                    tampon = f"{tampon} {p}".strip()
            if tampon:
                morceaux.append(tampon)
            continue

        if len(courant) + len(bloc) + 2 > CHUNK_MAX and courant:
            morceaux.append(courant)
            courant = bloc
        else:
            courant = f"{courant}\n\n{bloc}".strip()

    if courant:
        morceaux.append(courant)

    return [f"{prefixe}{m}" for m in morceaux] or [f"{prefixe}{texte}".strip()]


# ---------------------------------------------------------------------------
# 4. HUGGING FACE
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    if not HF_TOKEN:
        raise HTTPException(503, "HF_TOKEN absent.")
    try:
        r = requests.post(
            HF_EMBED_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": text},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise HTTPException(502, f"Hugging Face injoignable : {e}")

    if r.status_code != 200:
        log.error("HF %s -> %s", r.status_code, r.text[:400])
        raise HTTPException(502, f"Hugging Face a renvoye {r.status_code}: {r.text[:200]}")

    vec = r.json()
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = vec[0]
    if not (isinstance(vec, list) and vec and isinstance(vec[0], (int, float))):
        raise HTTPException(502, f"Format d'embedding inattendu : {str(vec)[:200]}")
    return vec


def appeler_llm(system: str, user: str, max_tokens: int = 800) -> str:
    from huggingface_hub import InferenceClient

    hf = InferenceClient(api_key=HF_TOKEN)
    completion = hf.chat_completion(
        model=MODEL_ID,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. EXTRACTION STRUCTUREE
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "Tu extrais des donnees structurees de comptes rendus de reunion client B2B. "
    "Tu reponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni apres, "
    "sans balises de code. "
    "Tu n'inventes rien : si une information est absente du texte, tu laisses le "
    "champ vide ou la liste vide. "
    "Le champ 'categorie' de chaque probleme DOIT etre choisi exactement dans la "
    "liste imposee, sans reformulation."
)


def extraire_structure(texte: str, titre: str) -> Dict[str, Any]:
    """Demande au LLM un JSON de problemes et d'actions.

    Le prompt impose la liste fermee de categories. Toute valeur hors liste est
    ramenee a "Autre" cote Python : on ne fait pas confiance au LLM pour
    respecter une contrainte, on la verifie.
    """
    cats = "\n".join(f'- "{c}"' for c in CATEGORIES)
    user = f"""Titre de la reunion : {titre}

Compte rendu :
{texte[:6000]}

Categories autorisees (recopier a l'identique) :
{cats}

Reponds avec ce JSON exactement :
{{
  "client": "nom de l'entreprise cliente",
  "contact": "nom de la personne rencontree",
  "type_reunion": "RDV Support | Setup | Suivi | Formation | Autre",
  "problemes": [
    {{"categorie": "une des categories ci-dessus", "description": "une phrase"}}
  ],
  "actions": [
    {{"responsable": "nom", "tache": "une phrase"}}
  ]
}}"""

    brut = appeler_llm(EXTRACT_SYSTEM, user, max_tokens=1200)

    # Le LLM entoure souvent son JSON de ```json ... ``` malgre la consigne.
    nettoye = re.sub(r"^```(?:json)?|```$", "", brut.strip(), flags=re.M).strip()
    debut, fin = nettoye.find("{"), nettoye.rfind("}")
    if debut == -1 or fin == -1:
        raise HTTPException(502, f"Le LLM n'a pas renvoye de JSON : {brut[:200]}")

    try:
        data = json.loads(nettoye[debut:fin + 1])
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"JSON invalide du LLM : {e} | {nettoye[:200]}")

    # Garde-fou : on force les categories dans la liste autorisee.
    for p in data.get("problemes") or []:
        if p.get("categorie") not in CATEGORIES:
            log.info("Categorie hors liste ramenee a Autre : %r", p.get("categorie"))
            p["categorie"] = "Autre"

    return data


# ---------------------------------------------------------------------------
# 6. SCHEMAS
# ---------------------------------------------------------------------------

class NoteModel(BaseModel):
    note_id: Optional[str] = None
    client: Optional[str] = None      # si absent, le LLM le deduit du texte
    note: str
    date: Optional[str] = None
    titre: Optional[str] = None
    extraire: Optional[bool] = True   # False = indexation seule, sans appel LLM


class ChatModel(BaseModel):
    question: str
    n_results: Optional[int] = 5
    client: Optional[str] = None
    min_score: Optional[float] = None


# ---------------------------------------------------------------------------
# 7. DIAGNOSTIC
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/health")
def health():
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
        "min_score": MIN_SCORE_DEFAULT,
        "categories": CATEGORIES,
    }


@app.get("/debug/pinecone")
def debug_pinecone():
    try:
        return get_index().describe_index_stats()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Pinecone: {e}")


@app.get("/debug/embedding")
def debug_embedding(text: str = "test de vectorisation"):
    vec = get_embedding(text)
    return {"dimension": len(vec), "apercu": vec[:5]}


@app.get("/debug/search")
def debug_search(question: str, n_results: int = 8):
    """Scores bruts de Pinecone, sans filtrage ni LLM. Sert a calibrer
    MIN_SCORE : compare les scores des notes pertinentes a ceux du bruit."""
    res = get_index().query(
        vector=get_embedding(question), top_k=n_results, include_metadata=True
    )
    return {
        "min_score_actuel": MIN_SCORE_DEFAULT,
        "resultats": [
            {
                "score": _score(m),
                "retenu": (_score(m) or 0) >= MIN_SCORE_DEFAULT,
                "client": _meta(m).get("client"),
                "extrait": (_meta(m).get("texte") or "")[:150],
            }
            for m in _matches(res)
        ],
    }


@app.post("/debug/chunk")
def debug_chunk(data: NoteModel):
    """Montre le nettoyage et le decoupage SANS rien ecrire ni appeler le LLM.
    A utiliser avant toute ingestion en masse pour verifier le resultat."""
    c = nettoyer_note(data.note)
    morceaux = decouper(c["texte"], data.titre or "")
    return {
        "lien_doc": c["lien_doc"],
        "longueur_nettoyee": len(c["texte"]),
        "nb_morceaux": len(morceaux),
        "morceaux": [{"n": i + 1, "taille": len(m), "texte": m}
                     for i, m in enumerate(morceaux)],
    }


# ---------------------------------------------------------------------------
# 8. HELPERS PINECONE
# ---------------------------------------------------------------------------

def _matches(res):
    m = getattr(res, "matches", None)
    if m is None:
        m = res["matches"] if isinstance(res, dict) else []
    return m or []


def _meta(match):
    m = getattr(match, "metadata", None)
    if m is None and isinstance(match, dict):
        m = match.get("metadata")
    return m or {}


def _score(match):
    s = getattr(match, "score", None)
    if s is None and isinstance(match, dict):
        s = match.get("score")
    return s


# ---------------------------------------------------------------------------
# 9. INGESTION
# ---------------------------------------------------------------------------

@app.post("/zoho-webhook")
def recevoir_note_zoho(data: NoteModel):
    brut = (data.note or "").strip()
    if not brut:
        raise HTTPException(400, "Note vide")

    c = nettoyer_note(brut)
    texte, lien_doc = c["texte"], c["lien_doc"]
    titre = (data.titre or texte.split("\n", 1)[0])[:200]

    date_note = data.date or str(_date.today())
    # Postgres exige AAAA-MM-JJ sur une colonne date. Zoho envoie parfois un
    # horodatage ISO complet : on tronque plutot que d'echouer a l'insertion.
    if len(date_note) > 10:
        date_note = date_note[:10]

    note_id = data.note_id or f"zoho_{date_note}_{abs(hash(brut)) % 10**8}"

    # --- Extraction structuree (facultative) ---
    extrait: Dict[str, Any] = {}
    if data.extraire:
        try:
            extrait = extraire_structure(texte, titre)
        except HTTPException as e:
            # L'extraction est un plus : on n'abandonne pas l'indexation RAG
            # parce que le LLM a mal repondu.
            log.error("Extraction echouee: %s", e.detail)
            extrait = {"_erreur": str(e.detail)}

    client = data.client or extrait.get("client") or "Client Inconnu"

    # --- Pinecone : un vecteur par morceau ---
    morceaux = decouper(texte, titre)
    vecteurs = []
    for i, m in enumerate(morceaux):
        vecteurs.append({
            "id": f"{note_id}#{i}",
            "values": get_embedding(m),
            "metadata": {
                "note_id": str(note_id),
                "client": client,
                "date": date_note,
                "titre": titre,
                "lien_doc": lien_doc,
                "morceau": i,
                "texte": m,
            },
        })

    try:
        get_index().upsert(vectors=vecteurs)
    except Exception as e:
        log.error("Pinecone upsert: %s", e)
        raise HTTPException(502, f"Echec de l'indexation Pinecone : {e}")

    # --- Supabase : reunion + problemes + actions ---
    avertissements = []
    sb = None
    try:
        sb = get_supabase()
        sb.table("reunions").upsert({
            "id": str(note_id),
            "client": client,
            "contact": extrait.get("contact"),
            "date_reunion": date_note,
            "type_reunion": extrait.get("type_reunion"),
            "titre": titre,
            "lien_doc": lien_doc,
            "texte_complet": texte,
        }).execute()
    except Exception as e:
        log.error("Supabase reunions: %s", e)
        avertissements.append(f"reunions: {e}")

    nb_pb = nb_ac = 0
    if sb is not None and not extrait.get("_erreur"):
        # On purge avant d'inserer : sans ca, reingerer la meme note dupliquerait
        # tous ses problemes et fausserait durablement les comptages.
        try:
            sb.table("problemes").delete().eq("reunion_id", str(note_id)).execute()
            sb.table("actions").delete().eq("reunion_id", str(note_id)).execute()
        except Exception as e:
            log.warning("Purge prealable: %s", e)

        lignes_pb = [
            {"reunion_id": str(note_id), "client": client, "date_reunion": date_note,
             "categorie": p.get("categorie", "Autre"),
             "description": (p.get("description") or "")[:1000]}
            for p in (extrait.get("problemes") or []) if p.get("categorie")
        ]
        lignes_ac = [
            {"reunion_id": str(note_id), "client": client, "date_reunion": date_note,
             "responsable": a.get("responsable"),
             "tache": (a.get("tache") or "")[:1000]}
            for a in (extrait.get("actions") or []) if a.get("tache")
        ]

        for table, lignes in (("problemes", lignes_pb), ("actions", lignes_ac)):
            if not lignes:
                continue
            try:
                sb.table(table).insert(lignes).execute()
                if table == "problemes":
                    nb_pb = len(lignes)
                else:
                    nb_ac = len(lignes)
            except Exception as e:
                log.error("Supabase %s: %s", table, e)
                avertissements.append(f"{table}: {e}")

    return {
        "status": "partial" if avertissements else "success",
        "note_id": note_id,
        "client": client,
        "morceaux_indexes": len(vecteurs),
        "problemes_extraits": nb_pb,
        "actions_extraites": nb_ac,
        "extraction": extrait if not extrait.get("_erreur") else None,
        "avertissements": avertissements or None,
    }


# ---------------------------------------------------------------------------
# 10. RAG SEMANTIQUE
# ---------------------------------------------------------------------------

@app.post("/chat-rag")
def chat_rag(data: ChatModel):
    question = (data.question or "").strip()
    if not question:
        raise HTTPException(400, "Question vide")

    seuil = MIN_SCORE_DEFAULT if data.min_score is None else data.min_score

    kwargs = {
        "vector": get_embedding(question),
        "top_k": data.n_results or 5,
        "include_metadata": True,
    }
    if data.client:
        kwargs["filter"] = {"client": {"$eq": data.client}}

    try:
        res = get_index().query(**kwargs)
    except Exception as e:
        raise HTTPException(502, f"Recherche Pinecone impossible : {e}")

    retenus = [m for m in _matches(res)
               if _score(m) is None or _score(m) >= seuil]

    if not retenus:
        return {"status": "success", "question": question,
                "reponse": NO_INFO, "sources": []}

    contexte, sources = [], []
    for m in retenus:
        meta = _meta(m)
        contexte.append(
            f"[Client: {meta.get('client','Inconnu')} | Date: {meta.get('date','')}]\n"
            f"{meta.get('texte','')}"
        )
        sources.append({
            "doc": meta.get("texte", ""),
            "score": _score(m),
            "metadata": {
                "client": meta.get("client"),
                "date": meta.get("date"),
                "titre": meta.get("titre"),
                "lien_doc": meta.get("lien_doc"),
            },
        })

    system = (
        "Tu es un assistant CRM factuel pour le suivi client B2B. "
        "Tu reponds exclusivement a partir des notes fournies. "
        "Tu n'utilises aucune connaissance externe et tu n'extrapoles jamais. "
        f'Si les notes ne contiennent pas la reponse, tu ecris exactement : "{NO_INFO}" '
        "Tu cites le client et la date des notes utilisees. Tu reponds en francais."
    )
    user = "Notes CRM :\n\n" + "\n\n---\n\n".join(contexte) + f"\n\nQuestion : {question}"

    try:
        reponse = appeler_llm(system, user, max_tokens=500)
    except Exception as e:
        raise HTTPException(502, f"Generation impossible : {e}")

    return {"status": "success", "question": question,
            "reponse": reponse, "sources": sources}


# ---------------------------------------------------------------------------
# 11. ANALYTIQUE SQL
# ---------------------------------------------------------------------------
# Ces routes ne passent PAS par le LLM. Les chiffres viennent de Postgres, donc
# ils sont exacts et reproductibles. C'est ce qui alimentera les dashboards.

@app.get("/stats/problemes")
def stats_problemes(
    debut: Optional[str] = Query(None, description="AAAA-MM-JJ inclus"),
    fin: Optional[str] = Query(None, description="AAAA-MM-JJ inclus"),
    client: Optional[str] = None,
):
    """Comptage des problemes par categorie. Repond a "le probleme le plus
    remonte en aout" : /stats/problemes?debut=2026-08-01&fin=2026-08-31"""
    q = get_supabase().table("problemes").select("categorie,client,date_reunion")
    if debut:
        q = q.gte("date_reunion", debut)
    if fin:
        q = q.lte("date_reunion", fin)
    if client:
        q = q.eq("client", client)

    lignes = q.execute().data or []
    compte: Dict[str, int] = {}
    for l in lignes:
        cat = l.get("categorie") or "Autre"
        compte[cat] = compte.get(cat, 0) + 1

    classement = sorted(compte.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "periode": {"debut": debut, "fin": fin},
        "client": client,
        "total": len(lignes),
        "par_categorie": [{"categorie": c, "nb": n} for c, n in classement],
        "top": classement[0][0] if classement else None,
    }


@app.get("/stats/dernier-rdv")
def dernier_rdv(client: str):
    """Derniere reunion d'un client, avec ses problemes et actions.
    Un tri par date, pas une recherche semantique."""
    sb = get_supabase()
    reunions = (sb.table("reunions").select("*")
                .ilike("client", f"%{client}%")
                .order("date_reunion", desc=True).limit(1).execute().data)
    if not reunions:
        return {"trouve": False, "client_recherche": client}

    r = reunions[0]
    pbs = sb.table("problemes").select("categorie,description") \
        .eq("reunion_id", r["id"]).execute().data or []
    acts = sb.table("actions").select("responsable,tache,fait") \
        .eq("reunion_id", r["id"]).execute().data or []

    return {"trouve": True, "reunion": r, "problemes": pbs, "actions": acts}


@app.get("/stats/actions")
def stats_actions(responsable: Optional[str] = None, fait: Optional[bool] = None):
    """Suivi des actions a mener, filtrable par responsable et par statut."""
    q = get_supabase().table("actions").select("*")
    if responsable:
        q = q.ilike("responsable", f"%{responsable}%")
    if fait is not None:
        q = q.eq("fait", fait)
    lignes = q.order("date_reunion", desc=True).execute().data or []
    return {"total": len(lignes), "actions": lignes}


@app.get("/stats/clients")
def stats_clients():
    """Vue d'ensemble : nombre de reunions et de problemes par client."""
    sb = get_supabase()
    reunions = sb.table("reunions").select("client,date_reunion").execute().data or []
    problemes = sb.table("problemes").select("client").execute().data or []

    agg: Dict[str, Dict[str, Any]] = {}
    for r in reunions:
        cl = r.get("client") or "Inconnu"
        e = agg.setdefault(cl, {"client": cl, "nb_reunions": 0,
                                "nb_problemes": 0, "derniere_reunion": None})
        e["nb_reunions"] += 1
        d = r.get("date_reunion")
        if d and (e["derniere_reunion"] is None or d > e["derniere_reunion"]):
            e["derniere_reunion"] = d
    for p in problemes:
        cl = p.get("client") or "Inconnu"
        agg.setdefault(cl, {"client": cl, "nb_reunions": 0,
                            "nb_problemes": 0, "derniere_reunion": None})
        agg[cl]["nb_problemes"] += 1

    return {"clients": sorted(agg.values(),
                              key=lambda c: c["nb_problemes"], reverse=True)}
