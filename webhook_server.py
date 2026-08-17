"""
Zoho CRM -> RAG + Analytique (Pinecone + Supabase + Hugging Face)

Deux chemins de donnees, chacun pour ce qu'il sait faire :

  PINECONE (semantique) -> questions ouvertes, formulations libres.
  SUPABASE (structure)  -> comptages, filtres, tris, dashboards.
                           Postgres compte ; un LLM ne compte pas.

Trois mecanismes assurent le rappel (retrouver une info qui existe bien) :

  1. PREFIXE ENRICHI. Chaque morceau indexe porte "Client: X | Contact: Y".
     Un embedding ne peut pas deviner que Robin BASSIN travaille chez The
     Bouillon Of Paris : l'information doit etre DANS le texte vectorise.
  2. RESOLUTION D'ALIAS. Un nom de contact mentionne dans la question est
     resolu vers son client via Supabase, avant la recherche.
  3. RECHERCHE HYBRIDE. Vectoriel + mots-cles SQL, puis fusion. Les noms
     propres sont mal captes par les embeddings, mais parfaitement par un
     ILIKE. Les deux methodes sont complementaires, pas concurrentes.

Plus une REFORMULATION optionnelle : le LLM enrichit la question de synonymes
metier avant la recherche. Cout ~2-3 s, gain de rappel important sur les
formulations familieres. Desactivable via REFORMULER=false.

Les noms de clients sont normalises par clients.py (table canonique partagee
avec import_notes.py) : le webhook Zoho envoie le nom du CONTACT, la table le
rattache a sa societe.
"""

import os
import re
import json
import logging
from datetime import date as _date
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from clients import resoudre_client, CLIENT_INCONNU

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


def _env_float(nom: str, defaut: float) -> float:
    try:
        return float(os.environ.get(nom, str(defaut)))
    except ValueError:
        log.warning("%s illisible, repli sur %s", nom, defaut)
        return defaut


MIN_SCORE_DEFAULT = _env_float("MIN_SCORE", 0.15)
REFORMULER_DEFAUT = os.environ.get("REFORMULER", "true").lower() != "false"

# Categories avec une definition courte : sans elle, le LLM se rabat sur
# "Autre" des qu'un cas est limite, et "Autre" devient la categorie dominante.
CATEGORIES_DEF: Dict[str, str] = {
    "Facturation & import": "reception, import, formats de facture, plateformes "
                            "agreees, XML, scans, factures numeriques",
    "Integration caisse": "systeme de caisse, Popina, catalogue caisse, "
                          "interfacage, remontee automatique des ventes",
    "Produits & fournisseurs": "prix, hausses tarifaires, comparaison "
                               "fournisseurs, categories d'achats, mercuriales",
    "Recettes & marges": "fiches techniques, cout de revient, marge brute, "
                         "TVA, portions, rentabilite d'un plat",
    "Droits & acces": "comptes utilisateurs, permissions, restriction de "
                      "droits, identifiants de connexion",
    "Parametrage etablissement": "SIRET, coordonnees, informations legales, "
                                 "configuration initiale de l'etablissement",
    "Performance & donnees": "lenteurs, volume d'historique a charger, "
                             "qualite ou fiabilite des donnees",
    "Formation & prise en main": "accompagnement, navigation dans l'outil, "
                                 "autonomie de l'utilisateur, demonstration",
    "Demande d'evolution": "fonctionnalite absente ou souhaitee, amelioration "
                           "demandee, besoin non couvert",
    "Autre": "UNIQUEMENT si aucune categorie ci-dessus ne convient",
}

if os.environ.get("CATEGORIES"):
    CATEGORIES = [c.strip() for c in os.environ["CATEGORIES"].split(";") if c.strip()]
else:
    CATEGORIES = list(CATEGORIES_DEF.keys())

HF_EMBED_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"
    "/pipeline/feature-extraction"
)

REQUEST_TIMEOUT = 30
CHUNK_MAX = 900
NO_INFO = "Information non disponible dans la base de donnees CRM."

app = FastAPI(title="Zoho CRM to RAG + Analytique")


# ---------------------------------------------------------------------------
# 2. CLIENTS PARESSEUX
# ---------------------------------------------------------------------------

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
# 3. HUGGING FACE
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
# 4. NETTOYAGE ET DECOUPAGE
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")


def nettoyer_note(texte: str) -> Dict[str, str]:
    """Separe l'URL du contenu et normalise les retours a la ligne.

    Zoho transmet les sauts de ligne comme les deux caracteres \\ et n. Et
    l'URL du Google Doc est du bruit pur pour un embedding : on la retire du
    texte mais on la garde en metadonnee pour que la source reste ouvrable.
    """
    t = (texte or "").replace("\\n", "\n").replace("\\t", "\t")
    liens = URL_RE.findall(t)
    lien_doc = next((l for l in liens if "docs.google.com" in l),
                    liens[0] if liens else "")
    t = URL_RE.sub(" ", t)
    t = re.sub(r"-{2,}\s*SYNTH[EÈ]SE IA \(GEMINI\)\s*-{2,}", "", t, flags=re.I)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return {"texte": t.strip(), "lien_doc": lien_doc}


def decouper(texte: str, entete: str = "") -> List[str]:
    """Morceaux d'environ CHUNK_MAX caracteres, coupes sur les frontieres
    naturelles. L'entete est repetee sur chaque morceau : isole, un morceau
    perdrait sinon toute mention du client, du contact et de la date."""
    blocs = [b.strip() for b in re.split(r"\n\s*\n|\n(?=[-•*]\s)", texte) if b.strip()]
    prefixe = f"{entete}\n" if entete else ""

    morceaux, courant = [], ""
    for bloc in blocs:
        if len(bloc) > CHUNK_MAX:
            if courant:
                morceaux.append(courant)
                courant = ""
            tampon = ""
            for p in re.split(r"(?<=[.!?])\s+", bloc):
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
# 5. EXTRACTION STRUCTUREE
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "Tu extrais des donnees structurees de comptes rendus de reunion client B2B. "
    "Tu reponds UNIQUEMENT avec un objet JSON valide : pas de texte avant ou "
    "apres, pas de balises de code. "
    "Tu n'inventes rien : une information absente du texte donne un champ vide "
    "ou une liste vide."
)


def extraire_structure(texte: str, titre: str) -> Dict[str, Any]:
    """Demande au LLM un JSON de problemes et d'actions.

    Le prompt impose la liste fermee de categories AVEC leur definition, et
    exige une action par tache. Toute categorie hors liste est ramenee a
    "Autre" cote Python : on ne fait pas confiance au LLM pour respecter une
    contrainte, on la verifie.
    """
    cats = "\n".join(f'- "{c}" : {d}' for c, d in CATEGORIES_DEF.items()
                     if c in CATEGORIES)

    user = f"""Titre de la reunion : {titre}

Compte rendu :
{texte[:6000]}

CATEGORIES AUTORISEES (recopier le libelle exact, sans reformuler) :
{cats}

REGLES IMPERATIVES :
1. "type_reunion" : UN SEUL mot parmi RDV Support, Setup, Suivi, Formation,
   Autre. Ne recopie jamais la liste entiere.
2. "client" : le nom de l'ENTREPRISE (souvent dans le titre). "contact" : le
   nom de la PERSONNE rencontree. Ce sont deux choses differentes.
3. Une entree d'action = UNE SEULE tache avec UN SEUL responsable. Si le texte
   mentionne quatre taches, produis quatre entrees. N'agrege jamais plusieurs
   taches dans une meme chaine.
4. N'utilise "Autre" que si aucune autre categorie ne convient. Cherche
   d'abord la categorie la plus proche dans la liste.

Reponds avec ce JSON exactement :
{{
  "client": "",
  "contact": "",
  "type_reunion": "",
  "problemes": [{{"categorie": "", "description": "une phrase"}}],
  "actions": [{{"responsable": "", "tache": "une seule tache"}}]
}}"""

    brut = appeler_llm(EXTRACT_SYSTEM, user, max_tokens=1500)

    nettoye = re.sub(r"^```(?:json)?|```$", "", brut.strip(), flags=re.M).strip()
    debut, fin = nettoye.find("{"), nettoye.rfind("}")
    if debut == -1 or fin == -1:
        raise HTTPException(502, f"Le LLM n'a pas renvoye de JSON : {brut[:200]}")
    try:
        data = json.loads(nettoye[debut:fin + 1])
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"JSON invalide du LLM : {e} | {nettoye[:200]}")

    for p in data.get("problemes") or []:
        if p.get("categorie") not in CATEGORIES:
            log.info("Categorie hors liste -> Autre : %r", p.get("categorie"))
            p["categorie"] = "Autre"

    # Garde-fou sur type_reunion : le LLM recopie parfois le menu.
    tr = (data.get("type_reunion") or "").strip()
    if "|" in tr or len(tr) > 30:
        data["type_reunion"] = tr.split("|")[0].strip()[:30] or None

    return data


# ---------------------------------------------------------------------------
# 6. RESOLUTION D'ALIAS  (contact -> client)
# ---------------------------------------------------------------------------
# "Quel est le probleme de Robin BASSIN ?" et "... de The Bouillon Of Paris ?"
# doivent mener au meme endroit. Un embedding ne peut pas le deviner : le lien
# n'existe que dans la table reunions. On l'exploite explicitement.

def annuaire() -> List[Dict[str, str]]:
    try:
        lignes = get_supabase().table("reunions").select("client,contact").execute().data
    except Exception as e:
        log.warning("Annuaire indisponible: %s", e)
        return []
    vus, out = set(), []
    for l in lignes or []:
        cle = (l.get("client"), l.get("contact"))
        if cle not in vus:
            vus.add(cle)
            out.append({"client": l.get("client") or "", "contact": l.get("contact") or ""})
    return out


def resoudre_entites(question: str) -> Dict[str, Any]:
    """Repere dans la question un client ou un contact connu.

    On compare sur des mots significatifs (>3 lettres) plutot que sur la chaine
    entiere : "bouillon" doit matcher "The Bouillon Of Paris", et "robin" doit
    matcher "Robin BASSIN".
    """
    q = question.lower()
    clients_trouves, contacts_trouves = set(), set()

    for e in annuaire():
        for champ, cible in (("client", clients_trouves), ("contact", contacts_trouves)):
            val = (e.get(champ) or "").strip()
            if not val:
                continue
            mots = [m for m in re.split(r"\W+", val.lower()) if len(m) > 3]
            if val.lower() in q or (mots and any(m in q for m in mots)):
                cible.add(val)
                if champ == "contact" and e.get("client"):
                    clients_trouves.add(e["client"])

    return {"clients": sorted(clients_trouves), "contacts": sorted(contacts_trouves)}


# ---------------------------------------------------------------------------
# 7. REFORMULATION
# ---------------------------------------------------------------------------

def reformuler(question: str, entites: Dict[str, Any]) -> str:
    """Enrichit la question de synonymes metier et des entites resolues.

    Une question familiere ("on a quoi comme souci chez robin") produit un
    vecteur eloigne du texte des notes. En l'enrichissant, on remonte le
    rappel sans toucher a l'index.
    """
    contexte = ""
    if entites["clients"]:
        contexte += f"\nClients concernes : {', '.join(entites['clients'])}"
    if entites["contacts"]:
        contexte += f"\nContacts concernes : {', '.join(entites['contacts'])}"

    system = (
        "Tu reformules une question pour une recherche documentaire dans un CRM "
        "de logiciel de gestion pour la restauration. "
        "Tu produis UNE seule ligne de mots-cles, EXCLUSIVEMENT EN FRANCAIS, "
        "sans ponctuation superflue, sans introduction, sans explication. "
        "Tu conserves tous les noms propres et ajoutes ceux du contexte fourni. "
        "Tu n'ajoutes QUE des synonymes des mots deja presents dans la question. "
        "Tu n'INVENTES JAMAIS de theme absent de la question : pas d'hygiene, "
        "pas de securite alimentaire, pas de ressources humaines, pas de "
        "reclamation client, si la question n'en parle pas. "
        "Domaine reel des notes : facturation, import de donnees, caisse, "
        "produits, fournisseurs, recettes, marges, droits d'acces, parametrage, "
        "formation."
    )
    user = f"Question : {question}{contexte}\n\nLigne de mots-cles enrichie :"

    try:
        sortie = appeler_llm(system, user, max_tokens=120).strip()
        sortie = sortie.split("\n")[0].strip(' "')
        # Le LLM part parfois en chinois ou en anglais malgre la consigne. Des
        # caracteres non latins polluent l'embedding : on coupe la sortie a leur
        # premiere apparition plutot que de tout jeter.
        coupe = re.search(r"[^\x00-\x7F\u00C0-\u017F\s]", sortie)
        if coupe:
            sortie = sortie[:coupe.start()].strip()
        if 3 < len(sortie) < 400:
            return f"{question} {sortie}"
    except Exception as e:
        log.warning("Reformulation echouee, question brute conservee: %s", e)
    return question


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


def _id(match):
    i = getattr(match, "id", None)
    if i is None and isinstance(match, dict):
        i = match.get("id")
    return i


# ---------------------------------------------------------------------------
# 9. RECHERCHE HYBRIDE
# ---------------------------------------------------------------------------

def recherche_hybride(
    requete: str, entites: Dict[str, Any], top_k: int, seuil: float,
    client_filtre: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Vectoriel + mots-cles SQL, fusionnes.

    Le vectoriel capte le sens mais rate les noms propres. Le SQL capte les
    noms propres mais pas les paraphrases. Ensemble, ils couvrent les deux.
    Les resultats SQL entrent avec origine="sql" et echappent au seuil, qui ne
    s'applique qu'aux scores de similarite.
    """
    resultats: Dict[str, Dict[str, Any]] = {}

    # --- Voie vectorielle ---
    kwargs = {"vector": get_embedding(requete), "top_k": top_k, "include_metadata": True}
    cible = client_filtre or (entites["clients"][0] if len(entites["clients"]) == 1 else None)
    if cible:
        kwargs["filter"] = {"client": {"$eq": cible}}

    try:
        for m in _matches(get_index().query(**kwargs)):
            sc = _score(m)
            if sc is not None and sc < seuil:
                continue
            meta = _meta(m)
            resultats[str(_id(m))] = {
                "texte": meta.get("texte", ""), "score": sc, "origine": "vectoriel",
                "client": meta.get("client"), "date": meta.get("date"),
                "titre": meta.get("titre"), "lien_doc": meta.get("lien_doc"),
            }
    except Exception as e:
        log.error("Pinecone query: %s", e)
        raise HTTPException(502, f"Recherche Pinecone impossible : {e}")

    # --- Voie mots-cles ---
    # Utile surtout quand la question nomme une entite : on veut alors etre sur
    # de remonter ses reunions, meme si aucun vecteur ne passe le seuil.
    noms = entites["clients"] + entites["contacts"]
    if noms:
        try:
            sb = get_supabase()
            for nom in noms[:3]:
                lignes = (sb.table("reunions")
                          .select("id,client,contact,date_reunion,titre,lien_doc,texte_complet")
                          .or_(f"client.ilike.%{nom}%,contact.ilike.%{nom}%")
                          .order("date_reunion", desc=True).limit(3).execute().data) or []
                for r in lignes:
                    cle = f"sql:{r['id']}"
                    if cle in resultats:
                        continue
                    resultats[cle] = {
                        "texte": (r.get("texte_complet") or "")[:1500],
                        "score": None, "origine": "sql",
                        "client": r.get("client"), "date": str(r.get("date_reunion") or ""),
                        "titre": r.get("titre"), "lien_doc": r.get("lien_doc"),
                    }
        except Exception as e:
            log.warning("Recherche SQL: %s", e)

    # Les resultats scores d'abord, puis les correspondances SQL.
    return sorted(resultats.values(),
                  key=lambda r: (r["score"] is None, -(r["score"] or 0)))


# ---------------------------------------------------------------------------
# 10. SCHEMAS
# ---------------------------------------------------------------------------

class NoteModel(BaseModel):
    note_id: Optional[str] = None
    client: Optional[str] = None
    contact: Optional[str] = None       # nom de la personne, envoye par Zoho
    note: str
    date: Optional[str] = None
    titre: Optional[str] = None
    extraire: Optional[bool] = True


class ChatModel(BaseModel):
    question: str
    n_results: Optional[int] = 6
    client: Optional[str] = None
    min_score: Optional[float] = None
    reformuler: Optional[bool] = None   # None = valeur de REFORMULER


# ---------------------------------------------------------------------------
# 11. DIAGNOSTIC
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
        "reformulation": REFORMULER_DEFAUT,
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


@app.get("/debug/client")
def debug_client(nom: str):
    """Verifie ce que la table canonique renvoie pour un nom donne.

    Sert a tester un alias de contact SANS creer de note : si le resultat est
    identique au nom envoye, l'entree manque dans clients.py."""
    resolu = resoudre_client(nom)
    return {
        "envoye": nom,
        "resolu": resolu,
        "dans_la_table": bool(resolu) and resolu != nom.strip(),
    }


@app.get("/debug/entites")
def debug_entites(question: str):
    """Montre quels clients et contacts sont reconnus dans une question, et la
    reformulation produite. C'est l'outil pour comprendre pourquoi une question
    trouve ou ne trouve pas."""
    ent = resoudre_entites(question)
    return {
        "question": question,
        "entites": ent,
        "annuaire_taille": len(annuaire()),
        "reformulation": reformuler(question, ent) if REFORMULER_DEFAUT else None,
    }


@app.get("/debug/search")
def debug_search(question: str, n_results: int = 8, reformulation: bool = True):
    """Scores bruts, sans filtrage ni redaction. Sert a calibrer MIN_SCORE :
    compare les scores des morceaux pertinents a ceux du bruit."""
    ent = resoudre_entites(question)
    requete = reformuler(question, ent) if reformulation else question
    res = get_index().query(vector=get_embedding(requete),
                            top_k=n_results, include_metadata=True)
    return {
        "min_score_actuel": MIN_SCORE_DEFAULT,
        "entites": ent,
        "requete_utilisee": requete,
        "resultats": [
            {"score": _score(m), "retenu": (_score(m) or 0) >= MIN_SCORE_DEFAULT,
             "client": _meta(m).get("client"),
             "extrait": (_meta(m).get("texte") or "")[:150]}
            for m in _matches(res)
        ],
    }


@app.post("/debug/chunk")
def debug_chunk(data: NoteModel):
    """Nettoyage et decoupage, SANS ecriture ni appel LLM. A lancer avant toute
    ingestion en masse pour verifier le resultat."""
    c = nettoyer_note(data.note)
    entete = construire_entete(data.client or "", "", data.titre or "", "")
    morceaux = decouper(c["texte"], entete)
    return {
        "lien_doc": c["lien_doc"],
        "entete": entete,
        "longueur_nettoyee": len(c["texte"]),
        "nb_morceaux": len(morceaux),
        "morceaux": [{"n": i + 1, "taille": len(m), "texte": m}
                     for i, m in enumerate(morceaux)],
    }


# ---------------------------------------------------------------------------
# 12. INGESTION
# ---------------------------------------------------------------------------

def construire_entete(client: str, contact: str, titre: str, date: str) -> str:
    """Entete repetee sur chaque morceau indexe.

    C'est le mecanisme cle du rappel par nom : sans "Contact: Robin BASSIN"
    dans le texte vectorise, aucune question mentionnant Robin BASSIN ne peut
    matcher ce morceau."""
    bouts = []
    if client:
        bouts.append(f"Client: {client}")
    if contact:
        bouts.append(f"Contact: {contact}")
    if date:
        bouts.append(f"Date: {date}")
    ligne = " | ".join(bouts)
    return f"{ligne}\n{titre}".strip() if titre else ligne


@app.post("/zoho-webhook")
def recevoir_note_zoho(data: NoteModel):
    brut = (data.note or "").strip()
    if not brut:
        raise HTTPException(400, "Note vide")

    c = nettoyer_note(brut)
    texte, lien_doc = c["texte"], c["lien_doc"]
    titre = (data.titre or texte.split("\n", 1)[0])[:200]

    date_note = data.date or str(_date.today())
    if len(date_note) > 10:      # Zoho envoie parfois un horodatage ISO complet
        date_note = date_note[:10]

    note_id = data.note_id or f"zoho_{date_note}_{abs(hash(brut)) % 10**8}"

    extrait: Dict[str, Any] = {}
    if data.extraire:
        try:
            extrait = extraire_structure(texte, titre)
        except HTTPException as e:
            # L'extraction est un plus : on n'abandonne pas l'indexation RAG
            # parce que le LLM a mal repondu.
            log.error("Extraction echouee: %s", e.detail)
            extrait = {"_erreur": str(e.detail)}

    # Zoho envoie le nom du CONTACT dans "client" (le champ societe n'est pas
    # disponible sur un declencheur Remarques). clients.py rattache ce nom a sa
    # societe ; un nom hors table est conserve tel quel avec un warning.
    client = (resoudre_client(data.client)
              or resoudre_client(extrait.get("client"))
              or CLIENT_INCONNU)
    # Le champ Zoho passe devant l'extraction : donnee fiable contre donnee
    # deduite par le LLM.
    contact = data.contact or extrait.get("contact") or ""

    # --- Pinecone : un vecteur par morceau, entete enrichie ---
    entete = construire_entete(client, contact, titre, date_note)
    morceaux = decouper(texte, entete)
    vecteurs = [{
        "id": f"{note_id}#{i}",
        "values": get_embedding(m),
        "metadata": {"note_id": str(note_id), "client": client, "contact": contact,
                     "date": date_note, "titre": titre, "lien_doc": lien_doc,
                     "morceau": i, "texte": m},
    } for i, m in enumerate(morceaux)]

    try:
        get_index().upsert(vectors=vecteurs)
    except Exception as e:
        log.error("Pinecone upsert: %s", e)
        raise HTTPException(502, f"Echec de l'indexation Pinecone : {e}")

    # --- Supabase ---
    avertissements, sb = [], None
    try:
        sb = get_supabase()
        sb.table("reunions").upsert({
            "id": str(note_id), "client": client, "contact": contact or None,
            "date_reunion": date_note, "type_reunion": extrait.get("type_reunion"),
            "titre": titre, "lien_doc": lien_doc, "texte_complet": texte,
        }).execute()
    except Exception as e:
        log.error("Supabase reunions: %s", e)
        avertissements.append(f"reunions: {e}")

    nb_pb = nb_ac = 0
    if sb is not None and not extrait.get("_erreur"):
        # Purge avant insertion : sans ca, reingerer la meme note dupliquerait
        # ses problemes et fausserait durablement les comptages.
        for t in ("problemes", "actions"):
            try:
                sb.table(t).delete().eq("reunion_id", str(note_id)).execute()
            except Exception as e:
                log.warning("Purge %s: %s", t, e)

        base = {"reunion_id": str(note_id), "client": client, "date_reunion": date_note}
        lignes_pb = [{**base, "categorie": p.get("categorie", "Autre"),
                      "description": (p.get("description") or "")[:1000]}
                     for p in (extrait.get("problemes") or []) if p.get("categorie")]
        lignes_ac = [{**base, "responsable": a.get("responsable"),
                      "tache": (a.get("tache") or "")[:1000]}
                     for a in (extrait.get("actions") or []) if a.get("tache")]

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
        "note_id": note_id, "client": client, "contact": contact,
        "morceaux_indexes": len(vecteurs),
        "problemes_extraits": nb_pb, "actions_extraites": nb_ac,
        "extraction": extrait if not extrait.get("_erreur") else None,
        "avertissements": avertissements or None,
    }


# ---------------------------------------------------------------------------
# 13. RAG
# ---------------------------------------------------------------------------

@app.post("/chat-rag")
def chat_rag(data: ChatModel):
    question = (data.question or "").strip()
    if not question:
        raise HTTPException(400, "Question vide")

    seuil = MIN_SCORE_DEFAULT if data.min_score is None else data.min_score
    faire_reformulation = REFORMULER_DEFAUT if data.reformuler is None else data.reformuler

    entites = resoudre_entites(question)
    requete = reformuler(question, entites) if faire_reformulation else question

    trouves = recherche_hybride(requete, entites, data.n_results or 6,
                                seuil, data.client)

    if not trouves:
        return {"status": "success", "question": question, "reponse": NO_INFO,
                "sources": [], "entites": entites}

    contexte = "\n\n---\n\n".join(
        f"[Client: {r.get('client') or 'Inconnu'} | Date: {r.get('date') or ''}]\n"
        f"{r.get('texte','')}" for r in trouves
    )

    # Une consigne de refus trop stricte bloque des reponses legitimes : aucune
    # phrase d'une note ne dit litteralement "le probleme de X est...", donc le
    # LLM concluait qu'il ne pouvait pas repondre. On lui dit explicitement que
    # SYNTHETISER est attendu, et on reserve NO_INFO au vrai hors-sujet.
    system = (
        "Tu es un assistant CRM factuel pour le suivi client B2B. "
        "Tu reponds a partir des notes fournies, sans connaissance externe. "
        "SYNTHETISER plusieurs elements d'une note est une reponse valide et "
        "attendue : si la question porte sur les problemes d'un client, tu "
        "listes tous les problemes, difficultes, points a traiter et actions "
        "en attente que tu trouves dans les notes. "
        "Un contact et son entreprise designent la meme situation : une "
        "question sur une personne concerne les notes de son entreprise. "
        f'Tu n\'ecris "{NO_INFO}" QUE si les notes ne parlent pas du tout du '
        "sujet demande. Ne l'ecris jamais quand les notes contiennent des "
        "elements de reponse, meme partiels. "
        "Tu cites le client et la date des notes utilisees. Tu reponds en francais."
    )
    user = f"Notes CRM :\n\n{contexte}\n\nQuestion : {question}"

    try:
        reponse = appeler_llm(system, user, max_tokens=600)
    except Exception as e:
        raise HTTPException(502, f"Generation impossible : {e}")

    return {
        "status": "success", "question": question,
        "requete_recherche": requete if faire_reformulation else None,
        "entites": entites, "reponse": reponse,
        "sources": [{"doc": r["texte"], "score": r["score"], "origine": r["origine"],
                     "metadata": {"client": r.get("client"), "date": r.get("date"),
                                  "titre": r.get("titre"), "lien_doc": r.get("lien_doc")}}
                    for r in trouves],
    }


# ---------------------------------------------------------------------------
# 14. ANALYTIQUE SQL
# ---------------------------------------------------------------------------
# Ces routes ne passent PAS par le LLM : les chiffres viennent de Postgres,
# donc ils sont exacts et reproductibles.

@app.get("/stats/problemes")
def stats_problemes(
    debut: Optional[str] = Query(None, description="AAAA-MM-JJ inclus"),
    fin: Optional[str] = Query(None, description="AAAA-MM-JJ inclus"),
    client: Optional[str] = None,
):
    """Comptage par categorie. "Le probleme le plus remonte en aout" :
    /stats/problemes?debut=2026-08-01&fin=2026-08-31"""
    q = get_supabase().table("problemes").select("categorie,client,date_reunion")
    if debut:
        q = q.gte("date_reunion", debut)
    if fin:
        q = q.lte("date_reunion", fin)
    if client:
        q = q.ilike("client", f"%{client}%")

    lignes = q.execute().data or []
    compte: Dict[str, int] = {}
    for l in lignes:
        cat = l.get("categorie") or "Autre"
        compte[cat] = compte.get(cat, 0) + 1
    classement = sorted(compte.items(), key=lambda kv: kv[1], reverse=True)

    return {"periode": {"debut": debut, "fin": fin}, "client": client,
            "total": len(lignes),
            "par_categorie": [{"categorie": c, "nb": n} for c, n in classement],
            "top": classement[0][0] if classement else None}


@app.get("/stats/dernier-rdv")
def dernier_rdv(client: str):
    """Derniere reunion d'un client OU d'un contact, avec problemes et actions.
    Un tri par date, pas une recherche semantique."""
    sb = get_supabase()
    reunions = (sb.table("reunions").select("*")
                .or_(f"client.ilike.%{client}%,contact.ilike.%{client}%")
                .order("date_reunion", desc=True).limit(1).execute().data)
    if not reunions:
        return {"trouve": False, "recherche": client}

    r = reunions[0]
    pbs = sb.table("problemes").select("categorie,description") \
        .eq("reunion_id", r["id"]).execute().data or []
    acts = sb.table("actions").select("responsable,tache,fait") \
        .eq("reunion_id", r["id"]).execute().data or []
    return {"trouve": True, "reunion": r, "problemes": pbs, "actions": acts}


@app.get("/stats/actions")
def stats_actions(responsable: Optional[str] = None, fait: Optional[bool] = None,
                  client: Optional[str] = None):
    q = get_supabase().table("actions").select("*")
    if responsable:
        q = q.ilike("responsable", f"%{responsable}%")
    if client:
        q = q.ilike("client", f"%{client}%")
    if fait is not None:
        q = q.eq("fait", fait)
    lignes = q.order("date_reunion", desc=True).execute().data or []
    return {"total": len(lignes), "actions": lignes}


@app.get("/stats/clients")
def stats_clients():
    sb = get_supabase()
    reunions = sb.table("reunions").select("client,contact,date_reunion").execute().data or []
    problemes = sb.table("problemes").select("client").execute().data or []

    agg: Dict[str, Dict[str, Any]] = {}
    for r in reunions:
        cl = r.get("client") or "Inconnu"
        e = agg.setdefault(cl, {"client": cl, "contacts": [], "nb_reunions": 0,
                                "nb_problemes": 0, "derniere_reunion": None})
        e["nb_reunions"] += 1
        ct = r.get("contact")
        if ct and ct not in e["contacts"]:
            e["contacts"].append(ct)
        d = r.get("date_reunion")
        if d and (e["derniere_reunion"] is None or d > e["derniere_reunion"]):
            e["derniere_reunion"] = d
    for p in problemes:
        cl = p.get("client") or "Inconnu"
        agg.setdefault(cl, {"client": cl, "contacts": [], "nb_reunions": 0,
                            "nb_problemes": 0, "derniere_reunion": None})
        agg[cl]["nb_problemes"] += 1

    return {"clients": sorted(agg.values(),
                              key=lambda c: c["nb_problemes"], reverse=True)}
