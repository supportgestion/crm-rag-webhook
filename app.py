import json
import sqlite3
import ollama
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

# 1. Modèle d'embeddings Hugging Face
print("🔄 Chargement du modèle d'embeddings Hugging Face...")
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class HFEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embedder.encode(input).tolist()

    def name(self) -> str:
        return "hf_multilingual_minilm"

# 2. Base vectorielle locale (ChromaDB)
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(
    name="crm_notes",
    embedding_function=HFEmbeddingFunction()
)

# 3. Base SQL locale pour le Dashboard
conn = sqlite3.connect('analytics_crm.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client TEXT,
        date_note TEXT,
        categorie TEXT,
        resume_probleme TEXT
    )
''')
conn.commit()

# 4. Ingestion d'une note CRM
def ingerer_note(note_id, client, date_note, texte_note):
    print(f"\n⚙️ Traitement de la note : {client}...")
    
    prompt_json = f"""
    Analyse cette note de réunion et extrait le problème principal au format JSON STRICT.
    Ne rajoute AUCUN texte supplémentaire.

    Format attendu :
    {{
        "categorie": "nom court de la catégorie du problème",
        "resume_probleme": "description en une phrase du problème"
    }}

    Note :
    {texte_note}
    """
    
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_json}])
    
    try:
        data = json.loads(response['message']['content'])
    except Exception:
        data = {"categorie": "Général", "resume_probleme": "Analyse requise"}

    # Stockage SQL
    cursor.execute('''
        INSERT INTO incidents (client, date_note, categorie, resume_probleme)
        VALUES (?, ?, ?, ?)
    ''', (client, date_note, data['categorie'], data['resume_probleme']))
    conn.commit()

    # Stockage Vectoriel RAG
    collection.add(
        documents=[texte_note],
        metadatas=[{"client": client, "date": date_note, "categorie": data['categorie']}],
        ids=[str(note_id)]
    )
    print("✅ Note indexée dans ChromaDB et enregistrée en SQL !")

# 5. Question au Chatbot RAG
def poser_question_chatbot(question):
    print(f"\n💬 Question : '{question}'")
    
    results = collection.query(query_texts=[question], n_results=2)
    docs_pertinents = "\n---\n".join(results['documents'][0])

    prompt_rag = f"""
    Tu es l'assistant CRM. Réponds à la question en te basant UNIQUEMENT sur les extraits de notes suivants.
    
    Notes :
    {docs_pertinents}

    Question : {question}
    """
    
    reponse = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_rag}])
    print("\n🤖 Réponse du Chatbot :")
    print(reponse['message']['content'])

# --- TEST ---
if __name__ == "__main__":
    note_trinco = """
    Août 5, 2026 - Suivi / Trinco / Parametrage comptable
    Problème d'importation vers Penny Lane en raison des numéros de pièces non reconnus.
    Contacter Popina pour les exports des ventes journalières.
    """
    
    ingerer_note("note_001", "Trinco", "2026-08-05", note_trinco)
    poser_question_chatbot("Quel problème rencontre le client Trinco pour la comptabilité ?")