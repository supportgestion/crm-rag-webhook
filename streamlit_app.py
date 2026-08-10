import json
import sqlite3
import pandas as pd
import streamlit as st
import ollama
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="CRM IA Assistant", page_icon="🤖", layout="wide")

# 1. Modèle d'embeddings & ChromaDB
@st.cache_resource
def load_rag_engine():
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
    return collection

collection = load_rag_engine()

# 2. Connexion SQL
def get_sql_data():
    conn = sqlite3.connect('analytics_crm.db')
    df = pd.read_sql_query("SELECT * FROM incidents", conn)
    conn.close()
    return df

# --- INTERFACE STREAMLIT ---
st.title("🚀 CRM Intelligence Assistant")

tab1, tab2, tab3 = st.tabs(["💬 Chatbot RAG", "➕ Ajouter une note", "📊 Dashboard Incidents"])

# TAB 1 : CHATBOT
with tab1:
    st.subheader("Posez une question sur l'historique client")
    question = st.text_input("Exemple : Quel est le problème de Trinco avec Pennylane ?")
    
    if st.button("Rechercher") and question:
        with st.spinner("Analyse des notes CRM en cours..."):
            results = collection.query(query_texts=[question], n_results=2)
            docs = "\n---\n".join(results['documents'][0])
            
            prompt_rag = f"""
            Tu es un assistant CRM. Réponds en français à la question selon ces extraits :
            {docs}
            
            Question : {question}
            """
            reponse = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_rag}])
            
            st.success("Réponse de l'assistant :")
            st.write(reponse['message']['content'])
            
            with st.expander("🔍 Sources consultées dans ChromaDB"):
                st.write(docs)

# TAB 2 : INGESTION
with tab2:
    st.subheader("Ingérer une nouvelle note CRM")
    client = st.text_input("Nom du client")
    date_note = st.date_input("Date de la note")
    texte_note = st.text_area("Contenu de la note de réunion")
    
    if st.button("Enregistrer la note"):
        if client and texte_note:
            with st.spinner("Analyse et indexation..."):
                prompt_json = f"""
                Analyse cette note et extrait le problème au format JSON STRICT.
                {{"categorie": "nom court", "resume_probleme": "une phrase"}}
                Note : {texte_note}
                """
                response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_json}])
                
                try:
                    data = json.loads(response['message']['content'])
                except Exception:
                    data = {"categorie": "Général", "resume_probleme": "Analyse manuelle requise"}
                
                # Sauvegarde SQL
                conn = sqlite3.connect('analytics_crm.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO incidents (client, date_note, categorie, resume_probleme)
                    VALUES (?, ?, ?, ?)
                ''', (client, str(date_note), data['categorie'], data['resume_probleme']))
                conn.commit()
                conn.close()
                
                # Sauvegarde ChromaDB
                note_id = f"note_{pd.Timestamp.now().timestamp()}"
                collection.add(
                    documents=[texte_note],
                    metadatas=[{"client": client, "date": str(date_note)}],
                    ids=[note_id]
                )
                st.success("✅ Note enregistrée en SQL et vectorisée dans ChromaDB !")

# TAB 3 : DASHBOARD
with tab3:
    st.subheader("Analyse structurée des incidents (SQL)")
    df = get_sql_data()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Aucun incident enregistré.")