import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Assistant CRM", page_icon="🤖", layout="centered")

st.title("🤖 Chatbot CRM - Base de connaissances")
st.markdown("Posez vos questions sur l'historique des clients ou les incidents récents.")

# L'URL de ton API Railway (le nouveau serveur opérationnel)
API_URL = "https://crm-rag-webhook-production.up.railway.app/chat-rag"

# Initialisation de l'historique de chat façon Gemini
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Barre de saisie (le champ de texte en bas de l'écran)
if prompt := st.chat_input("Ex: Quel est le problème de Jihad ZAKHOUR ?"):
    
    # 1. Afficher la question de l'utilisateur dans le chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Interroger ton serveur Railway
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les notes du CRM..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"question": prompt, "n_results": 3}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    reponse_ia = data.get("reponse", "Désolé, je n'ai pas pu formuler de réponse.")
                    
                    # Afficher la réponse
                    st.markdown(reponse_ia)
                    
                    # Afficher les sources cachées dans un menu déroulant
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 Voir les notes CRM utilisées"):
                            for source in sources:
                                client = source.get("metadata", {}).get("client", "Inconnu")
                                date = source.get("metadata", {}).get("date", "Inconnue")
                                st.caption(f"**Client : {client}** (le {date})")
                                st.info(source.get("doc", ""))

                    # Sauvegarder la réponse dans l'historique
                    st.session_state.messages.append({"role": "assistant", "content": reponse_ia})
                else:
                    st.error(f"Le serveur CRM est indisponible (Code {response.status_code}). Attendez quelques secondes qu'il se réveille.")
                    
            except Exception as e:
                st.error("Impossible de se connecter au serveur Railway. Vérifiez que l'API est en ligne.")
