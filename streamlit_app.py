import os

import requests
import streamlit as st

st.set_page_config(page_title="Assistant CRM", page_icon="🤖", layout="centered")

# URL de l'API, surchargeable sans toucher au code (secret Streamlit ou variable
# d'environnement). Evite de reediter le fichier a chaque changement d'hebergeur.
DEFAULT_API = "https://crm-rag-webhook-production.up.railway.app"
API_BASE = os.environ.get("API_BASE", st.secrets.get("API_BASE", DEFAULT_API)
                          if hasattr(st, "secrets") else DEFAULT_API).rstrip("/")

st.title("🤖 Chatbot CRM - Base de connaissances")
st.caption("Posez vos questions sur l'historique des clients ou les incidents récents.")

# --- Barre laterale de diagnostic ------------------------------------------
with st.sidebar:
    st.subheader("Diagnostic")
    st.code(API_BASE, language=None)
    if st.button("Tester l'API"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=60)
            st.success(f"HTTP {r.status_code}")
            st.json(r.json())
        except Exception as e:
            st.error(f"Injoignable : {e}")
    if st.button("Compter les vecteurs Pinecone"):
        try:
            r = requests.get(f"{API_BASE}/debug/pinecone", timeout=60)
            st.json(r.json())
        except Exception as e:
            st.error(f"Injoignable : {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Quel est le problème de Jihad ZAKHOUR ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les notes du CRM..."):
            try:
                # timeout indispensable : sans lui, un backend endormi fait
                # pendre l'interface indefiniment.
                response = requests.post(
                    f"{API_BASE}/chat-rag",
                    json={"question": prompt, "n_results": 3},
                    timeout=120,
                )
            except requests.RequestException as e:
                st.error(f"Connexion impossible à {API_BASE} : {e}")
                st.stop()

            if response.status_code == 200:
                data = response.json()
                reponse_ia = data.get("reponse", "(réponse vide)")
                st.markdown(reponse_ia)

                sources = data.get("sources", [])
                if sources:
                    with st.expander("📄 Voir les notes CRM utilisées"):
                        for source in sources:
                            meta = source.get("metadata", {})
                            score = source.get("score")
                            suffixe = f" — similarité {score:.3f}" if score else ""
                            st.caption(
                                f"**Client : {meta.get('client', 'Inconnu')}** "
                                f"(le {meta.get('date', 'Inconnue')}){suffixe}"
                            )
                            st.info(source.get("doc", ""))

                st.session_state.messages.append(
                    {"role": "assistant", "content": reponse_ia}
                )
            else:
                # On affiche le message d'erreur reel du backend au lieu de le
                # masquer derriere un "patientez 20 secondes" trompeur.
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Erreur {response.status_code} : {detail}")
