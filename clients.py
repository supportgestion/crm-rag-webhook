# -*- coding: utf-8 -*-
"""
Noms de clients — forme officielle unique. SOURCE UNIQUE DE VERITE.

Importe par import_notes.py (notes Gemini locales) et par webhook_server.py
(notes Zoho). Toute modification se fait ICI et nulle part ailleurs.

Deux roles :
  1. Une seule graphie par client, quelle que soit celle du titre Gemini.
  2. ALIAS CONTACT -> SOCIETE. Le webhook Zoho ne peut pas transmettre le nom
     du compte (champ lookup non pris en charge sur un declencheur Remarques),
     il envoie le nom du contact. La table le rattache a sa societe, sinon
     /stats/clients se remplit de noms de personnes.

A committer dans supportgestion/crm-rag-webhook pour que Railway le deploie.
"""

import logging
import re
import unicodedata

log = logging.getLogger("crm-rag")

# Repli quand aucun nom n'est disponible. Valeur identique a celle deja
# utilisee par webhook_server.py, pour ne pas creer un second panier de rebut.
CLIENT_INCONNU = "Client Inconnu"


def cle_client(nom: str) -> str:
    """Reduit un nom de client a une cle comparable : sans accents,
    sans ponctuation, en minuscules. 'BEST KEBAB DISTRIBUTION' et
    'Best kebab distribution' donnent la meme cle."""
    if not nom:
        return ""
    k = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    k = re.sub(r"[^A-Za-z0-9]+", "_", k).strip("_").lower()
    return k


# Cle normalisee -> nom affiche officiel.
# C'est la reference unique : stats, alias, affichage.
CLIENTS_CANONIQUES = {
    "amphytro": "AMPHYTRO",
    "banette_bargemon": "Banette Bargemon",
    "best_kebab_distribution": "Best Kebab Distribution",
    "brittany_ferries": "Brittany Ferries",
    "camping_bel_air": "Camping Bel Air",
    "chez_dupont": "Chez Dupont",
    "delices_de_manon_et_thomas": "Delices de Manon et Thomas",
    "distri_pizza": "Distri Pizza",
    "groupe_aminian": "Groupe Aminian",
    "hanoi": "HANOI",
    "jespp": "JESPP",
    "l_arch_1972": "L'Arch 1972",
    "la_pause_gourmande": "La Pause Gourmande",
    "maison_canaguette": "Maison Canaguette",
    "ora": "ORA",
    "piccirillo": "Piccirillo",
    "restaurant_d_ete": "Restaurant d'Ete",
    "sarl_lacab": "SARL LACAB",
    "sarl_louno": "SARL LOUNO",
    "trinco": "Trinco",
    "vauban": "VAUBAN",

    # Note sans nom de societe.
    "marine_lesage": "Support Gestion+ (Marine Lesage)",

    # -----------------------------------------------------------------------
    # ALIAS CONTACT -> SOCIETE
    # -----------------------------------------------------------------------
    # Le webhook Zoho envoie le nom du CONTACT dans le champ "client".
    # Chaque interlocuteur doit figurer ici, sinon son nom apparaitra tel quel
    # dans /stats/clients (avec un warning dans les logs Railway).
    #
    # A COMPLETER — remplacer par les vraies societes :
    # "jihad_zakhour": "???",
    # "leo_berthault": "???",
    #
    # Exemple de la forme attendue :
    # "robin_bassin": "The Bouillon Of Paris",
}


def resoudre_client(nom):
    """Nom officiel d'un client, ou None si rien d'exploitable.

    Un nom inconnu est renvoye TEL QUEL, pas remplace par un sentinelle : un
    nouveau client ne doit pas perdre son identite dans l'entete indexee, sous
    peine de devenir introuvable par le RAG. Le warning est le signal qu'il
    faut completer la table. C'est le comportement de l'ancienne
    extraire_client(), conserve a l'identique.
    """
    brut = (str(nom).strip() if nom else "")
    if not brut:
        return None
    officiel = CLIENTS_CANONIQUES.get(cle_client(brut))
    if officiel:
        return officiel
    log.warning("Client hors table canonique, conserve tel quel : %r", brut)
    return brut


def clients_officiels():
    """Noms affiches, pour le controle du dry-run."""
    return set(CLIENTS_CANONIQUES.values())
