import azure.functions as func
import logging
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import fitz  # PyMuPDF
import requests
import json
from collections import defaultdict
from PIL import Image
import pytesseract
import io

# Charger les variables d'environnement
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\tesseract\tesseract.exe"  # Ajuster si nécessaire

# Azure OpenAI
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("OPENAI_API_VERSION") or "2023-05-15"

# Azure AI Search
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
search_index = os.getenv("AZURE_SEARCH_INDEX_NAME")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version
)

def extraire_texte_avec_ocr(pdf_bytes):
    """Extrait le texte d'un PDF, avec fallback OCR si pages images."""
    texte_total = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        texte = page.get_text()
        if texte.strip():
            texte_total.append(texte)
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            texte_ocr = pytesseract.image_to_string(img, lang="fra+eng")
            texte_total.append(texte_ocr)

    return "\n".join(texte_total)

def analyser_cv(cv_text):
    """Analyse un CV et retourne profile_type + skills + direction avec règles métier."""
    prompt = f"""
Tu es un extracteur de données pour des CV.

Objectif :  
- Retourner UNIQUEMENT un objet JSON valide de la forme :  
{{  
  "profile_type": "...",           # Intitulé de poste précis  
  "profile_category": "...",       # Une des valeurs suivantes : "Technique", "RH", "Finance", "Autre"  
  "technical_skills": ["comp1","comp2",...]  # Liste des compétences techniques si "profile_category" est "Technique", sinon liste vide  
}}

Consignes importantes :  
- "profile_type" est l’intitulé de poste principal détecté dans le CV.  
- "profile_category" doit être :  
  - "Technique" pour les profils liés au développement, infrastructure, data, cloud, etc.  
  - "RH" pour les profils liés aux ressources humaines, recrutement, talent acquisition, etc.  
  - "Finance" pour les profils liés à la comptabilité, contrôle de gestion, audit, etc.  
  - "Autre" pour tout autre métier.   
- Pour les profils "Technique","RH","Finance", liste uniquement les compétences techniques (langages, outils, frameworks, cloud…).  
- Ne pas inclure de soft skills ou compétences non techniques dans "technical_skills".

Texte du CV :  
{cv_text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        return json.loads(content)

    except json.JSONDecodeError:
        logging.error(f"Réponse non JSON : {content}")
        return {"error": "Impossible de parser le JSON"}
    except Exception as e:
        logging.error(f"Erreur analyse CV : {e}")
        return {"error": str(e)}

def rechercher_direction(skills):
    """Recherche la direction la plus pertinente via Azure AI Search."""
    try:
        url = f"{search_endpoint}/indexes/{search_index}/docs/search?api-version=2023-07-01-Preview"
        query = {
            "search": " ".join(skills),
            "queryType": "semantic",
            "semanticConfiguration": "rag-1754943219165-semantic-configuration",
            "top": 5,
            "select": "chunk,title",
            "answers": "extractive",
            "captions": "extractive"
        }
        headers_search = {
            "Content-Type": "application/json",
            "api-key": search_api_key
        }
        resp = requests.post(url, headers=headers_search, json=query)
        resp.raise_for_status()

        directions = []
        for item in resp.json().get("@search.answers", []):
            text = item["text"]
            score = item["score"]
            start = text.find('"direction":')
            if start != -1:
                start_quote = text.find('"', start + 12)
                end_quote = text.find('"', start_quote + 1)
                direction = text[start_quote + 1:end_quote]
                directions.append((direction, score))

        if not directions:
            return None

        agg = defaultdict(list)
        for direction, score in directions:
            agg[direction].append(score)
        return max(agg.items(), key=lambda x: sum(x[1]) / len(x[1]))[0]

    except Exception as e:
        logging.error(f"Erreur recherche direction : {e}")
        return None

def main(myblob: func.InputStream):
    """Déclenché automatiquement par l'ajout d'un fichier dans le conteneur Blob."""
    logging.info(f"📂 Blob détecté : {myblob.name}, taille : {myblob.length} octets")

    try:
        # Lire le contenu du blob (PDF)
        blob_data = myblob.read()
        texte = extraire_texte_avec_ocr(blob_data)

        if not texte.strip():
            logging.warning("⚠️ Aucun texte détecté même avec OCR")
            return

        # Analyse avec Azure OpenAI
        analyse = analyser_cv(texte)
        if "error" in analyse:
            logging.error(f"Erreur d'analyse pour {myblob.name}: {analyse['error']}")
            return

        # Recherche direction via Azure Search
        direction = rechercher_direction(analyse["technical_skills"])

        # Log du résultat final
        resultat = {
            "file": myblob.name,
            "profile_type": analyse.get("profile_type"),
            "direction": direction
        }
        logging.info(f"✅ Résultat : {json.dumps(resultat, ensure_ascii=False)}")

    except Exception as e:
        logging.error(f"❌ Erreur traitement blob {myblob.name} : {e}", exc_info=True)
