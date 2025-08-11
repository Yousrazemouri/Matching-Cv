import azure.functions as func
import logging
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import fitz  # PyMuPDF
from requests_toolbelt.multipart import decoder
import requests
import json
from collections import defaultdict

# Charger les variables d'environnement
load_dotenv()

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

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('HTTP trigger fonction appelée.')

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }

    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=headers)

    if req.method == "GET":
        html_content = """
        <html>
        <head><title>Analyse de CV</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>Uploader votre CV</h2>
            <form action="" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf" required>
                <br><br>
                <button type="submit">Analyser</button>
            </form>
        </body>
        </html>
        """
        return func.HttpResponse(html_content, mimetype="text/html", headers=headers)

    if req.method == "POST":
        try:
            content_type = req.headers.get('Content-Type')
            if not content_type or 'multipart/form-data' not in content_type:
                return func.HttpResponse("Le contenu doit être multipart/form-data.", status_code=400, headers=headers)

            body = req.get_body()
            multipart_data = decoder.MultipartDecoder(body, content_type)

            file_bytes = None
            for part in multipart_data.parts:
                if b'application/pdf' in part.headers.get(b'Content-Type', b''):
                    file_bytes = part.content
                    break

            if not file_bytes:
                return func.HttpResponse("Fichier PDF non trouvé.", status_code=400, headers=headers)

            # Extraction du texte
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            cv_text = "".join([page.get_text() for page in doc])

            if not cv_text.strip():
                return func.HttpResponse("Impossible d'extraire le texte du PDF.", status_code=400, headers=headers)

            # ✅ Prompt enrichi
            prompt = f"""
Tu es un extracteur de données techniques pour des CV.

Objectif :
- Retourner UNIQUEMENT un objet JSON valide de la forme :
{{"profile_type":"...", "technical_skills":["comp1","comp2","..."]}}
- Pas de texte additionnel, pas de balises, pas de commentaires.

Définition des champs :
- "profile_type" : un libellé court et précis du type de profil principal déduit du CV.
- "technical_skills" : liste des compétences techniques (langages, frameworks, bibliothèques, outils, services cloud, bases de données, systèmes, protocoles/normes, plateformes).

Règles pour "profile_type" :
- Déduis le type principal à partir des intitulés de poste, missions, réalisations et technologies dominantes (donne plus de poids aux expériences récentes).

Règles pour "technical_skills" :
- Inclure uniquement des compétences techniques (pas de soft skills, langues, diplômes, méthodes : Agile, Scrum, etc.).
- Tu peux DÉDUIRE des compétences si l'usage est clairement impliqué par les expériences/missions.
  Exemples :
  • "Développement d'APIs REST en Python sur Azure Functions" → ["Python", "Azure Functions", "REST API"]
  • "Mise en place de pipelines CI/CD avec Azure DevOps" → ["Azure DevOps", "CI/CD"]
  • "Modèles de classification avec scikit-learn" → ["scikit-learn"]
- N'invente rien qui ne soit pas soutenu par le texte.
- Pas de doublons, liste plate de chaînes.

Renvoie UNIQUEMENT le JSON demandé.

Texte du CV :
{cv_text}
"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content
            parsed = json.loads(answer)
            skills = parsed["technical_skills"]

            # Appel Azure AI Search
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
            search_response = requests.post(url, headers=headers_search, json=query)
            search_response.raise_for_status()
            search_results = search_response.json()

            # Agrégation des directions
            directions = []
            for item in search_results.get("@search.answers", []):
                text = item["text"]
                score = item["score"]
                start = text.find('"direction":')
                if start != -1:
                    start_quote = text.find('"', start + 12)
                    end_quote = text.find('"', start_quote + 1)
                    direction = text[start_quote + 1:end_quote]
                    directions.append((direction, score))

            agg = defaultdict(list)
            for direction, score in directions:
                agg[direction].append(score)

            summary = "\n".join([f"{d} | Score moyen: {sum(s)/len(s):.3f}" for d, s in agg.items()])

            html_result = f"""
            <html>
            <head><title>Résultat de l'analyse</title></head>
            <body style="font-family: Arial; padding: 20px;">
                <h2>Résultat de l'analyse</h2>
                <pre>{answer}</pre>
                <h3>Matching avec les directions ICUBE</h3>
                <pre>{summary}</pre>
                <br>
                <a href="">⬅ Retour</a>
            </body>
            </html>
            """
            return func.HttpResponse(html_result, mimetype="text/html", headers=headers)

        except Exception as e:
            logging.error(f"Erreur pendant le traitement : {e}", exc_info=True)
            return func.HttpResponse(f"Erreur serveur : {str(e)}", status_code=500, headers=headers)

    return func.HttpResponse("Méthode non autorisée", status_code=405, headers=headers)
