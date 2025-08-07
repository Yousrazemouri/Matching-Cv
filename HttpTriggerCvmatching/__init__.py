import azure.functions as func
import logging
import os
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# ✅ Lis les variables d'environnement définies dans Azure App Settings
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Exemple : https://matchingcv.openai.azure.com/
api_key = os.getenv("AZURE_OPENAI_KEY")        # Exemple : clé secrète

# ✅ Vérifie que les variables sont bien définies
if not endpoint or not api_key:
    raise ValueError("Les variables d'environnement AZURE_OPENAI_ENDPOINT ou AZURE_OPENAI_KEY sont manquantes.")

# ✅ Initialise le client OpenAI avec clé API
client = OpenAIClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

@app.route(route="http_triggerCvmatching")
def http_triggerCvmatching(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('HTTP trigger fonction appelée.')

    try:
        prompt = req.params.get('prompt')
        if not prompt:
            req_body = req.get_json()
            prompt = req_body.get('prompt')

        if not prompt:
            return func.HttpResponse(
                "Veuillez fournir un prompt dans la requête.",
                status_code=400
            )

        # ✅ Appel OpenAI
        response = client.get_chat_completions(
            deployment_id="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return func.HttpResponse(answer, status_code=200)

    except Exception as e:
        # ✅ Ajoute une trace pour debug précis
        logging.error(f"Erreur pendant l'appel OpenAI: {e}", exc_info=True)
        return func.HttpResponse(f"Erreur serveur : {str(e)}", status_code=500)
