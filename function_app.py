import azure.functions as func
import logging
import os
from azure.ai.openai import OpenAIClient
from azure.identity import DefaultAzureCredential

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Configure Azure OpenAI client
endpoint = os.getenv("https://matchingcv.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview")
key = os.getenv("AZURE_OPENAI_KEY")

client = OpenAIClient(endpoint, credential=key)

@app.route(route="http_triggerCvmatching")
def http_triggerCvmatching(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    prompt = req.params.get('prompt')
    if not prompt:
        try:
            req_body = req.get_json()
            prompt = req_body.get('prompt')
        except ValueError:
            pass

    if not prompt:
        return func.HttpResponse(
            "Please pass a 'prompt' in the query string or in the request body",
            status_code=400
        )
    
    try:
        # Appel à Azure OpenAI
        response = client.get_chat_completions(
            deployment_id="your-deployment-name",  # Remplace par le nom de ton déploiement
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return func.HttpResponse(answer, status_code=200)
    except Exception as e:
        logging.error(f"Azure OpenAI call failed: {e}")
        return func.HttpResponse(f"Error calling Azure OpenAI: {e}", status_code=500)
