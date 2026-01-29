import os
from dotenv import load_dotenv

load_dotenv()  # charge le fichier .env

from google import genai

# Vérification
print("Clé GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="models/gemini-flash-latest",
    contents="Bonjour, dis-moi quelque chose de simple"
)
print(response.text)
