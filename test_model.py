# test_ollama.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = genai.GenerativeModel(os.getenv("BASE_MODEL_NAME"))

print("Testing Google AI connection...")
try:
    response = client.generate_content("Say hello")
    print("Success!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")