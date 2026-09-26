import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "your_gemini_api_key_here"))

response = client.models.generate_content(
    model="gemini-1.5-flash",  # or "gemini-1.5-flash", "gemini-1.5-pro"
    contents="Explain how AI works in a few words",
)

print(response.text)