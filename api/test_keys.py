from dotenv import load_dotenv
import os
import openai

load_dotenv()

# Test OpenAI key
try:
    client = openai.OpenAI(api_key=os.getenv("API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("✅ OpenAI API key is working!")
except Exception as e:
    print(f"❌ OpenAI API key error: {e}")

# Check if keys are loaded
print(f"OpenAI key loaded: {bool(os.getenv('API_KEY'))}")
print(f"Amadeus key loaded: {bool(os.getenv('AMADEUS_API_KEY'))}")
print(f"Amadeus secret loaded: {bool(os.getenv('AMADEUS_API_SECRET'))}")