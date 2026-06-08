import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

prompt = "Say hello in one short sentence as a friendly spider robot named Zeus."

print(f"Sending prompt to {MODEL}...")

resp = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 64},
    },
    timeout=60,
)

resp.raise_for_status()

data = resp.json()
print("Raw response JSON keys:", list(data.keys()))
print("\nModel reply:\n")
print(data.get("response", "<no response>"))
