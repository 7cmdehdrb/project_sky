import requests
import json

# Ollama 서버 URL
url = "http://192.168.0.164:11434/api/tags"

try:
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        models = [m["name"] for m in data["models"]]
        print(f"Accessible: Yes")
        print(f"Models ({len(models)}): {models}")
    else:
        print("Accessible: No")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Accessible: No - Error: {str(e)}")
