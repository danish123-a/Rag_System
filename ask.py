import requests

BASE_URL = "http://localhost:8000"

def ask(query, top_k=5):
    payload = {
        "query": query,
        "top_k": top_k
    }
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    return r.json()

if __name__ == "__main__":
    result = ask("What does this project do?")
    print(result)
