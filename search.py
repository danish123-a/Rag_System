import requests

BASE_URL = "http://localhost:8000"

def search(query, top_k=5):
    payload = {
        "query": query,
        "top_k": top_k
    }
    r = requests.post(f"{BASE_URL}/search", json=payload)
    return r.json()

if __name__ == "__main__":
    result = search("example document", top_k=3)
    print(result)
