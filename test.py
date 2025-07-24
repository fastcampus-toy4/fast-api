import requests
import chromadb
client = chromadb.HttpClient(host="155.248.175.96", port=8000)
for col in client.list_collections():
    print(col['name'], col['id'])

res = requests.options("http://155.248.175.96:8000/ask")
print(res.status_code)
print(res.text)