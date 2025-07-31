from chromadb.config import Settings
import chromadb
from chromadb import HttpClient 

client = HttpClient(host="155.248.175.96", port=8000)  

collections = client.list_collections()
for col in collections:
    print(col.name)