import os
import chromadb

from dotenv import load_dotenv
from flask import current_app, g
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

load_dotenv()

_chroma_client: ClientAPI | None = None
_chroma_collection: Collection | None = None

def get_chroma_client() -> ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
    return _chroma_client
    
def get_chroma_collection() -> Collection:
    global _chroma_collection
    if _chroma_collection is None:
        chroma_client = get_chroma_client()
        _chroma_collection = chroma_client.get_collection(name='ncert_class_X')
    return _chroma_collection