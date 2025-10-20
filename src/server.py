from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.embedding import EmbeddingManager
from src.vector_store import VectorStore
from src.search import Retriever
from src.llm import simple_RAG
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ncert-rag.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize core components once at startup
embedding_manager = EmbeddingManager()
vector_store = VectorStore(collection_name="RP-RAG", persist_dir="./data/vector_store/")
retriever = Retriever(vector_store=vector_store, embedding_manager=embedding_manager)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class AskRequest(BaseModel):
    query: str
    n_results: int | None = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: AskRequest):
    k = payload.n_results or 5
    results = retriever.retrieve(payload.query, k=k)
    answer = simple_RAG(payload.query, retriever, llm, top_k=k)
    return {
        "answer": answer,
        "sources": [
            {
                "id": r["id"],
                "metadata": r["metadata"],
                "similarity_score": r["similarity_score"],
            }
            for r in results
        ],
    }


