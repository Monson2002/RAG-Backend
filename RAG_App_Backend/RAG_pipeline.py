from typing import List, TypedDict
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

template = """
        You are a helpful assistant. Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use five sentences maximum and keep the answer as concise as possible.

            Context: {context}

            Question: {question}

            Helpful Answer:
        """

custom_rag_prompt = PromptTemplate.from_template(template)

# ---- Define state ----
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# ---- Retrieval function ----
def retrieve(state: State):
    similar_chunks = vector_store.similarity_search(state['question'], 5)
    return {"context": similar_chunks}

# ---- Generator function ----
def generate_answers(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state['context'])
    message = custom_rag_prompt.invoke(
        {"question": state['question'], "context": docs_content}
    )
    response = llm.invoke(message)
    return {"answer": response.content}

# ---- Build pipeline ----
graph_builder = StateGraph(State).add_sequence([retrieve, generate_answers])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def run_rag_pipeline(query: str):
    return graph.invoke({"question": query})