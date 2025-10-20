from src.search import Retriever
from langchain_google_genai import ChatGoogleGenerativeAI

def simple_RAG(query: str, retriever: Retriever, llm: ChatGoogleGenerativeAI, top_k: int = 5):
    results = retriever.retrieve(query=query)
    context = "\n\n".join([i['document'] for i in results])
    if not context:
        return 'No relevant context found.'
    
    template = """
            You are a helpful assistant, for reasearch papers. You haev research papers as data, so you need to be aware of how a research paper is structured, (like where to find the authors of the paper and stuff etc.). Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use five sentences maximum and keep the answer as concise as possible.
            Context: {context}
            Question: {query}
            Helpful Answer:
        """
    
    response = llm.invoke([template.format(context=context, query=query)])
    return response.content