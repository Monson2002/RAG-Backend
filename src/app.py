from dotenv import load_dotenv
from src.llm import simple_RAG
from src.search import Retriever
from src.dataloader import load_data
from src.vector_store import VectorStore
from src.embedding import EmbeddingManager
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__=="__main__":
    docs = load_data('data')
   
    embedding_manager = EmbeddingManager()
    
    # Chunking
    chunks = embedding_manager.split_docs(docs)

    # Chunk to Embed
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    # # Normalizing distances
    # embeddings = np.array(embeddings)
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Creating a Vector Store
    vectorStore = VectorStore(collection_name='RP-RAG', persist_dir='./data/vector_store/') #ensure collection_name is 3-512 characters
    # vectorStore.add_docs(docs=chunks, embeddings=embeddings)

    # Creating a Retriever
    retriever = Retriever(vector_store=vectorStore, embedding_manager=embedding_manager)

    question = "How does MAE use autoencoder, is it for attack or for defense?"
    # retrieved_docs = retriever.retrieve(question)

    # LLM
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

    # llm.invoke('Write me a song')
    llm_ans = simple_RAG(question, retriever, llm, 3)

    print(llm_ans)
