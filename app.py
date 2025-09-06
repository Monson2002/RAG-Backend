import os

from openai import OpenAI
from flask_cors import CORS
from chroma import get_chroma_collection
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["*"])

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

collection = get_chroma_collection()

# @app.route("/api/documents/", methods=["POST"])
# def add_documents():
#     try:
#         request_body = request.get_json()
#         ids = request_body.get('ids')
#         documents = request_body.get('documents')
#         metadatas = request_body.get('metadatas')
        
#         collection = get_chroma_collection()

#         collection.add(
#             ids=ids,
#             documents=documents,
#             metadatas=metadatas,
#         )
#         return jsonify({"message": "Documents added successfully", "ids": ids}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# ("How does the process of excretion take place inside the human body. Explain")

@app.route("/api/ask/", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":  # Preflight request
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
        return response, 200
    try: 
        data = request.get_json()  
        query = data.get("query", "")
        retrieved_info = collection.query(
            query_texts=[query],
            n_results=5 
        )
        context = retrieved_info['documents']
        prompt = """Use the following pieces of context to answer the question at the end. If the answer is not explicitly in the context, you can answer based on your knowledges. 
        Use ten sentences maximum and keep the answer as concise as possible.
    
        Context: {context}
    
        Question: {query}
    
        Helpful Answer:
        """.format(context=context, query=query)
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return jsonify({
            "query": query,
            "answer": response.choices[0].message.content
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)