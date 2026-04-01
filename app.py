from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

chatModel = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answering_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("msg") if data else None

    if not msg:
        return jsonify({"error": "No message received"}), 400

    try:
        response = rag_chain.invoke({"input": msg})
        return jsonify({
            "response": response.get("answer", "")
        })
    except Exception as e:
        print(e)
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)