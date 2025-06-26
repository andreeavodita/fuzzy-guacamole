from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from gpt4all import GPT4All
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = GPT4All("mistral-7b-openorca.Q4_0.gguf")

# One-time FAISS index build or load
def create_or_load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("my_faiss_index"):
        return FAISS.load_local("my_faiss_index", embeddings)
    else:
        loader = DirectoryLoader("my_documents/", glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("my_faiss_index")
        return db

db = create_or_load_vectorstore()

@app.get("/chatbot/response")
def chatbot_response(prompt: str):
    query = prompt
    relevant_docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    final_prompt = f"""You are a helpful assistant. Use the information below to answer the question. Only use the information provided.

Context:
{context}

Question:
{query}

Answer:"""

    response = model.generate(final_prompt)

    return {
        "response": response
    }
