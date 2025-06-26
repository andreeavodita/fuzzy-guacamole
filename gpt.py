from fastapi import FastAPI
from pydantic import BaseModel
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai
import os

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

# One-time FAISS index build or load
def create_or_load_vectorstore():
    if os.path.exists("my_faiss_index"):
        return FAISS.load_local("my_faiss_index", OpenAIEmbeddings())
    else:
        loader = DirectoryLoader("my_documents/", glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("my_faiss_index")
        return db

db = create_or_load_vectorstore()

openai.api_key = os.getenv("OPENAI_API_KEY")  # Or set directly if testing

@app.post("/chatbot/response")
def chatbot_response(prompt_data: Prompt):
    query = prompt_data.prompt
    relevant_docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    final_prompt = f"""You are a helpful assistant. Use the information below to answer the question. Only use the information provided.

Context:
{context}

Question:
{query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return {
        "response": response["choices"][0]["message"]["content"]
    }
