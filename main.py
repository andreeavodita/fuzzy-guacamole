from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str

@app.get("/chatbot/response")
def chatbot_response(prompt_data: Prompt):
    prompt = prompt_data.prompt
    response = prompt[::-1]  # Example processing
    return {"response": response}
