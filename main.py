from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/chatbot/response")
def chatbot_response(prompt_data: Prompt):
    prompt = prompt_data.prompt
    response = prompt[::-1]  # Example processing
    return {"response": response}
