from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    input: str

@app.post("/process")
def process(data: InputData):
    input_string = data.input

    # Example processing
    output_string = input_string[::-1]  # Just an example

    return {"output": output_string}
