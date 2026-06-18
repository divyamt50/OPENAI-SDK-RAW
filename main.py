from llm import async_client
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Ask(BaseModel):
    query:str

@app.post('/get_response')
async def get_response(query:Ask):
    resp = await async_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":query.query}]
    )

    return {"answer":f"{resp.choices[0].message.content}"}