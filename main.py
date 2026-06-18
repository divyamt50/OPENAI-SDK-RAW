import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from dotenv import load_dotenv
from schema import Ask

load_dotenv()

GROQ_URL = os.getenv("GROQ_URL")
history = [{"role":"system","content":"You are concise."}]


@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.llm = AsyncOpenAI(
        api_key=GROQ_URL,
        base_url="https://api.groq.com/openai/v1"
    )

    yield

    await app.state.llm.close()

app = FastAPI(lifespan=lifespan)

@app.get('/')
async def health_check():
    return {"msg":"everything fine"}

@app.post('/query')
async def ask_question(request:Request, body:Ask):
    client = request.app.state.llm

    resp = await client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [{"role":"user", "content":f"{body.query}"}]
    )

    return {
        "response":resp.choices[0].message.content
    }

@app.post('/turn_query')
async def turn(request:Request, user_query:Ask) -> str:
    user_input = {"role":"user", "content":user_query.query}
    history.append(user_input)
    client = request.app.state.llm

    resp = await client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = history
    )

    response_final = resp.choices[0].message.content

    history.append({"role":"assistant", "content":response_final})

    return response_final