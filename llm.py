import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI

GROQ_URL = os.getenv("GROQ_URL")

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.llm = AsyncOpenAI(
        api_key=GROQ_URL,
        base_url="https://api.groq.com/openai/v1"
    )

    yield
    await app.state.llm.close()

app = FastAPI(lifespan=lifespan)