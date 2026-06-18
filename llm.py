import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

GROQ_URL = os.getenv("GROQ_URL")

async_client = AsyncOpenAI(
    api_key=GROQ_URL,
    base_url="https://api.groq.com/openai/v1"
)