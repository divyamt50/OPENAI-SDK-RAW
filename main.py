import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

GROQ_URL = os.getenv("GROQ_URL")

client = OpenAI(
    api_key=GROQ_URL,
    base_url="https://api.groq.com/openai/v1"
)

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role":"user", "content":"What is deadpool's real name"}]
)

print(resp.choices[0].message.content)