import logging
logger = logging.getLogger("llm.cost")
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from schema import Ask

GROQ_URL = os.getenv("GROQ_URL")


PRICES = {
    "gpt-4o-mini":            (0.00015, 0.00060),
    "gpt-4o":                 (0.00250, 0.01000),
    "llama-3.1-8b-instant":   (0.0,     0.0),      # Groq free tier
    "llama-3.3-70b-versatile":(0.0,     0.0),
    "nomic-embed-text":       (0.0,     0.0),      # Ollama, local
}

def get_cost_estimation(model:str, input_token:int, output_token:int):
    pin, pout = PRICES.get(model, (0.0, 0.0))
    price = (input_token/1000) * pin + (output_token/1000) * pout
    return price

def log_usage(model:str, usage) -> float:
    c = get_cost_estimation(model, usage.prompt_tokens, usage.completion_tokens)
    logger.info(
        f"llm_call model {model}, input tokens {usage.prompt_tokens}, output tokens {usage.completion_tokens}, total tokens {usage.total_tokens}, cost {c:.6f}"
    )
client = AsyncOpenAI(api_key=GROQ_URL,
        base_url="https://api.groq.com/openai/v1"
)

async def get_cost_call(msgs:Ask):
    resp = await client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    log_usage("gpt-4o-mini", resp.usage)