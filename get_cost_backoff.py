import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse
from asyncio import Semaphore
from openai import(
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError
)
from schema import Ask
from contextlib import asynccontextmanager
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm")

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")
PRICES = {
    "llama-3.1-8b-instant":(0.00, 0.00)
    }
MODEL = "llama-3.1-8b-instant"
MAX_CONCURRENCY = 5
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)

def count_usd(model, usage):
    cin, cout = PRICES.get(model, (0.00, 0.00))
    final_price = (usage.prompt_token/1000) * cin + (usage.completion_token) * cout
    return final_price


@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.llm = await AsyncOpenAI(
        api_key=GEMINI_KEY,
        base_url=os.getenv("BASE_URL")
    )

    app.state.sem = Semaphore(MAX_CONCURRENCY)
    yield
    await app.state.llm.close()

app = FastAPI(lifespan=lifespan)

async def stream_output(client, messages):
    for attempt in range(4):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                stream_options={"include_usage":True}
            )
        except (BadRequestError, AuthenticationError):
            raise
        except APIStatusError as e:
            if e < 500:
                raise
        except RETRYABLE:
            pass
        if attempt < 3:
            wait = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning

