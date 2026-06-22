import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from dotenv import load_dotenv
from schema import Ask
from fastapi.responses import StreamingResponse
import logging
import json

load_dotenv()

GROQ_URL = os.getenv("GROQ_URL")
history = [{"role":"system","content":"You are concise."}]

logger = logging.getLogger(__name__)

def log_cost(usage):
    logger.info(
        "Prompt tokens, completion tokens, Token usage",
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens
    )

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
async def turn(request:Request, user_query:Ask):
    user_input = {"role":"user", "content":user_query.query}
    history.append(user_input)
    client = request.app.state.llm

    resp = await client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = history,
        temperature = 0.7,
        max_tokens = 500
    )

    response_final = resp.choices[0].message.content

    history.append({"role":"assistant", "content":response_final})

    user_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens
    total_tokens = resp.usage.total_tokens
    role = resp.choices[0].message.role
    finish_reason = resp.choices[0].finish_reason

    return {
        "response":response_final,
        "user_tokens":user_tokens,
        "completion_tokens":completion_tokens,
        "total_tokens":total_tokens,
        "role":role,
        "finish_reason":finish_reason
    }



@app.post("/streaming-responses")
async def get_streaming_response(request:Request, query:Ask):
    client = request.app.state.llm
    async def streaming_response():
        stream = await client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages = [{"role":"user", "content":query.query}],
            stream = True,
            stream_options = {"include_usage":True}
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                log_cost(chunk.usage)
        
    return StreamingResponse(streaming_response(), media_type = "text/plain")

def city_weather(city:str, unit:str = "celsius")-> dict:
    return {"city":city, "temp": 31, "unit":unit, "conditions":"humid"}

tools = [
    {
        "type":"function",
        "function":{
            "name":"city_weather",
            "description":"function to get weather condition of a given city",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"name of the city"
                    },
                    "unit":{
                        "type":"string",
                        "enum":["celsius", "fahrenheit"],
                        "description":"Temperature unit"
                    }
                },
                "required":["city"]
            }
        }
    }
]