import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from dotenv import load_dotenv
from schema import *
from fastapi.responses import StreamingResponse
import logging
import json
import tiktoken
from asyncio import Semaphore
from get_cost_backoff import *

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

@app.post("/chat/tools")
async def answer_with_tools(request:Request, user_questions:str)-> str:
    client = request.app.state.llm
    messages = [{"role":"user", "content":user_questions}]

    first = await client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = messages,
        tools = tools
    )

    msg = first.choices[0].message

    if not msg.tool_calls:
        return msg.content
    
    messages.append(msg)

    for call in msg.tool_calls:
        args = json.loads(call.function.parameters)
        if call.function.name == "city_weather":
            result = city_weather(**args)
        else:
            result = {"error":f"unknown tool call {call.function.name}"}

        messages.append(
            {
                "role":"tool",
                "tool_call_id":call.id,
                "content":json.dumps(result)
            }
        )

    second = await client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = messages
    )

    return second.choices[0].message.content



@app.post("/chat/embeddings")
async def get_vector(body:Ask):
    embedder = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    resp = await embedder.embeddings.create(
        model="gemini-embedding-001",
        input=body.query
    )

    print({"resp":resp})

    res = {
        "vector":resp.data[0].embedding,
        "dimensions":len(resp.data[0].embedding),
        "first_four_elements":resp.data[0].embedding[:4]
    }

    return res

@app.post("/chat/batch_embedding")
async def batch_embedding(body:AskList):
    embedder = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    resp = await embedder.embeddings.create(
        model="gemini-embedding-001",
        input=body.queries
    )

    vector = [d.embedding for d in resp.data]
    num_of_embeddings = len(vector)
    embedding_dimension = len(vector[0])

    res = {
        "vector":vector,
        "num_of_embeddings":num_of_embeddings,
        "embedding_dimension":embedding_dimension
    }

    return res

@app.post("/count_tokens")
async def count_tokens(query:Ask):
    enc = tiktoken.get_encoding("o200k_base")
    n_tokens = len(enc.encode(query.query))
    return n_tokens

# writing a fast api that will return 


@app.post('/chat/stream-output')
async def chat_stream_output(body:Ask):
    client:AsyncOpenAI = app.state.llm
    sem:Semaphore = app.state.sem

    messages = [{"role":"user", "messages":body.query}]

    async def generate():
        async with sem:
            try:
                stream = await stream_output(client, messages)
            except Exception as e:
                logger.error(e)
                yield "[error: Service busy please retry]"
                return
            
            usage = None
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                if chunk.usage:
                    usage = chunk.usage
            
            if usage:
                logger.info(
                    f"llm_call model = {MODEL} in = {usage.prompt_token} out = {usage.completion_token} total = {usage.total_tokens} cost = {count_usd(MODEL, usage)}")
            
    return StreamingResponse(generate(), media_type="text/plain")