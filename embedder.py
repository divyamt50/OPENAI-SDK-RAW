import os
import asyncio
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector_async
from openai import AsyncOpenAI
from schema import AskList
from psycopg import AsyncConnection

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_URL")

embedder = AsyncOpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

async def embed(body:AskList) -> list[list[float]]:
    resp = await embedder.embeddings.create(
        model="gemini-embedding-001",
        input=body.queries
    )

    return [d for d in resp.data]

async def main():
    conn = await psycopg.AsyncConnection.connect(
        "postgresql://postgres:password@localhost:5432/postgres_db"
    )

    await conn.execute(
        """
        CREATE EXTENSION IF NOT EXISTS vector
        """
    )

    await register_vector_async(conn)

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents(
        id BIG SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding VECTOR(768)
        )
        """
    )

    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS doc_idx
        ON documents USING hnsw(embedding vector_cosine_ops)
        """
    )

    corpus = AskList(queries=[
        "VACUUM reclaims storage in Postgres.",
        "Celery runs background tasks."
    ])

    vector = await embed(corpus)

    for content, vec in zip(corpus.queries, vector):
        await conn.execute(
            "INSERT INTO documents(content, embedding) VALUES(%s,%s)",
            (content, vec)
        )
    
    await conn.commit()

    # retreive

    question = "How do i clean up tuples that aren't being used?"
    qvec = await embed(AskList(queries=[question]))

    async with conn.cursor as cur:
        await cur.execute(
            """
                SELECT content, 1 - (embedding <=> %s) AS SIMILARITY
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT 1
            """,(qvec[0], qvec[0])
        )

        for row in await cur.fetchall():
            print(f"Match score:{row[1]:.3f}, Text:{row[0]}")

    await conn.close()