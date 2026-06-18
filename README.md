# Groq API Chat Completion Example

A simple Python script demonstrating how to use the Groq API with the OpenAI-compatible client to generate chat completions using the `llama-3.1-8b-instant` model.

## Prerequisites

- Python 3.8+
- Groq API Key
- Required Python packages:
  - openai
  - python-dotenv

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/groq-chat-example.git
cd groq-chat-example
```

### 2. Install dependencies

```bash
pip install openai python-dotenv
```

### 3. Create a `.env` file

Create a `.env` file in the project root:

```env
GROQ_URL=your_groq_api_key_here
```

> Note: Although the variable is named `GROQ_URL` in this example, it actually stores your Groq API key.

## Code

```python
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
    messages=[
        {
            "role": "user",
            "content": "What is deadpool's real name"
        }
    ]
)

print(resp.choices[0].message.content)
```

## Running the Script

```bash
python app.py
```

## Example Output

```text
Deadpool's real name is Wade Wilson.
```

## Project Structure

```text
.
├── app.py
├── .env
├── requirements.txt
└── README.md
```

## Requirements File

Create a `requirements.txt` file:

```txt
openai
python-dotenv
```

## Model Used

- `llama-3.1-8b-instant`

## API Endpoint

```text
https://api.groq.com/openai/v1
```

## License

This project is licensed under the MIT License.