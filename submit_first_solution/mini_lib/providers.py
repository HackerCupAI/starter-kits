import os
from dataclasses import dataclass
import openai

@dataclass
class Provider:
    client: openai.OpenAI
    model: str

PROVIDERS = {
    "cerebras": Provider(
        client=openai.OpenAI(
            base_url="https://api.cerebras.ai/v1/", 
            api_key=os.getenv("CEREBRAS_API_KEY")
        ),
        model="llama3.1-70b"
    ),
    "octo": Provider(
        client=openai.OpenAI(
            base_url="https://text.octoai.run/v1",
            api_key=os.environ.get("OCTO_API_KEY")
        ),
        model="meta-llama-3.1-70b-instruct"
    ),
    "groq": Provider(
        client=openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        ),
        model="llama-3.1-70b-versatile"
    ),
    "fireworks": Provider(
        client=openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1", 
            api_key=os.getenv("FIREWORKS_API_KEY")
        ),
        model="accounts/fireworks/models/llama-v3p1-70b-instruct"
    ),
    "together": Provider(
        client=openai.OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        ),
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    ),
    "openai": Provider(
        client=openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        model="gpt-4o-mini"
    )
}

client = PROVIDERS["cerebras"].client
model = PROVIDERS["cerebras"].model
prompt= "hola"

# new (text, images, audio)
response = client.chat.completions.create(
    model=model,
    max_tokens=2048,
    temperature=0.7,
    messages=[
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ],
)

# old (text only)

response = client.chat.completions.create(
    model=model,
    max_tokens=2048,
    temperature=0.7,
    messages=[
        {"role": "user", "content": prompt},
    ],
)