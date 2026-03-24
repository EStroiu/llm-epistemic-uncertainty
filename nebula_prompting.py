from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

NEBULA_BASE_URL = 'https://nebula.cs.vu.nl/api/'
_NEBULA_CLIENT = None


def get_nebula_client() -> OpenAI:
    global _NEBULA_CLIENT

    if _NEBULA_CLIENT is None:
        nebula_api_key = os.getenv('NEBULA_API_KEY')
        if not nebula_api_key:
            raise RuntimeError(
                "Missing NEBULA_API_KEY. Create a .env file in the project root with:\n"
                "NEBULA_API_KEY=YOUR_KEY_HERE"
            )

        _NEBULA_CLIENT = OpenAI(base_url=NEBULA_BASE_URL, api_key=nebula_api_key)

    return _NEBULA_CLIENT


def get_nebula_models():
    nebula = get_nebula_client()
    models = []
    for model in nebula.models.list().data:
        models.append(model.id)
    return models

def prompt_nebula(model, system_prompt, user_prompt, configs=None):
    nebula = get_nebula_client()
    prompt_parameters = {
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
    }

    if configs:
        prompt_parameters.update(configs)

    response = nebula.chat.completions.create(**prompt_parameters)

    return response

if __name__=='__main__':
    available_models = get_nebula_models()
    print(f"Models: {available_models}\n\n")
    
    config = {
        "max_tokens": 25
    }

    response = prompt_nebula("deepseek-r1:1.5b", "You are a helpful assistant", "What is the capital of Brazil?", config)
    print(f"Response: {response.choices[0].message.content}\n\n")
    print(f"Usage stats: {response.usage}")
