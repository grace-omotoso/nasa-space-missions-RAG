from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    SYSTEM_PROMPT = """You are a helpful NASA Expert with vast knowledge about space missions"""
 
    # TODO: Create OpenAI Client
    client = OpenAI(
            api_key=openai_key,
            base_url="https://openai.vocareum.com/v1"
    )
    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *conversation_history,
    {
        "role": "user",
        "content": f"{user_message}\n\nContext:\n{context}"
    }
]

    # TODO: Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    # TODO: Return response
    return response.choices[0].message.content
