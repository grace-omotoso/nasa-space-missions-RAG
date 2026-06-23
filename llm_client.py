from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt telling LLM to generate answers based of provided documents
    SYSTEM_PROMPT = """You are a helpful NASA Expert with vast knowledge about space missions.
    Your answers must follow these rules:
    1. Base your answers ONLY on the retrieved context provided with each user message.
    2. Always cite your sources by referencing the 'Source' field from the context (e.g. "According to [source]...").
    3. If the context does not contain enough information to answer the question, say so clearly — do not make up facts.
    4. Be concise and precise. Use technical language appropriate for NASA mission documentation.
    5. If multiple sources support a point, cite all of them.
    """
 
    # Create OpenAI Client
    client = OpenAI(
            api_key=openai_key,
            base_url="https://openai.vocareum.com/v1"
    )
    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    # only include role and content from conversation history
    * [{"role": turn["role"], "content": turn["content"]} for turn in conversation_history],
    {
        "role": "user",
        "content": f"{user_message}\n\nRetrieved Context:\n{context}"
    }
]

    # TODO: Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    # TODO: Return response
    return response.choices[0].message.content
