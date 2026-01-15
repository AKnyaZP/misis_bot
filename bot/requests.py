from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)

async def get_hf_response(prompt: str, hf_client: InferenceClient) -> str:
    # композитинг запроса на любой апи
    result = await hf_client.text_generation(prompt)
    return result