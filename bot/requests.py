import logging
from bot.services.hf_service import HFService

logger = logging.getLogger(__name__)

async def get_hf_response(prompt: str, hf_service: HFService, use_rag: bool = True) -> str:
    """
    Get response from HF service.
    
    Args:
        prompt: User prompt
        hf_service: HFService instance
        use_rag: If True, use RAG with Qdrant (default: False)
    """
    result = await hf_service.generate(prompt, use_rag=use_rag)
    return result