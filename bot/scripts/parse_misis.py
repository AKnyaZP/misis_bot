"""
Скрипт для парсинга сайта misis.ru и загрузки данных в Qdrant.
"""
import asyncio
import logging
from pathlib import Path
import sys

# Добавляем корневую директорию в путь для локального запуска
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.config import settings
from bot.services.hf_service import HFService
from bot.parsers.misis_parser import MisisParser
from loguru import logger

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Основная функция для парсинга и загрузки данных."""
    logger.info("Starting MISIS.ru parser...")
    
    # Инициализируем сервис
    hf_service = HFService(settings)
    
    # Список важных страниц для парсинга
    important_pages = [
        "https://misis.ru/",
        "https://misis.ru/applicants/",
        "https://misis.ru/students/",
        "https://misis.ru/education/",
        "https://misis.ru/university/",
        "https://misis.ru/university/news/",
        "https://misis.ru/applicants/admission/",
        "https://misis.ru/applicants/admission/baccalaureate-and-specialty/",
        "https://misis.ru/applicants/admission/magistracy/",
    ]
    
    # Парсим сайт
    async with MisisParser(base_url="https://misis.ru", max_pages=50) as parser:
        logger.info("Starting to parse important pages...")
        documents = await parser.parse_specific_pages(important_pages)
        
        if not documents:
            logger.warning("No documents parsed. Trying full crawl...")
            documents = await parser.crawl(start_urls=["https://misis.ru/"])
    
    if not documents:
        logger.error("No documents were parsed. Exiting.")
        return
    
    logger.info(f"Parsed {len(documents)} document chunks")
    
    # Загружаем в Qdrant
    try:
        logger.info("Loading documents into Qdrant...")
        hf_service.add_langchain_documents(documents)
        logger.info(f"Successfully loaded {len(documents)} documents into Qdrant")
        
        # Тестируем семантический поиск
        logger.info("Testing semantic search...")
        test_query = "поступление в университет"
        results = await hf_service.semantic_search(test_query, k=3)
        logger.info(f"Semantic search test query: '{test_query}'")
        logger.info(f"Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            logger.info(f"Result {i}: {doc.metadata.get('title', 'No title')} - {doc.page_content[:100]}...")
        
    except Exception as e:
        logger.error(f"Error loading documents into Qdrant: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Parsing interrupted by user")
    except Exception as e:
        logger.exception(f"Error during parsing: {e}")
        sys.exit(1)
