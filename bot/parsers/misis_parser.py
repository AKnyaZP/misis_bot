import re
import asyncio
from typing import List, Set, Dict, Optional
from urllib.parse import urljoin, urlparse
from loguru import logger

import aiohttp
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from bot.config import settings


class MisisParser:
    """Парсер сайта misis.ru для извлечения контента и создания чанков."""
    
    def __init__(self, base_url: str = "https://misis.ru", max_pages: int = 100):
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _is_valid_url(self, url: str) -> bool:
        """Проверяет, является ли URL валидным для парсинга."""
        parsed = urlparse(url)
        # Игнорируем внешние ссылки, файлы, якоря
        if parsed.netloc and parsed.netloc not in ["misis.ru", "www.misis.ru"]:
            return False
        if any(url.endswith(ext) for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar"]):
            return False
        if url.startswith("mailto:") or url.startswith("tel:") or url.startswith("#"):
            return False
        return True

    def _normalize_url(self, url: str) -> str:
        """Нормализует URL."""
        if url.startswith("/"):
            url = urljoin(self.base_url, url)
        # Убираем якоря и параметры
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Извлекает текстовый контент из HTML."""
        # Удаляем скрипты и стили
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Извлекаем текст из основных элементов
        text_parts = []
        
        # Заголовки
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text = heading.get_text(strip=True)
            if text:
                text_parts.append(text)

        # Параграфы
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 10:  # Игнорируем короткие фрагменты
                text_parts.append(text)

        # Списки
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if text and len(text) > 10:
                text_parts.append(text)

        # Другие текстовые элементы
        for div in soup.find_all("div", class_=re.compile(r"content|text|article|news")):
            text = div.get_text(strip=True)
            if text and len(text) > 20:
                text_parts.append(text)

        # Объединяем и очищаем
        full_text = "\n".join(text_parts)
        # Удаляем множественные пробелы и переносы
        full_text = re.sub(r"\s+", " ", full_text)
        full_text = re.sub(r"\n\s*\n", "\n", full_text)
        
        return full_text.strip()

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> Set[str]:
        """Извлекает ссылки со страницы."""
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            normalized = self._normalize_url(href)
            if self._is_valid_url(normalized) and normalized not in self.visited_urls:
                links.add(normalized)
        return links

    async def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Загружает и парсит страницу."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: status {response.status}")
                    return None
                
                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")
                return soup
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    async def parse_page(self, url: str) -> Optional[Dict[str, any]]:
        """Парсит одну страницу и возвращает данные."""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        logger.info(f"Parsing: {url}")

        soup = await self._fetch_page(url)
        if not soup:
            return None

        text = self._extract_text(soup)
        if not text or len(text) < 50:  # Игнорируем слишком короткие страницы
            return None

        links = self._extract_links(soup, url)
        
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        return {
            "url": url,
            "title": title_text,
            "text": text,
            "links": links
        }

    def create_chunks(self, text: str, metadata: Dict[str, any]) -> List[Document]:
        """Создает чанки из текста."""
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )
        return chunks

    async def crawl(self, start_urls: List[str] = None) -> List[Document]:
        """
        Обходит сайт и создает документы для Qdrant.
        
        Args:
            start_urls: Список начальных URL для обхода. Если None, начинается с главной страницы.
        
        Returns:
            Список Document объектов для добавления в Qdrant.
        """
        if start_urls is None:
            start_urls = [self.base_url]

        all_documents = []
        urls_to_visit = set(start_urls)
        
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            # Берем URL из очереди
            current_url = urls_to_visit.pop()
            
            # Парсим страницу
            page_data = await self.parse_page(current_url)
            
            if page_data:
                # Создаем чанки
                metadata = {
                    "url": page_data["url"],
                    "title": page_data["title"],
                    "source": "misis.ru"
                }
                chunks = self.create_chunks(page_data["text"], metadata)
                all_documents.extend(chunks)
                
                # Добавляем новые ссылки в очередь
                new_links = page_data["links"] - self.visited_urls
                urls_to_visit.update(new_links)
                
                logger.info(f"Created {len(chunks)} chunks from {page_data['url']}")
            
            # Небольшая задержка между запросами
            await asyncio.sleep(0.5)

        logger.info(f"Total pages parsed: {len(self.visited_urls)}")
        logger.info(f"Total chunks created: {len(all_documents)}")
        
        return all_documents

    async def parse_specific_pages(self, urls: List[str]) -> List[Document]:
        """Парсит конкретные страницы без обхода ссылок."""
        all_documents = []
        
        for url in urls:
            page_data = await self.parse_page(url)
            if page_data:
                metadata = {
                    "url": page_data["url"],
                    "title": page_data["title"],
                    "source": "misis.ru"
                }
                chunks = self.create_chunks(page_data["text"], metadata)
                all_documents.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {url}")
        
        return all_documents
