import re
import asyncio
from typing import Callable, Optional, List
from loguru import logger
import hashlib

from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from duckduckgo_search import DDGS
import aiohttp
from bs4 import BeautifulSoup

from bot.config import settings


class HFService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._client: Optional[InferenceClient] = None
        self._llm: Optional[HuggingFaceEndpoint] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[Qdrant] = None

    def _get_client(self) -> InferenceClient:
        """Get or create InferenceClient for GPT-oSS-20b."""
        if self._client is None:
            # Используем модель из настроек (GPT-oSS-20b или другую)
            model_id = self.settings.hf_model
            self._client = InferenceClient(
                model=model_id,
                token=self.settings.hf_token,
            )
        return self._client

    def _get_llm(self) -> HuggingFaceEndpoint:
        """Get or create LangChain HuggingFaceEndpoint."""
        if self._llm is None:
            model_id = self.settings.hf_model
            self._llm = HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/models/{model_id}",
                huggingfacehub_api_token=self.settings.hf_token,
                task="text-generation",
                model_kwargs={
                    "max_new_tokens": self.settings.max_new_tokens,
                    "temperature": self.settings.temperature,
                }
            )
        return self._llm

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embeddings model."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cuda"},
            )
        return self._embeddings

    def _get_qdrant_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key if self.settings.qdrant_api_key else None,
        )

    def _get_vectorstore(self) -> Qdrant:
        """Get or create Qdrant vectorstore."""
        if self._vectorstore is None:
            qdrant_client = self._get_qdrant_client()
            embeddings = self._get_embeddings()

            # Проверяем существование коллекции
            try:
                collections = qdrant_client.get_collections()
                collection_exists = any(
                    col.name == self.settings.qdrant_collection_name
                    for col in collections.collections
                )
                if not collection_exists:
                    # Создаем коллекцию если её нет
                    qdrant_client.create_collection(
                        collection_name=self.settings.qdrant_collection_name,
                        vectors_config=VectorParams(
                            size=384,  # Размер для paraphrase-multilingual-MiniLM-L12-v2
                            distance=Distance.COSINE,
                        ),
                    )
            except Exception as e:
                logger.warning(f"Error checking collections: {e}")

            self._vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=self.settings.qdrant_collection_name,
                embeddings=embeddings,
            )
        return self._vectorstore

    async def generate(self, prompt: str, use_rag: bool = True) -> str:
        """
        Generate response using HF model.
        
        Args:
            prompt: User prompt
            use_rag: If True, use RAG with Qdrant vectorstore (default: True)
        """
        if use_rag:
            return await self._generate_with_rag(prompt)
        else:
            return await self._generate_direct(prompt)

    async def _generate_direct(self, prompt: str) -> str:
        """Generate response directly using InferenceClient (as in example)."""
        client = self._get_client()

        def _infer() -> str:
            try:
                system_instructions = (
                    "Ты русскоязычный ассистент для студентов и абитуриентов Университета МИСИС. "
                    "Помогай отвечать на вопросы об университете, образовательных программах, "
                    "поступлении, студенческой жизни и других аспектах университета. "
                    "Отвечай кратко, информативно и по делу. "
                    "Если не знаешь ответа, честно скажи об этом."
                )
                messages = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt},
                ]

                logger.debug(f"Calling chat_completion with messages={messages}")

                result = client.chat_completion(
                    messages=messages,
                    max_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )

                logger.debug(f"Raw result: {result}")

                # Extract the assistant's reply
                if hasattr(result, "choices") and result.choices:
                    content = result.choices[0].message.content
                    logger.info(f"Extracted content: '{content}'")
                    return content if content else ""

                logger.warning(f"Result has no choices: {result}")
                return str(result)
            except StopIteration:
                return ""
            except Exception as e:
                logger.error(f"Error in _infer: {str(e)}")
                raise

        return await self._run(_infer)

    async def _generate_with_rag(self, prompt: str) -> str:
        """Generate response using web search and Qdrant cache for Q&A pairs."""
        try:
            # Сначала проверяем кэш вопрос-ответ в Qdrant
            logger.info(f"Checking Q&A cache for query: {prompt}")
            cached_answer = await self._get_cached_answer(prompt)
            
            if cached_answer:
                logger.info("Found cached answer")
                return cached_answer
            
            # Если кэш пуст, делаем веб-поиск по misis.ru
            logger.info("Cache empty, performing web search on misis.ru")
            web_docs = await self._web_search_misis(prompt)
            
            if not web_docs:
                logger.warning("No results from web search, using direct generation")
                answer = await self._generate_direct(prompt)
            else:
                # Формируем контекст из результатов поиска
                context = "\n\n".join([doc.page_content for doc in web_docs])
                # Генерируем ответ на основе контекста
                answer = await self._generate_with_context(prompt, context)
            
            # Сохраняем пару вопрос-ответ в кэш
            logger.info("Saving Q&A pair to cache")
            await self._cache_qa_pair(prompt, answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}")
            # Fallback to direct generation
            return await self._generate_direct(prompt)

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to Qdrant vectorstore."""
        try:
            vectorstore = self._get_vectorstore()
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            vectorstore.add_documents(documents)
            logger.info(f"Added {len(texts)} documents to vectorstore")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def add_langchain_documents(self, documents: List[Document]):
        """Add LangChain Document objects to Qdrant vectorstore."""
        try:
            vectorstore = self._get_vectorstore()
            vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vectorstore")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def _get_cached_answer(self, question: str) -> Optional[str]:
        """Проверяет кэш вопрос-ответ в Qdrant."""
        try:
            vectorstore = self._get_vectorstore()
            question_hash = hashlib.md5(question.encode()).hexdigest()
            
            def _search() -> Optional[str]:
                # Ищем похожие вопросы через семантический поиск
                results = vectorstore.similarity_search(question, k=3)
                for doc in results:
                    metadata = doc.metadata
                    # Проверяем, что это кэшированная пара вопрос-ответ
                    if metadata.get('type') == 'qa_pair':
                        # Проверяем точное совпадение хеша
                        if metadata.get('question_hash') == question_hash:
                            # Извлекаем ответ из page_content (формат: "Вопрос: ...\nОтвет: ...")
                            content = doc.page_content
                            if "Ответ:" in content:
                                answer = content.split("Ответ:")[-1].strip()
                                return answer
                            return content
                        # Если вопрос очень похож, тоже возвращаем
                        cached_question = metadata.get('question', '')
                        if cached_question and self._questions_similar(question, cached_question):
                            content = doc.page_content
                            if "Ответ:" in content:
                                answer = content.split("Ответ:")[-1].strip()
                                return answer
                            return content
                return None
            
            return await asyncio.to_thread(_search)
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            return None

    def _questions_similar(self, q1: str, q2: str, threshold: float = 0.8) -> bool:
        """Проверяет, похожи ли два вопроса (простая проверка по словам)."""
        # Простая проверка: если больше 80% слов совпадают
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return False
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold

    async def _cache_qa_pair(self, question: str, answer: str) -> None:
        """Сохраняет пару вопрос-ответ в кэш Qdrant."""
        try:
            vectorstore = self._get_vectorstore()
            question_hash = hashlib.md5(question.encode()).hexdigest()
            
            # Создаем документ: используем вопрос для эмбеддинга, но сохраняем ответ в content
            # Это позволяет искать по вопросу, но получать ответ
            doc = Document(
                page_content=answer,  # Сохраняем ответ
                metadata={
                    'question': question,
                    'question_hash': question_hash,
                    'type': 'qa_pair',
                    'source': 'web_search_cache'
                }
            )
            
            # Для эмбеддинга используем вопрос, но сохраняем ответ
            # Создаем временный документ с вопросом для эмбеддинга
            question_doc = Document(
                page_content=question,
                metadata=doc.metadata
            )
            
            def _add():
                # Добавляем документ, эмбеддинг будет создан из page_content (вопроса)
                # Но мы хотим использовать вопрос для поиска, а ответ для хранения
                # Поэтому создаем документ с комбинацией вопрос + ответ для эмбеддинга
                combined_content = f"Вопрос: {question}\nОтвет: {answer}"
                combined_doc = Document(
                    page_content=combined_content,
                    metadata=doc.metadata
                )
                vectorstore.add_documents([combined_doc])
            
            await asyncio.to_thread(_add)
            logger.info(f"Cached Q&A pair for question hash: {question_hash[:8]}...")
        except Exception as e:
            logger.error(f"Error caching Q&A pair: {str(e)}")

    async def _web_search_misis(self, query: str, max_results: int = 5) -> List[Document]:
        """Выполняет веб-поиск по домену misis.ru и извлекает контент."""
        try:
            # Формируем запрос с ограничением по домену
            search_query = f"site:misis.ru {query}"
            logger.info(f"Searching web with query: {search_query}")
            
            def _search() -> List[dict]:
                """Синхронный поиск через DuckDuckGo."""
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(
                            search_query,
                            max_results=max_results,
                            region='ru-ru'
                        ))
                    return results
                except Exception as e:
                    logger.error(f"Error in DuckDuckGo search: {str(e)}")
                    return []
            
            search_results = await asyncio.to_thread(_search)
            
            if not search_results:
                logger.warning("No search results found")
                return []
            
            # Извлекаем контент со страниц
            documents = []
            async with aiohttp.ClientSession() as session:
                for result in search_results:
                    url = result.get('href', '')
                    title = result.get('title', '')
                    snippet = result.get('body', '')
                    
                    if not url:
                        continue
                    
                    try:
                        # Пытаемся получить полный контент со страницы
                        content = await self._fetch_page_content(session, url)
                        if content:
                            # Используем полный контент, если удалось получить
                            text = content
                        else:
                            # Иначе используем snippet из результатов поиска
                            text = f"{title}\n{snippet}"
                        
                        doc = Document(
                            page_content=text,
                            metadata={
                                'url': url,
                                'title': title,
                                'source': 'web_search',
                                'query_hash': hashlib.md5(query.encode()).hexdigest()
                            }
                        )
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error fetching content from {url}: {str(e)}")
                        # Используем snippet как fallback
                        doc = Document(
                            page_content=f"{title}\n{snippet}",
                            metadata={
                                'url': url,
                                'title': title,
                                'source': 'web_search_snippet',
                                'query_hash': hashlib.md5(query.encode()).hexdigest()
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} documents from web search")
            return documents
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []

    async def _fetch_page_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Извлекает текстовый контент со страницы."""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Удаляем скрипты и стили
                    for script in soup(["script", "style", "nav", "header", "footer"]):
                        script.decompose()
                    
                    # Извлекаем текст
                    text = soup.get_text()
                    # Очищаем от лишних пробелов
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Ограничиваем размер (первые 2000 символов)
                    if len(text) > 2000:
                        text = text[:2000] + "..."
                    
                    return text
        except Exception as e:
            logger.debug(f"Could not fetch content from {url}: {str(e)}")
            return None

    async def _generate_with_context(self, prompt: str, context: str) -> str:
        """Генерирует ответ на основе контекста."""
        client = self._get_client()
        
        def _infer() -> str:
            try:
                system_instructions = (
                    "Ты ассистент для студентов и абитуриентов Университета МИСИС. "
                    "Используй следующий контекст из официального сайта университета для ответа на вопрос. "
                    "Если в контексте нет полного ответа, можешь дополнить своими знаниями, но приоритет отдавай информации из контекста."
                )
                messages = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"Контекст из сайта МИСИС:\n{context}\n\nВопрос пользователя: {prompt}\n\nДай развернутый и полезный ответ на основе контекста:"},
                ]
                
                logger.debug(f"Calling chat_completion with context")
                
                result = client.chat_completion(
                    messages=messages,
                    max_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )
                
                logger.debug(f"Raw result: {result}")
                
                # Extract the assistant's reply
                if hasattr(result, "choices") and result.choices:
                    content = result.choices[0].message.content
                    logger.info(f"Extracted content: '{content[:100]}...'")
                    return self.clean_output(content) if content else ""
                
                logger.warning(f"Result has no choices: {result}")
                return str(result)
            except StopIteration:
                return ""
            except Exception as e:
                logger.error(f"Error in _infer: {str(e)}")
                raise
        
        return await self._run(_infer)

    @staticmethod
    def clean_output(text: str) -> str:
        """Clean output text."""
        cleaned = text.strip()
        cleaned = re.sub(r"^Ассистент:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    async def _run(callable_fn: Callable[[], str]) -> str:
        """Run inference in thread pool."""
        try:
            return await asyncio.to_thread(callable_fn)
        except Exception as err:
            logger.exception(f"Generation error: {err}")
            raise RuntimeError("Ошибка генерации ответа от модели") from err
