import re
import asyncio
from typing import Callable, Optional, List
from loguru import logger
import hashlib

from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
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
        """Создает или возвращает InferenceClient для модели HuggingFace."""
        if self._client is None:
            model_id = self.settings.hf_model
            self._client = InferenceClient(
                model=model_id,
                token=self.settings.hf_token,
            )
        return self._client

    def _get_llm(self) -> HuggingFaceEndpoint:
        """Создает или возвращает LangChain HuggingFaceEndpoint."""
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
        """Создает или возвращает модель эмбеддингов."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cuda"},
            )
        return self._embeddings

    def _get_qdrant_client(self) -> QdrantClient:
        """Создает клиент Qdrant."""
        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key if self.settings.qdrant_api_key else None,
        )

    def _get_vectorstore(self) -> Optional[Qdrant]:
        """Создает или возвращает векторное хранилище Qdrant."""
        if self._vectorstore is None:
            try:
                qdrant_client = self._get_qdrant_client()
                qdrant_client.get_collections()
            except Exception as e:
                logger.warning(f"Qdrant недоступен по адресу {self.settings.qdrant_url}: {e}")
                logger.warning("RAG функциональность будет отключена.")
                return None
            
            try:
                embeddings = self._get_embeddings()
            except Exception as e:
                logger.warning(f"Ошибка загрузки модели эмбеддингов: {e}")
                return None

            try:
                collections = qdrant_client.get_collections()
                collection_exists = any(
                    col.name == self.settings.qdrant_collection_name
                    for col in collections.collections
                )
                if not collection_exists:
                    qdrant_client.create_collection(
                        collection_name=self.settings.qdrant_collection_name,
                        vectors_config=VectorParams(
                            size=384,
                            distance=Distance.COSINE,
                        ),
                    )
            except Exception as e:
                logger.warning(f"Ошибка проверки коллекций: {e}")

            try:
                self._vectorstore = Qdrant(
                    client=qdrant_client,
                    collection_name=self.settings.qdrant_collection_name,
                    embeddings=embeddings,
                )
            except Exception as e:
                logger.warning(f"Ошибка создания vectorstore: {e}")
                return None
        return self._vectorstore

    async def generate(self, prompt: str, use_rag: bool = True) -> str:
        """Генерирует ответ используя модель HuggingFace."""
        if use_rag:
            return await self._generate_with_rag(prompt)
        else:
            return await self._generate_direct(prompt)

    async def _generate_direct(self, prompt: str) -> str:
        """Генерирует ответ напрямую через InferenceClient."""
        client = self._get_client()

        def _infer() -> str:
            try:
                system_prompt = (
                    "Ты ассистент для студентов и абитуриентов Университета МИСИС. "
                    "Отвечай на вопросы об университете, программах, поступлении и студенческой жизни. "
                    "Будь кратким и информативным. Если не знаешь ответа, так и скажи."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                result = client.chat_completion(
                    messages=messages,
                    max_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )

                if hasattr(result, "choices") and result.choices:
                    content = result.choices[0].message.content
                    return content if content else ""

                logger.warning(f"Неожиданный формат ответа: {result}")
                return str(result)
            except StopIteration:
                return ""
            except Exception as e:
                logger.error(f"Ошибка в _infer: {str(e)}")
                raise

        return await self._run(_infer)

    async def _generate_with_rag(self, prompt: str) -> str:
        """Генерирует ответ используя веб-поиск и кэш Q&A в Qdrant."""
        try:
            vectorstore = self._get_vectorstore()
            
            if vectorstore is None:
                logger.info("Qdrant недоступен, используем только веб-поиск")
                web_docs = await self._web_search_misis(prompt)
                if not web_docs:
                    return await self._generate_direct(prompt)
                context = "\n\n".join([doc.page_content for doc in web_docs])
                return await self._generate_with_context(prompt, context)
            
            cached_answer = await self._get_cached_answer(prompt)
            if cached_answer:
                return cached_answer
            
            web_docs = await self._web_search_misis(prompt)
            if not web_docs:
                answer = await self._generate_direct(prompt)
            else:
                context = "\n\n".join([doc.page_content for doc in web_docs])
                answer = await self._generate_with_context(prompt, context)
            
            await self._cache_qa_pair(prompt, answer)
            
            return answer
            
        except RuntimeError as e:
            raise e
        except Exception as e:
            logger.error(f"Ошибка в RAG генерации: {str(e)}")
            error_str = str(e)
            if "403" in error_str or "Forbidden" in error_str or "permissions" in error_str.lower():
                raise RuntimeError(
                    "Токен HuggingFace не имеет прав для использования Inference API. "
                    "Создайте новый токен с правами 'read' на https://huggingface.co/settings/tokens"
                )
            return await self._generate_direct(prompt)

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Добавляет документы в векторное хранилище Qdrant."""
        try:
            vectorstore = self._get_vectorstore()
            if vectorstore is None:
                logger.warning("Qdrant недоступен, документы не добавлены")
                return
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            vectorstore.add_documents(documents)
            logger.info(f"Добавлено {len(texts)} документов в векторное хранилище")
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {str(e)}")
            raise

    def add_langchain_documents(self, documents: List[Document]):
        """Добавляет LangChain Document объекты в векторное хранилище."""
        try:
            vectorstore = self._get_vectorstore()
            if vectorstore is None:
                logger.warning("Qdrant недоступен, документы не добавлены")
                return
            vectorstore.add_documents(documents)
            logger.info(f"Добавлено {len(documents)} документов в векторное хранилище")
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {str(e)}")
            raise

    async def _get_cached_answer(self, question: str) -> Optional[str]:
        """Проверяет кэш вопрос-ответ в Qdrant."""
        try:
            vectorstore = self._get_vectorstore()
            if vectorstore is None:
                return None
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
            logger.error(f"Ошибка проверки кэша: {str(e)}")
            return None

    def _questions_similar(self, q1: str, q2: str, threshold: float = 0.8) -> bool:
        """Проверяет схожесть двух вопросов по словам."""
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
            if vectorstore is None:
                logger.debug("Qdrant недоступен, пропускаем кэширование")
                return
            question_hash = hashlib.md5(question.encode()).hexdigest()
            
            metadata = {
                'question': question,
                'question_hash': question_hash,
                'type': 'qa_pair',
                'source': 'web_search_cache'
            }
            
            def _add():
                combined_content = f"Вопрос: {question}\nОтвет: {answer}"
                combined_doc = Document(
                    page_content=combined_content,
                    metadata=metadata
                )
                vectorstore.add_documents([combined_doc])
            
            await asyncio.to_thread(_add)
            logger.info(f"Закэширована пара Q&A для хеша вопроса: {question_hash[:8]}...")
        except Exception as e:
            logger.error(f"Ошибка кэширования пары Q&A: {str(e)}")

    async def _web_search_misis(self, query: str, max_results: int = 5) -> List[Document]:
        """Выполняет веб-поиск по домену misis.ru и извлекает контент."""
        try:
            search_query = f"site:misis.ru {query}"
            logger.info(f"Поиск в интернете: {search_query}")
            
            def _search() -> List[dict]:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(
                            search_query,
                            max_results=max_results,
                            region='ru-ru'
                        ))
                    return results
                except Exception as e:
                    logger.error(f"Ошибка поиска в DuckDuckGo: {str(e)}")
                    return []
            
            search_results = await asyncio.to_thread(_search)
            
            if not search_results:
                return []
            
            documents = []
            async with aiohttp.ClientSession() as session:
                for result in search_results:
                    url = result.get('href', '')
                    title = result.get('title', '')
                    snippet = result.get('body', '')
                    
                    if not url:
                        continue
                    
                    try:
                        content = await self._fetch_page_content(session, url)
                        text = content if content else f"{title}\n{snippet}"
                        
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
                        logger.warning(f"Ошибка получения контента с {url}: {str(e)}")
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
            
            logger.info(f"Извлечено {len(documents)} документов из веб-поиска")
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка веб-поиска: {str(e)}")
            return []

    async def _fetch_page_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Извлекает текстовый контент со страницы."""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    for script in soup(["script", "style", "nav", "header", "footer"]):
                        script.decompose()
                    
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
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
                user_prompt = f"Контекст из сайта МИСИС:\n{context}\n\nВопрос: {prompt}\n\nОтветь на основе контекста:"
                messages = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_prompt},
                ]
                
                result = client.chat_completion(
                    messages=messages,
                    max_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )
                
                if hasattr(result, "choices") and result.choices:
                    content = result.choices[0].message.content
                    return self.clean_output(content) if content else ""
                
                logger.warning(f"Неожиданный формат ответа: {result}")
                return str(result)
            except StopIteration:
                return ""
            except Exception as e:
                error_str = str(e)
                logger.error(f"Ошибка в _infer (с контекстом): {error_str}")
                
                if "403" in error_str or "Forbidden" in error_str:
                    if "permissions" in error_str.lower() or "Inference Providers" in error_str:
                        logger.error("Токен HuggingFace не имеет прав для использования Inference API")
                        logger.error("Решение: создайте токен с правами 'read' на https://huggingface.co/settings/tokens")
                        raise RuntimeError(
                            "Токен HuggingFace не имеет достаточных прав для использования Inference API. "
                            "Создайте новый токен с правами 'read' на https://huggingface.co/settings/tokens"
                        )
                    else:
                        raise RuntimeError("Доступ к HuggingFace API запрещен. Проверьте токен.")
                elif "401" in error_str or "Unauthorized" in error_str:
                    raise RuntimeError("Неверный токен HuggingFace. Проверьте HF_TOKEN в bot/.env")
                else:
                    raise RuntimeError(f"Ошибка при обращении к модели: {error_str[:200]}")
        
        return await self._run(_infer)

    @staticmethod
    def clean_output(text: str) -> str:
        """Очищает вывод от префиксов."""
        cleaned = text.strip()
        cleaned = re.sub(r"^Ассистент:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    async def _run(callable_fn: Callable[[], str]) -> str:
        """Выполняет инференс в отдельном потоке."""
        try:
            return await asyncio.to_thread(callable_fn)
        except RuntimeError as err:
            raise err
        except Exception as err:
            logger.exception(f"Ошибка генерации: {err}")
            error_str = str(err)
            if "403" in error_str or "Forbidden" in error_str:
                raise RuntimeError(
                    "Токен HuggingFace не имеет прав для использования Inference API. "
                    "Создайте новый токен с правами 'read' на https://huggingface.co/settings/tokens"
                )
            raise RuntimeError(f"Ошибка генерации ответа от модели: {error_str[:200]}")
