import re
import asyncio
from typing import Callable, Optional, List
from loguru import logger
import hashlib

from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
        self._qa_chain: Optional[RetrievalQA] = None

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

    def _get_qa_chain(self) -> RetrievalQA:
        """Get or create RetrievalQA chain."""
        if self._qa_chain is None:
            llm = self._get_llm()
            vectorstore = self._get_vectorstore()

            prompt_template = """Ты ассистент для студентов и абитуриентов Университета МИСИС. 
Используй следующий контекст из официального сайта университета для ответа на вопрос.
Если в контексте нет полного ответа, можешь дополнить своими знаниями, но приоритет отдавай информации из контекста.

Контекст из сайта МИСИС:
{context}

Вопрос пользователя: {question}

Дай развернутый и полезный ответ на основе контекста:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
            )
        return self._qa_chain

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
        """Generate response using RAG with web search and Qdrant cache."""
        try:
            # Сначала проверяем кэш в Qdrant
            logger.info(f"Checking cache for query: {prompt}")
            cached_docs = await self.semantic_search(prompt, k=3)
            
            # Если в кэше есть релевантные результаты, используем их
            if cached_docs and len(cached_docs) > 0:
                logger.info(f"Found {len(cached_docs)} cached documents")
                context = "\n\n".join([doc.page_content for doc in cached_docs])
            else:
                # Если кэш пуст, делаем веб-поиск
                logger.info("Cache empty, performing web search")
                web_docs = await self._web_search_misis(prompt)
                
                if web_docs:
                    # Сохраняем результаты в кэш
                    logger.info(f"Saving {len(web_docs)} documents to cache")
                    await asyncio.to_thread(self.add_langchain_documents, web_docs)
                    context = "\n\n".join([doc.page_content for doc in web_docs])
                else:
                    logger.warning("No results from web search, using direct generation")
                    return await self._generate_direct(prompt)
            
            # Генерируем ответ на основе контекста
            return await self._generate_with_context(prompt, context)
            
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

    async def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Выполняет семантический поиск по Qdrant (кэш)."""
        try:
            vectorstore = self._get_vectorstore()
            
            def _search() -> List[Document]:
                results = vectorstore.similarity_search(query, k=k)
                return results
            
            return await asyncio.to_thread(_search)
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

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
