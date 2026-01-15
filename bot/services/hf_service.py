import re
import asyncio
from typing import Callable, Optional, List
from loguru import logger

from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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
                model_kwargs={"device": "cpu"},
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
        """Generate response using RAG with Qdrant."""
        try:
            qa_chain = self._get_qa_chain()

            def _infer() -> str:
                result = qa_chain.invoke({"query": prompt})
                answer = result.get("result", "")
                return self.clean_output(answer)

            return await self._run(_infer)
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
        """Выполняет семантический поиск по Qdrant."""
        try:
            vectorstore = self._get_vectorstore()
            
            def _search() -> List[Document]:
                results = vectorstore.similarity_search(query, k=k)
                return results
            
            return await asyncio.to_thread(_search)
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

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
