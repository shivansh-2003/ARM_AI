"""
RAG Query Pipeline
Handles retrieval and generation using LangChain RAG pattern
"""

from typing import Dict
from . import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGQueryPipeline:
    """RAG Pipeline for querying documents with context-aware responses"""

    def __init__(self, vector_store_path: str = None):
        """
        Initialize RAG query pipeline

        Args:
            vector_store_path: Path to existing vector store (optional)
        """
        print("Initializing RAG Query Pipeline...")

        # Initialize embeddings
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}
        )

        # Load vector store
        vector_store_path = vector_store_path or config.VECTOR_STORE_PATH
        print(f"Loading vector store from: {vector_store_path}")
        self.vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings,
            collection_name=config.COLLECTION_NAME
        )

        # Initialize LLM
        print(f"Initializing LLM: {config.LLM_MODEL}")
        self.llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=config.LLM_TEMPERATURE
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type=config.SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVAL_K}
        )

        # Define prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer based on the context above. If the answer is not in the context, say "I don't have enough information to answer this question based on the provided context."
"""
        )
        
        # Build RAG chain using LangChain RAG pattern
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        print("RAG Query Pipeline initialized successfully!")

    def _format_docs(self, docs) -> str:
        """
        Format retrieved documents into context string

        Args:
            docs: List of retrieved Document objects

        Returns:
            Formatted context string
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> Dict:
        """
        Query the RAG system

        Args:
            question: User's question

        Returns:
            Dictionary with answer and source documents
        """
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}\n")

        # Retrieve relevant documents
        print(f"Retrieving top {config.RETRIEVAL_K} relevant chunks...")
        retrieved_docs = self.retriever.invoke(question)

        print(f"Retrieved {len(retrieved_docs)} documents")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            print(f"  {i+1}. {source} (Page {page})")

        # Generate response using RAG chain
        print("\nGenerating response...")
        answer = self.rag_chain.invoke(question)

        print(f"\n{'='*60}")
        print("RESPONSE:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")

        return {
            "answer": answer,
            "source_documents": retrieved_docs
        }


def main():
    """Example usage of RAG query pipeline"""
    try:
        rag_pipeline = RAGQueryPipeline()

        print("\n" + "="*60)
        print("RAG QUERY PIPELINE - INTERACTIVE MODE")
        print("="*60)
        print("Type your question to query the documents")
        print("Type 'quit' to exit")
        print("="*60 + "\n")

        while True:
            user_input = input("\nYour question (or 'quit'): ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input:
                rag_pipeline.query(user_input)

    except FileNotFoundError:
        print("\n❌ Error: Vector store not found!")
        print("Please run the ingestion pipeline first to create the vector store.")
        print("Example: python ingest.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
