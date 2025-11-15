"""
Document Ingestion Pipeline
Handles PDF, DOCX, and PPTX document loading, chunking, and vector store creation
"""

import os
from pathlib import Path
from typing import List
from . import config

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into vector store"""
    
    def __init__(self):
        """Initialize the ingestion pipeline"""
        print("Initializing Document Ingestion Pipeline...")
        
        # Initialize embeddings
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS
        )
        
        print("Pipeline initialized successfully!")
    
    def load_document(self, file_path: str) -> List:
        """
        Load document based on file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported: {config.SUPPORTED_EXTENSIONS}")
        
        print(f"Loading document: {file_path}")
        
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages/sections")
        
        return documents
    
    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        print(f"Chunking documents with size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List):
        """
        Create vector store from chunks
        
        Args:
            chunks: List of chunked Document objects
            
        Returns:
            Chroma vector store instance
        """
        print(f"Creating vector store with {len(chunks)} chunks...")
        
        vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=config.VECTOR_STORE_PATH,
                collection_name=config.COLLECTION_NAME
            )
        
        print(f"Vector store persisted to: {config.VECTOR_STORE_PATH}")
        return vectorstore
    
    def process_document(self, file_path: str):
        """
        Complete pipeline: load -> chunk -> embed -> store
        
        Args:
            file_path: Path to document file
            
        Returns:
            Chroma vector store instance
        """
        print("\n" + "="*60)
        print("STARTING DOCUMENT INGESTION PIPELINE")
        print("="*60)
        
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        vectorstore = self.create_vector_store(chunks)
        
        print("\n" + "="*60)
        print("INGESTION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
        return vectorstore
    
    def process_multiple_documents(self, file_paths: List[str]):
        """
        Process multiple documents into single vector store
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            Chroma vector store instance
        """
        print("\n" + "="*60)
        print(f"PROCESSING {len(file_paths)} DOCUMENTS")
        print("="*60)
        
        all_chunks = []
        
        for file_path in file_paths:
            print(f"\nProcessing: {file_path}")
            documents = self.load_document(file_path)
            chunks = self.chunk_documents(documents)
            all_chunks.extend(chunks)
        
        print(f"\nTotal chunks from all documents: {len(all_chunks)}")
        vectorstore = self.create_vector_store(all_chunks)
        
        print("\n" + "="*60)
        print("MULTI-DOCUMENT INGESTION COMPLETED")
        print("="*60 + "\n")
        
        return vectorstore


def main():
    """Example usage of the ingestion pipeline"""
    pipeline = DocumentIngestionPipeline()
    print("Ingestion pipeline ready to use!")
    print(f"Supported file types: {config.SUPPORTED_EXTENSIONS}")
    
    # Example usage:
    # pipeline.process_document("./data/sample.pdf")
    # pipeline.process_multiple_documents(["./data/doc1.pdf", "./data/doc2.docx"])


if __name__ == "__main__":
    main()
