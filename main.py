"""
Main RAG Pipeline Interface
Provides functionality to ingest documents and query the RAG system
"""

from pathlib import Path
from typing import List, Optional

from rag.ingest import DocumentIngestionPipeline
from rag.query import RAGQueryPipeline


# ============================================================================
# CONFIGURATION - Set your file paths and queries here
# ============================================================================

# File paths to ingest (PDF, DOCX, PPTX)
FILE_PATHS = [
    "1748315513376.pdf",
]

# Question to ask (leave None for interactive mode)
QUESTION = None  # Example: "What is machine learning?"

# ============================================================================


def ingest_files(file_paths: List[str]):
    """
    Ingest one or more documents into the vector store
    
    Args:
        file_paths: List of file paths to ingest
    """
    try:
        # Validate file paths
        invalid_files = []
        for file_path in file_paths:
            if not Path(file_path).exists():
                invalid_files.append(file_path)
        
        if invalid_files:
            print(f"❌ Error: The following files do not exist:")
            for f in invalid_files:
                print(f"  - {f}")
            return False
        
        pipeline = DocumentIngestionPipeline()
        
        if len(file_paths) > 1:
            pipeline.process_multiple_documents(file_paths)
        else:
            pipeline.process_document(file_paths[0])
        
        print("✅ Documents ingested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def query_rag(question: Optional[str] = None):
    """
    Query the RAG system
    
    Args:
        question: Question to ask (optional, will enter interactive mode if not provided)
    """
    try:
        rag_pipeline = RAGQueryPipeline()
        
        if question:
            rag_pipeline.query(question)
        else:
            # Interactive mode
            print("\n" + "="*60)
            print("RAG QUERY PIPELINE - INTERACTIVE MODE")
            print("="*60)
            print("Type your question to query the documents")
            print("Type 'quit' or 'exit' to exit")
            print("="*60 + "\n")
            
            while True:
                user_input = input("\nYour question (or 'quit'): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input:
                    rag_pipeline.query(user_input)
            
    except FileNotFoundError:
        print("\n❌ Error: Vector store not found!")
        print("Please run ingestion first to create the vector store.")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    print("="*60)
    print("RAG PIPELINE")
    print("="*60)
    print("What would you like to do?")
    print("1. Ingest documents")
    print("2. Retrieve/Query documents")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Ingest files
            if not FILE_PATHS:
                print("❌ Error: No file paths configured in FILE_PATHS")
                print("Please add file paths to FILE_PATHS in main.py")
                break
            
            print("\n" + "="*60)
            print("INGESTING DOCUMENTS")
            print("="*60)
            ingest_files(FILE_PATHS)
            print("\n✅ Ingestion complete!")
            break
            
        elif choice == "2":
            # Query documents
            print("\n" + "="*60)
            print("QUERYING RAG SYSTEM")
            print("="*60)
            
            if QUESTION:
                query_rag(QUESTION)
            else:
                query_rag()  # Interactive mode
            break
            
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
