"""
Main interface for Cognito-Droid pipelines
Provides options for document ingestion, querying, and mindmap graph generation
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.graphs import Neo4jGraph
import ollama

from rag.ingest import DocumentIngestionPipeline
from rag.query import RAGQueryPipeline
from mindmap.mindmap import (
    create_mindmap_workflow,
    print_mindmap_json,
    save_mindmap_to_file,
)
from mindmap.config import (
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    get_neo4j_connection_params,
)
from mindmap.prompt import GRAPH_QUERY_TEMPLATE, GRAPH_QUERY_SYSTEM_PROMPT


# ============================================================================
# CONFIGURATION - Set your file paths and queries here
# ============================================================================

# File paths to ingest into RAG (PDF, DOCX, PPTX)
FILE_PATHS = ["/Users/shivanshmahajan/Developer/ARM_AI/Chapter 9 2025.pptx"]

# Question to ask after ingestion (leave None for interactive mode)
QUESTION = None  # Example: "What is machine learning?"

# File paths to use when building the knowledge graph / mindmap
KNOWLEDGE_GRAPH_FILE_PATHS = ["/Users/shivanshmahajan/Developer/ARM_AI/Chapter 9 2025.pptx"]

# Optional custom query for detailed response when building the knowledge graph
KNOWLEDGE_GRAPH_QUERY = None

# Save generated mindmap JSON to disk when building the knowledge graph
SAVE_MINDMAP_JSON = False
MINDMAP_OUTPUT_FILE = "mindmap_output.json"

# Default query for summarising the existing Neo4j knowledge graph
GRAPH_OVERVIEW_QUERY = "Provide a concise, mindmap-style overview of all concepts and relationships in the current knowledge graph."

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


def extract_text_from_files(file_paths: List[str]) -> Optional[str]:
    """Load documents using the ingestion pipeline and return combined text."""
    try:
        pipeline = DocumentIngestionPipeline()
    except Exception as exc:
        print(f"❌ Failed to initialise ingestion pipeline: {exc}")
        return None

    combined_text = []

    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"❌ Skipping missing file: {file_path}")
            continue

        try:
            documents = pipeline.load_document(file_path)
            text = "\n\n".join(doc.page_content for doc in documents)
            if text.strip():
                combined_text.append(text)
            else:
                print(f"⚠️ No textual content extracted from {file_path}")
        except Exception as exc:
            print(f"❌ Error loading {file_path}: {exc}")

    if not combined_text:
        print("❌ Unable to extract text from the provided files.")
        return None

    return "\n\n".join(combined_text)


def build_knowledge_graph_from_files(
    file_paths: List[str],
    query: Optional[str] = None,
    save: bool = False,
    output_file: str = MINDMAP_OUTPUT_FILE,
) -> bool:
    """Generate a mindmap-based knowledge graph from the provided files."""
    text = extract_text_from_files(file_paths)
    if not text:
        return False

    initial_state = {
        "input_text": text,
        "mindmap_json": None,
        "graph_built": False,
        "query": query,
        "graph_context": None,
        "detailed_response": None,
    }

    try:
        workflow = create_mindmap_workflow()
        final_state = workflow.invoke(initial_state)
    except Exception as exc:
        print(f"❌ Failed to build knowledge graph: {exc}")
        import traceback
        traceback.print_exc()
        return False

    mindmap = final_state.get("mindmap_json")
    detailed_response = final_state.get("detailed_response")

    if mindmap:
        print("\n" + "="*60)
        print("GENERATED MINDMAP STRUCTURE")
        print("="*60)
        print_mindmap_json(mindmap)

        if save:
            try:
                save_mindmap_to_file(mindmap, output_file)
            except Exception as exc:
                print(f"⚠️ Unable to save mindmap to {output_file}: {exc}")

    if detailed_response:
        print("\n" + "="*60)
        print("DETAILED GRAPH SUMMARY")
        print("="*60)
        print(detailed_response)

    print("\n✅ Knowledge graph pipeline completed.")
    return True


def summarize_existing_graph(query: Optional[str] = None) -> bool:
    """Summarise the current Neo4j knowledge graph using the mindmap prompt."""
    query = query or GRAPH_OVERVIEW_QUERY

    try:
        graph = Neo4jGraph(**get_neo4j_connection_params())
    except Exception as exc:
        print(f"❌ Unable to connect to Neo4j: {exc}")
        return False

    graph_query = """
    MATCH (n:Concept)
    OPTIONAL MATCH (n)-[r]->(m:Concept)
    RETURN n.id as id, n.label as label, n.description as description,
           collect({target: m.label, relation: type(r), relation_type: r.relation_type}) as connections
    ORDER BY n.label
    """

    try:
        results = graph.query(graph_query)
    except Exception as exc:
        print(f"❌ Failed to retrieve graph context: {exc}")
        return False

    if not results:
        print("⚠️ Knowledge graph is empty. Please build it first.")
        return False

    graph_context_parts = []
    for record in results:
        context_str = f"**{record['label']}** (ID: {record['id']})\n"
        context_str += f"  Description: {record['description']}\n"
        connections = [c for c in record['connections'] if c['target'] is not None]
        if connections:
            context_str += "  Connections:\n"
            for conn in connections:
                rel_type = conn.get('relation_type', conn['relation'])
                context_str += f"    - {conn['relation']} → {conn['target']} ({rel_type})\n"
        graph_context_parts.append(context_str)

    graph_context = "\n".join(graph_context_parts)

    user_prompt = GRAPH_QUERY_TEMPLATE.format_messages(
        graph_context=graph_context,
        query=query,
    )[-1].content

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": GRAPH_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": OLLAMA_TEMPERATURE},
        )
    except Exception as exc:
        print(f"❌ Failed to generate graph summary: {exc}")
        return False

    output_text = ""
    if response and getattr(response, "message", None):
        output_text = response.message.content or ""

    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH OVERVIEW")
    print("="*60)
    print(output_text)

    return True


def print_menu():
    """Display the main menu."""
    print("="*60)
    print("COGNITO-DROID PIPELINE MENU")
    print("="*60)
    print("1. Ingest documents into RAG")
    print("2. Query RAG pipeline")
    print("3. Build knowledge graph (mindmap) from configured files")
    print("4. Generate overview of existing knowledge graph")
    print("Q. Quit")
    print("="*60)


def main():
    """Main entry point for command-line interaction."""
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip().lower()

        if choice == "1":
            if not FILE_PATHS:
                print("❌ Error: No file paths configured in FILE_PATHS")
            else:
                print("\n" + "="*60)
                print("INGESTING DOCUMENTS")
                print("="*60)
                ingest_files(FILE_PATHS)

        elif choice == "2":
            print("\n" + "="*60)
            print("QUERYING RAG SYSTEM")
            print("="*60)
            query_rag(QUESTION)

        elif choice == "3":
            if not KNOWLEDGE_GRAPH_FILE_PATHS:
                print("❌ Error: No file paths configured for knowledge graph generation")
            else:
                build_knowledge_graph_from_files(
                    KNOWLEDGE_GRAPH_FILE_PATHS,
                    query=KNOWLEDGE_GRAPH_QUERY,
                    save=SAVE_MINDMAP_JSON,
                    output_file=MINDMAP_OUTPUT_FILE,
                )

        elif choice == "4":
            summarize_existing_graph()

        elif choice in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please select 1, 2, 3, 4, or Q to quit.")

        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
