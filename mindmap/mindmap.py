"""
Main entrypoint for Mindmap Generator
Implements LangGraph workflow for mindmap generation, Neo4j graph building, and querying
"""

import argparse
import json
from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel, Field

# LangChain imports
from langchain_community.graphs import Neo4jGraph

# Ollama Python client
import ollama

# LangGraph imports
from langgraph.graph import StateGraph, END

# Local imports
from mindmap.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TEMPERATURE,
    NODE_LABEL,
    RELATIONSHIP_TYPE,
    validate_config,
    get_neo4j_connection_params,
)
from mindmap.prompt import (
    MINDMAP_GENERATION_TEMPLATE,
    GRAPH_QUERY_TEMPLATE,
    MINDMAP_SYSTEM_PROMPT,
    GRAPH_QUERY_SYSTEM_PROMPT,
)


# ============================================================================
# Pydantic Models for Structured Output
# Reference: https://docs.langchain.com/oss/python/langchain/structured-output#structured-output
# ============================================================================

class Relationship(BaseModel):
    """Represents a relationship between two nodes in the mindmap"""
    source_id: str = Field(description="ID of the source node")
    target_id: str = Field(description="ID of the target node")
    relation_type: str = Field(description="Type of relationship: prerequisite, related, contrasts, example, part_of")


class MindmapNode(BaseModel):
    """
    Hierarchical node structure for mindmap
    Supports recursive children and cross-node relationships
    """
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Concise concept name (2-5 words)")
    description: str = Field(description="Brief explanation of the concept (1-2 sentences)")
    children: List["MindmapNode"] = Field(default_factory=list, description="List of child nodes")
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships to other nodes")


# Enable recursive model
MindmapNode.model_rebuild()


# ============================================================================
# LangGraph State Definition
# ============================================================================

class MindmapState(TypedDict):
    """State for the mindmap generation workflow"""
    input_text: str  # Original input text
    mindmap_json: Optional[MindmapNode]  # Structured mindmap output
    graph_built: bool  # Flag indicating if Neo4j graph was built
    query: Optional[str]  # User query for detailed response
    graph_context: Optional[str]  # Retrieved context from Neo4j
    detailed_response: Optional[str]  # Final response from graph query


# ============================================================================
# Workflow Nodes
# ============================================================================

def _extract_json_from_markdown(text: str) -> str:
    """Strip markdown code fences from an LLM response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


def generate_mindmap_node(state: MindmapState) -> MindmapState:
    """
    Node 1: Generate structured mindmap from input text using Ollama with structured output
    """
    print("\n" + "="*60)
    print("STEP 1: GENERATING MINDMAP")
    print("="*60)
    
    # Initialize Ollama LLM
    user_prompt = MINDMAP_GENERATION_TEMPLATE.format_messages(
        input_text=state["input_text"]
    )[-1].content  # Extract formatted human prompt

    print(f"Processing input text ({len(state['input_text'])} characters)...")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": MINDMAP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": OLLAMA_TEMPERATURE},
        )
    except Exception as exc:
        print("❌ Ollama chat request failed during mindmap generation.")
        raise exc

    raw_text = ""
    if response and getattr(response, "message", None):
        raw_text = response.message.content or ""
    raw_text = _extract_json_from_markdown(raw_text)

    try:
        mindmap = MindmapNode.model_validate_json(raw_text)
    except Exception as exc:
        print("❌ Failed to parse mindmap JSON from model output.")
        print(f"Model response length: {len(raw_text)} characters")
        print("Model response (truncated to 1000 chars):\n", raw_text[:1000])
        raise exc
    
    print(f"✅ Mindmap generated with root node: {mindmap.label}")
    print(f"   Total child concepts: {len(mindmap.children)}")
    
    state["mindmap_json"] = mindmap
    return state


def build_neo4j_graph_node(state: MindmapState) -> MindmapState:
    """
    Node 2: Build knowledge graph in Neo4j from mindmap JSON
    Reference: https://neo4j.com/labs/genai-ecosystem/langchain/
    """
    print("\n" + "="*60)
    print("STEP 2: BUILDING NEO4J GRAPH")
    print("="*60)
    
    # Initialize Neo4j graph
    # Reference: https://neo4j.com/labs/genai-ecosystem/langchain/
    graph = Neo4jGraph(**get_neo4j_connection_params())
    
    # Clear existing graph (optional - comment out to preserve existing data)
    print("Clearing existing graph data...")
    graph.query("MATCH (n) DETACH DELETE n")
    
    # Build graph from mindmap
    mindmap = state["mindmap_json"]
    
    print(f"Building graph from mindmap: {mindmap.label}")
    
    # Recursive function to create nodes and relationships
    def create_graph_recursive(node: MindmapNode, parent_id: Optional[str] = None):
        """Recursively create nodes and relationships in Neo4j"""
        
        # Create node
        create_node_query = f"""
        MERGE (n:{NODE_LABEL} {{id: $id}})
        SET n.label = $label, n.description = $description
        RETURN n
        """
        graph.query(
            create_node_query,
            params={
                "id": node.id,
                "label": node.label,
                "description": node.description,
            }
        )
        
        # Create parent-child relationship
        if parent_id:
            create_parent_relationship = f"""
            MATCH (parent:{NODE_LABEL} {{id: $parent_id}})
            MATCH (child:{NODE_LABEL} {{id: $child_id}})
            MERGE (parent)-[r:HAS_CHILD]->(child)
            RETURN r
            """
            graph.query(
                create_parent_relationship,
                params={"parent_id": parent_id, "child_id": node.id}
            )
        
        # Create cross-node relationships
        for rel in node.relationships:
            create_relationship = f"""
            MATCH (source:{NODE_LABEL} {{id: $source_id}})
            MATCH (target:{NODE_LABEL} {{id: $target_id}})
            MERGE (source)-[r:{RELATIONSHIP_TYPE} {{relation_type: $relation_type}}]->(target)
            RETURN r
            """
            graph.query(
                create_relationship,
                params={
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relation_type": rel.relation_type,
                }
            )
        
        # Recursively create children
        for child in node.children:
            create_graph_recursive(child, parent_id=node.id)
    
    # Start recursive graph creation
    create_graph_recursive(mindmap)
    
    # Get statistics
    stats_query = f"""
    MATCH (n:{NODE_LABEL})
    OPTIONAL MATCH (n)-[r]->()
    RETURN count(DISTINCT n) as node_count, count(r) as relationship_count
    """
    stats = graph.query(stats_query)
    
    print(f"✅ Graph built successfully")
    print(f"   Nodes created: {stats[0]['node_count']}")
    print(f"   Relationships created: {stats[0]['relationship_count']}")
    
    state["graph_built"] = True
    return state


def query_graph_node(state: MindmapState) -> MindmapState:
    """
    Node 3: Query Neo4j graph and generate detailed response
    """
    print("\n" + "="*60)
    print("STEP 3: QUERYING KNOWLEDGE GRAPH")
    print("="*60)
    
    # Initialize Neo4j graph
    graph = Neo4jGraph(**get_neo4j_connection_params())
    
    # Retrieve entire graph context
    graph_query = f"""
    MATCH (n:{NODE_LABEL})
    OPTIONAL MATCH (n)-[r]->(m:{NODE_LABEL})
    RETURN n.id as id, n.label as label, n.description as description,
           collect({{target: m.label, relation: type(r), relation_type: r.relation_type}}) as connections
    ORDER BY n.label
    """
    
    print("Retrieving graph context...")
    results = graph.query(graph_query)
    
    # Format graph context for LLM
    graph_context_parts = []
    for record in results:
        context_str = f"**{record['label']}** (ID: {record['id']})\n"
        context_str += f"  Description: {record['description']}\n"
        
        # Add connections
        connections = [c for c in record['connections'] if c['target'] is not None]
        if connections:
            context_str += "  Connections:\n"
            for conn in connections:
                rel_type = conn.get('relation_type', conn['relation'])
                context_str += f"    - {conn['relation']} → {conn['target']} ({rel_type})\n"
        
        graph_context_parts.append(context_str)
    
    graph_context = "\n".join(graph_context_parts)
    
    print(f"✅ Retrieved context from {len(results)} nodes")
    
    state["graph_context"] = graph_context
    
    # Generate detailed response using LLM
    print("\nGenerating detailed response...")
    
    query = state.get("query", "Provide a comprehensive overview of all concepts and their relationships")

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
            options={"temperature": 0.3},
        )
    except Exception as exc:
        print("❌ Ollama chat request failed during graph query.")
        raise exc

    detailed_text = ""
    if response and getattr(response, "message", None):
        detailed_text = response.message.content or ""

    state["detailed_response"] = detailed_text
    
    print("✅ Detailed response generated")
    
    return state


# ============================================================================
# LangGraph Workflow
# ============================================================================

def create_mindmap_workflow() -> StateGraph:
    """
    Create LangGraph workflow for mindmap generation
    
    Flow: Input → Generate Mindmap → Build Graph → Query Graph → Output
    """
    workflow = StateGraph(MindmapState)
    
    # Add nodes
    workflow.add_node("generate_mindmap", generate_mindmap_node)
    workflow.add_node("build_graph", build_neo4j_graph_node)
    workflow.add_node("query_graph", query_graph_node)
    
    # Define edges
    workflow.set_entry_point("generate_mindmap")
    workflow.add_edge("generate_mindmap", "build_graph")
    workflow.add_edge("build_graph", "query_graph")
    workflow.add_edge("query_graph", END)
    
    return workflow.compile()


# ============================================================================
# Helper Functions
# ============================================================================

def print_mindmap_json(mindmap: MindmapNode, indent: int = 0):
    """
    Pretty print mindmap structure
    """
    prefix = "  " * indent
    print(f"{prefix}└─ {mindmap.label} (ID: {mindmap.id})")
    print(f"{prefix}   {mindmap.description}")
    
    if mindmap.relationships:
        print(f"{prefix}   Relationships: {len(mindmap.relationships)}")
        for rel in mindmap.relationships:
            print(f"{prefix}     → {rel.target_id} ({rel.relation_type})")
    
    for child in mindmap.children:
        print_mindmap_json(child, indent + 1)


def save_mindmap_to_file(mindmap: MindmapNode, filename: str = "mindmap_output.json"):
    """
    Save mindmap to JSON file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(mindmap.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"✅ Mindmap saved to {filename}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main entrypoint for Mindmap Generator
    """
    parser = argparse.ArgumentParser(description="Generate hierarchical mindmaps and build knowledge graphs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text to generate mindmap from"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional query for detailed response from knowledge graph"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save mindmap JSON to file"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    print("="*60)
    print("MINDMAP GENERATOR")
    print("="*60)
    
    if not validate_config():
        print("\n❌ Configuration validation failed. Please check environment variables.")
        return
    
    print("✅ Configuration validated")
    
    # Initialize state
    initial_state: MindmapState = {
        "input_text": args.input,
        "mindmap_json": None,
        "graph_built": False,
        "query": args.query,
        "graph_context": None,
        "detailed_response": None,
    }
    
    # Create and run workflow
    workflow = create_mindmap_workflow()
    
    print("\n🚀 Starting mindmap generation workflow...\n")
    
    try:
        # Execute workflow
        final_state = workflow.invoke(initial_state)
        
        # Print results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        print("\n📊 MINDMAP STRUCTURE:")
        print("-" * 60)
        print_mindmap_json(final_state["mindmap_json"])
        
        if args.save:
            save_mindmap_to_file(final_state["mindmap_json"])
        
        print("\n" + "="*60)
        print("📖 DETAILED RESPONSE:")
        print("="*60)
        print(final_state["detailed_response"])
        print("\n" + "="*60)
        
        print("\n✅ Workflow completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

