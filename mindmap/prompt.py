"""
Reusable LangChain prompt templates for Mindmap Generator
Defines prompts for mindmap generation and graph querying
"""

from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# Mindmap Generation System Prompt
# ============================================================================

MINDMAP_SYSTEM_PROMPT = """You are an expert knowledge structuring assistant. Your task is to analyze the provided text and extract a hierarchical mindmap of key concepts, subtopics, and their relationships.

**CRITICAL INSTRUCTIONS:**
1. Identify the main topic as the root node
2. Extract 3-7 major subtopics as direct children
3. For each subtopic, identify 2-5 supporting concepts
4. Capture relationships between concepts (prerequisites, related, contrasts, etc.)
5. Each node must have: unique id, label, description, and list of children
6. Relationships must specify: source_id, target_id, relation_type

**SCHEMA REQUIREMENTS:**
- id: Unique identifier (e.g., "node_1", "node_2")
- label: Concise concept name (2-5 words)
- description: Brief explanation (1-2 sentences)
- children: List of child nodes (recursive structure)
- relationships: List of edges to other nodes with relation_type

**RELATION TYPES:**
- "prerequisite": Node A must be understood before Node B
- "related": Nodes cover similar themes
- "contrasts": Opposing or alternative concepts
- "example": Concrete instance of abstract concept
- "part_of": Component relationship

**OUTPUT:** Return ONLY the structured JSON mindmap. Do NOT include any explanatory text before or after the JSON."""

# ============================================================================
# Mindmap Generation Prompt Template
# ============================================================================

MINDMAP_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", MINDMAP_SYSTEM_PROMPT),
    ("human", """Analyze the following text and generate a comprehensive mindmap:

TEXT:
{input_text}

Generate the mindmap with hierarchical structure and relationships.""")
])

# ============================================================================
# Graph Query System Prompt
# ============================================================================

GRAPH_QUERY_SYSTEM_PROMPT = """You are a knowledge graph assistant. Your task is to provide detailed, contextual responses by analyzing the knowledge graph structure.

**INSTRUCTIONS:**
1. Use the provided graph context to understand concept relationships
2. Synthesize information from multiple connected nodes
3. Explain hierarchies, dependencies, and cross-references
4. Provide comprehensive answers grounded in the graph structure
5. Cite specific concepts and relationships from the graph

**OUTPUT FORMAT:**
- Start with a concise summary (2-3 sentences)
- Detail main concepts and their relationships
- Highlight prerequisites, related concepts, and examples
- End with key takeaways or connections"""

# ============================================================================
# Graph Query Prompt Template
# ============================================================================

GRAPH_QUERY_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", GRAPH_QUERY_SYSTEM_PROMPT),
    ("human", """Based on the knowledge graph context below, provide a detailed response to the query.

GRAPH CONTEXT:
{graph_context}

QUERY: {query}

Detailed Response:""")
])

# ============================================================================
# Schema Example for Reference
# ============================================================================

MINDMAP_SCHEMA_EXAMPLE = """
Example mindmap structure:
{
  "id": "root",
  "label": "Quantum Computing",
  "description": "Computing paradigm based on quantum mechanics principles",
  "children": [
    {
      "id": "qubits",
      "label": "Qubits",
      "description": "Quantum bits that exist in superposition states",
      "children": [
        {
          "id": "superposition",
          "label": "Superposition",
          "description": "Quantum state existing in multiple states simultaneously",
          "children": [],
          "relationships": []
        }
      ],
      "relationships": [
        {
          "source_id": "superposition",
          "target_id": "entanglement",
          "relation_type": "related"
        }
      ]
    }
  ],
  "relationships": []
}
"""

