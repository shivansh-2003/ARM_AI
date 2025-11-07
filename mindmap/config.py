"""
Central configuration module for Mindmap Generator
Handles Ollama, Neo4j, and LangChain settings
"""

import os
from typing import Optional

# ============================================================================
# Ollama Configuration
# ============================================================================

OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.1  # Deterministic outputs for structured generation

# ============================================================================
# Neo4j Configuration
# ============================================================================

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "study"
NEO4J_PASSWORD =  "shivansh"

# Neo4j Database Configuration
NEO4J_DATABASE = "study"

# ============================================================================
# LangChain Configuration
# ============================================================================

# Structured output configuration
MAX_RETRIES = 3
INCLUDE_RAW = False

# Graph configuration
NODE_LABEL = "Concept"
RELATIONSHIP_TYPE = "RELATED_TO"

# ============================================================================
# Validation Functions
# ============================================================================

def validate_config() -> bool:
    """
    Validate that all required configuration is present
    
    Returns:
        bool: True if configuration is valid
    """
    required_vars = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    
    if missing:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        return False
    
    return True


def get_neo4j_connection_params() -> dict:
    """
    Get Neo4j connection parameters as dictionary
    
    Returns:
        dict: Connection parameters for Neo4j
    """
    return {
        "url": NEO4J_URI,
        "username": NEO4J_USERNAME,
        "password": NEO4J_PASSWORD,
        "database": NEO4J_DATABASE,
    }

