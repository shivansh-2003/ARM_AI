"""
RAG Pipeline Configuration
"""

# Document Processing Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"
NORMALIZE_EMBEDDINGS = True

# Vector Store Configuration
VECTOR_STORE_PATH = "./chroma_db"
COLLECTION_NAME = "cognito_droid_docs"

# Retrieval Configuration
RETRIEVAL_K = 5
SEARCH_TYPE = "similarity"

# LLM Configuration
LLM_MODEL = "gpt-oss:20b"
LLM_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.7

# Supported Document Types
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx"]
