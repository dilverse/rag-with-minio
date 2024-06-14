import os

LLM_MODEL = "phi3:3.8b-mini-128k-instruct-q8_0"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_ENDPOINT = "http://localhost:11434/api/chat"
EMBEDDING_ENDPOINT = "http://localhost:11434/api/embeddings"
CHAT_API_PATH = "/chat"
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
DOCS_TABLE = "docs"
EMBEDDINGS_DIM = 768
METADATA_PREFIX = "metadata"
EMBEDDING_DOCUMENT_PREFIX = "search_document"
EMBEDDING_QUERY_PREFIX = "search_query"

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["AWS_ENDPOINT"] = MINIO_ENDPOINT
os.environ["ALLOW_HTTP"] = "True"

RAG_PROMPT = """
DOCUMENT:
{documents}

QUESTION:
{user_question}

INSTRUCTIONS:
Answer in detail the user's QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT. Do not use sentence like "The document states" citing the document.
If the DOCUMENT doesn't contain the facts to answer the QUESTION only Respond with "Sorry! I Don't know"
"""
