import os

# Global configuration variables
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vectorstore")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Ensure required environment variables are set
if not LLM_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")
