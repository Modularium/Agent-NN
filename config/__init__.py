import os

# Global configuration variables
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vectorstore")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Default models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Create vector store directory if it doesn't exist
os.makedirs(VECTOR_DB_PATH, exist_ok=True)