import shutil
import os
from config import VECTOR_DB_DIR

def cleanup_vector_store():
    """Remove existing vector store directory."""
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
        print(f"Removed vector store directory: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    cleanup_vector_store()
