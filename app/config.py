import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")
VERTEX_PROJECT = os.getenv("VERTEX_AI_PROJECT_ID", "your-project")
VERTEX_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")