from langchain_google_vertexai import VertexAIEmbeddings
from app.config import VERTEX_PROJECT, VERTEX_LOCATION

def get_embeddings():
    """Initialize Vertex AI embeddings"""
    return VertexAIEmbeddings(
        model_name="textembedding-gecko@latest",
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION
    )
