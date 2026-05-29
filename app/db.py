import psycopg2
from pgvector.psycopg2 import register_vector
from app.config import DATABASE_URL

def init_db():
    """Initialize database schema"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                content TEXT,
                embedding vector(768),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print(" Database initialized successfully")
    except Exception as e:
        print(f" Database error: {str(e)}")

def get_connection():
    """Get database connection"""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn