# RAG-based Document Q&A API

Production-ready Retrieval-Augmented Generation (RAG) service for querying documents with LLMs.

## Architecture
User Question
↓
Embed Query (Vertex AI)
↓
Retrieve Similar Docs (PostgreSQL pgvector)
↓
Generate Answer (Gemini LLM)
↓
Return Answer + Sources

## Tech Stack

- **LLM**: Vertex AI (Gemini)
- **Embeddings**: Vertex AI Embeddings
- **Vector Store**: PostgreSQL with pgvector
- **API**: FastAPI
- **Framework**: LangChain
- **Testing**: Pytest

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start PostgreSQL
```bash
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -d pgvector/pgvector:latest
```

### 3. Configure .env
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

### 4. Run API
```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

## API Endpoints

### Health Check
```bash
GET /health
```

### Upload Document
```bash
POST /documents/upload
Content-Type: multipart/form-data

curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"
```

### Query Documents
```bash
POST /query
Content-Type: application/json

{
  "question": "What is machine learning?"
}
```

Response:
```json
{
  "answer": "Machine learning is...",
  "sources": ["document.pdf"],
  "latency_ms": 1234.56
}
```

## Features

✅ PDF document ingestion
✅ Vector embedding generation (Vertex AI)
✅ Semantic search with pgvector
✅ LLM-powered answer generation
✅ Production monitoring
✅ Comprehensive test suite

## Testing

```bash
pytest tests/ -v
```

## Deployment

### Google Cloud Run
```bash
gcloud run deploy rag-api \
  --source . \
  --allow-unauthenticated \
  --region us-central1
```

## Author

Saanvi Reddy Baradi
- GitHub: [@saanvireddy](https://github.com/saanvireddy)
- LinkedIn: [Saanvi Reddy](https://linkedin.com/in/saanvi-reddy)
