from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from app.embeddings import get_embeddings
from app.db import get_connection
from typing import List, Tuple
import time

class RAGPipeline:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.llm = VertexAI(model_name="gemini-pro")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_documents(self, file_path: str) -> List[dict]:
        """Load PDF and chunk into smaller pieces"""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            chunks = self.splitter.split_documents(docs)
            return [
                {"content": c.page_content, "source": file_path} 
                for c in chunks
            ]
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def store_documents(self, documents: List[dict]):
        """Store documents with embeddings in PostgreSQL"""
        conn = get_connection()
        cursor = conn.cursor()
        
        for doc in documents:
            # Generate embedding
            embedding = self.embeddings.embed_query(doc["content"])
            
            # Store in DB
            cursor.execute(
                "INSERT INTO documents (filename, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
                (
                    doc["source"],
                    doc["content"],
                    embedding,
                    '{"source": "' + doc["source"] + '"}'
                )
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Stored {len(documents)} document chunks")
    
    def query(self, question: str) -> Tuple[str, List[str], float]:
        """RAG query: retrieve relevant docs + generate answer"""
        start_time = time.time()
        
        try:
            # Step 1: Embed the question
            query_embedding = self.embeddings.embed_query(question)
            
            # Step 2: Retrieve top 3 relevant documents
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content, filename FROM documents 
                ORDER BY embedding <=> %s 
                LIMIT 3
            """, (query_embedding,))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Step 3: Check if we found documents
            if not results:
                latency = (time.time() - start_time) * 1000
                return "No relevant documents found.", [], latency
            
            # Step 4: Build context from retrieved documents
            context = "\n\n".join([f"[Source: {r[1]}]\n{r[0]}" for r in results])
            sources = list(set([r[1] for r in results]))
            
            # Step 5: Generate answer using LLM
            prompt = f"""Based on the following documents, answer the question concisely and accurately.

Documents:
{context}

Question: {question}

Answer:"""
            
            answer = self.llm.invoke(prompt)
            latency = (time.time() - start_time) * 1000
            
            return answer, sources, latency
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return f"Error: {str(e)}", [], latency