from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from typing import List, Tuple
import time
import os
from dotenv import load_dotenv
load_dotenv()
CHROMA_DIR = "./chroma_db"

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
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
        """Store documents with embeddings in ChromaDB"""
        try:
            texts = [d["content"] for d in documents]
            metadatas = [{"source": d["source"]} for d in documents]
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            print(f"✅ Stored {len(documents)} document chunks")
        except Exception as e:
            raise Exception(f"Error storing documents: {str(e)}")

    def query(self, question: str) -> Tuple[str, List[str], float]:
        """RAG query: retrieve relevant docs + generate answer"""
        start_time = time.time()
        try:
            # Retrieve top 3 relevant chunks
            results = self.vectorstore.similarity_search(question, k=3)

            if not results:
                latency = (time.time() - start_time) * 1000
                return "No relevant documents found. Please upload a PDF first.", [], latency

            # Build context
            context = "\n\n".join([
                f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.page_content}"
                for r in results
            ])
            sources = list(set([r.metadata.get('source', 'unknown') for r in results]))

            # Generate answer
            prompt = f"""Based on the following documents, answer the question concisely and accurately.
If the answer is not in the documents, say so clearly.

Documents:
{context}

Question: {question}

Answer:"""

            response = self.llm.invoke(prompt)
            answer = response.content
            latency = (time.time() - start_time) * 1000

            return answer, sources, latency

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return f"Error: {str(e)}", [], latency