from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from contextlib import asynccontextmanager
from common.logging_config import setup_logging, LoggerMiddleware
from common.config import get_db_settings, get_service_settings
from common.database import get_db_connection

# Initialize settings
settings = get_service_settings()
db_settings = get_db_settings()

# Initialize logging
logger = setup_logging("rag-service", settings.LOG_LEVEL)


class Query(BaseModel):
    question: str
    top_k: Optional[int] = 3


class SearchResult(BaseModel):
    document_title: str
    content: str
    similarity_score: float


# Initialize model in lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    logger.info("Loading embedding model...")
    try:
        app.state.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down RAG service...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(LoggerMiddleware, logger=logger)


@app.post("/query", response_model=List[SearchResult])
async def query_documents(query: Query, request: Request):
    logger.info(f"Processing query: {query.question}")
    try:
        # Generate embedding using app state model
        query_embedding = app.state.model.encode(query.question)
        logger.info(f"Query embedding generated, shape: {query_embedding.shape}")

        # Search for similar documents
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check document counts
                cur.execute("SELECT COUNT(*) FROM documents")
                doc_count = cur.fetchone()[0]
                logger.info(f"Total documents in database: {doc_count}")

                cur.execute("SELECT COUNT(*) FROM embeddings")
                emb_count = cur.fetchone()[0]
                logger.info(f"Total embeddings in database: {emb_count}")

                logger.info("Executing similarity search")
                cur.execute(
                    """
                    SELECT 
                        d.title,
                        d.content,
                        1 - (e.embedding <=> %s::vector) as similarity
                    FROM 
                        documents d
                        JOIN embeddings e ON d.id = e.document_id
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query.top_k)
                )

                results = []
                for row in cur.fetchall():
                    result = SearchResult(
                        document_title=row[0],
                        content=row[1],
                        similarity_score=float(row[2])
                    )
                    logger.debug(f"Found match: {result.document_title} with score {result.similarity_score}")
                    results.append(result)

                logger.info(f"Search completed, found {len(results)} results")
                return results

    except Exception as e:
        logger.error("Error processing query", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        # Verify model is loaded
        if not hasattr(app.state, 'model'):
            raise RuntimeError("Model not initialized")

        return {
            "status": "healthy",
            "database": "connected",
            "model": "loaded",
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/diagnostics")
async def get_diagnostics():
    """System diagnostics endpoint"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check documents
                cur.execute("SELECT COUNT(*) FROM documents")
                doc_count = cur.fetchone()[0]

                # Check embeddings
                cur.execute("SELECT COUNT(*) FROM embeddings")
                emb_count = cur.fetchone()[0]

                # Get sample document
                cur.execute(
                    "SELECT title, content FROM documents LIMIT 1"
                )
                sample_doc = cur.fetchone()

                # Get sample embedding
                cur.execute("SELECT embedding FROM embeddings LIMIT 1")
                sample_emb = cur.fetchone()

                return {
                    "document_count": doc_count,
                    "embedding_count": emb_count,
                    "sample_document": {
                        "title": sample_doc[0] if sample_doc else None,
                        "content": sample_doc[1] if sample_doc else None
                    } if sample_doc else None,
                    "sample_embedding_shape": len(sample_emb[0]) if sample_emb else None,
                    "database_connection": "success",
                    "environment": settings.ENVIRONMENT
                }
    except Exception as e:
        logger.error("Diagnostics check failed", exc_info=True)
        return {
            "error": str(e),
            "database_connection": "failed",
            "environment": settings.ENVIRONMENT
        }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting RAG service in {settings.ENVIRONMENT} environment")
    uvicorn.run(app, host="0.0.0.0", port=8000)