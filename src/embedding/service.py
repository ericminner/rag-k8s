from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from psycopg2.extras import Json as PsycopgJson
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from common.logging_config import setup_logging, LoggerMiddleware
from common.config import get_db_settings, get_service_settings
from common.database import get_db_connection

# Initialize settings
settings = get_service_settings()
db_settings = get_db_settings()

# Initialize logging
logger = setup_logging("embedding-service", settings.LOG_LEVEL)


class Document(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


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
        logger.info("Shutting down embedding service...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(LoggerMiddleware, logger=logger)


@app.post("/embed")
async def create_embedding(document: Document, request: Request):
    """Create and store document embeddings"""
    logger.info("Processing document", extra={"document_title": document.title})

    try:
        # Generate embedding
        logger.debug("Generating embedding...")
        embedding = app.state.model.encode(document.content)
        logger.debug("Embedding generated successfully")

        # Store document and its embedding
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                logger.debug("Storing document...")
                metadata_json = PsycopgJson(document.metadata) if document.metadata else None

                # Insert document with retry logic
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        cur.execute(
                            """
                            INSERT INTO documents (title, content, metadata)
                            VALUES (%s, %s, %s)
                            RETURNING id
                            """,
                            (document.title, document.content, metadata_json)
                        )
                        document_id = cur.fetchone()[0]
                        logger.info("Document stored", extra={"document_id": document_id})
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            raise
                        logger.warning(f"Retrying document insertion (attempt {retry_count})", exc_info=True)

                # Insert embedding with retry logic
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        cur.execute(
                            """
                            INSERT INTO embeddings (document_id, embedding)
                            VALUES (%s, %s)
                            RETURNING id
                            """,
                            (document_id, embedding.tolist())
                        )
                        embedding_id = cur.fetchone()[0]
                        logger.info("Embedding stored", extra={"embedding_id": embedding_id})
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            raise
                        logger.warning(f"Retrying embedding insertion (attempt {retry_count})", exc_info=True)

                conn.commit()

                return {
                    "id": document_id,
                    "title": document.title,
                    "embedding_id": embedding_id,
                    "status": "success"
                }

    except Exception as e:
        logger.error("Error processing document", exc_info=True)
        # Ensure we don't expose internal error details to client
        raise HTTPException(
            status_code=500,
            detail="Failed to process document. Please try again later."
        )


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
            "error": str(e),
            "environment": settings.ENVIRONMENT
        }


@app.get("/diagnostics")
async def get_diagnostics():
    """System diagnostics endpoint"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get embedding stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_embeddings,
                        AVG(array_length(embedding, 1)) as avg_embedding_size,
                        MIN(created_at) as oldest_embedding,
                        MAX(created_at) as newest_embedding
                    FROM embeddings
                """)
                stats = cur.fetchone()

                return {
                    "embedding_count": stats[0],
                    "average_embedding_size": float(stats[1]) if stats[1] else 0,
                    "oldest_embedding": stats[2].isoformat() if stats[2] else None,
                    "newest_embedding": stats[3].isoformat() if stats[3] else None,
                    "model_name": "all-MiniLM-L6-v2",
                    "environment": settings.ENVIRONMENT,
                    "database_connection": "healthy"
                }
    except Exception as e:
        logger.error("Diagnostics check failed", exc_info=True)
        return {
            "error": str(e),
            "database_connection": "unhealthy",
            "environment": settings.ENVIRONMENT
        }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting embedding service in {settings.ENVIRONMENT} environment")
    uvicorn.run(app, host="0.0.0.0", port=8000)