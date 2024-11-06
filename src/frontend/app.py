import streamlit as st
import requests
from requests.exceptions import RequestException
from common.logging_config import setup_logging
from common.config import get_service_settings
from typing import Tuple, Optional, Dict, Any, List
import time

# Initialize settings and logging
settings = get_service_settings()
logger = setup_logging("frontend-service", settings.LOG_LEVEL)


def make_request(
        method: str,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: int = 10
) -> Tuple[bool, Any]:
    """Generic request handler with retry logic and logging"""
    try:
        start_time = time.time()
        response = requests.request(
            method=method,
            url=url,
            json=json,
            timeout=timeout
        )
        duration = time.time() - start_time

        logger.info(
            f"API request completed",
            extra={
                "method": method,
                "url": url,
                "duration": f"{duration:.2f}s",
                "status_code": response.status_code
            }
        )

        response.raise_for_status()
        return True, response.json()
    except RequestException as e:
        logger.error(
            "API request failed",
            extra={
                "method": method,
                "url": url,
                "error": str(e)
            },
            exc_info=True
        )
        return False, f"Error: {str(e)}"


def add_document(title: str, content: str, source: str = None) -> Tuple[bool, str]:
    """Add a document to the system via the embedding service."""
    logger.info(f"Adding document: {title}")

    payload = {
        "title": title,
        "content": content,
        "metadata": {"source": source} if source else {}
    }

    success, result = make_request(
        method="POST",
        url=f"{settings.EMBEDDING_SERVICE_URL}/embed",
        json=payload
    )

    if success:
        logger.info(f"Document added successfully: {title}")
        return True, "Document added successfully!"
    return False, result


def query_documents(question: str, top_k: int) -> Tuple[bool, List[Dict[str, Any]]]:
    """Query the RAG system for answers."""
    logger.info(f"Querying system with: {question}")

    payload = {
        "question": question,
        "top_k": top_k
    }

    success, results = make_request(
        method="POST",
        url=f"{settings.RAG_SERVICE_URL}/query",
        json=payload
    )

    if success:
        logger.info(f"Query successful, found {len(results)} results")
        return True, results
    return False, [{"error": results}]


def check_services_health() -> Dict[str, str]:
    """Check if all required services are healthy."""
    services = {
        "Embedding Service": f"{settings.EMBEDDING_SERVICE_URL}/health",
        "RAG Service": f"{settings.RAG_SERVICE_URL}/health"
    }

    status = {}
    for service_name, url in services.items():
        success, _ = make_request(
            method="GET",
            url=url,
            timeout=5
        )
        status[service_name] = "✅ Online" if success else "❌ Offline"
    return status


def main():
    # Page configuration
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Document Q&A System")

    # Environment banner
    if settings.ENVIRONMENT != "production":
        st.warning(f"Running in {settings.ENVIRONMENT} environment")

    # System Status
    if st.sidebar.checkbox("Show System Status"):
        st.sidebar.subheader("System Status")
        status = check_services_health()
        for service, state in status.items():
            st.sidebar.text(f"{service}: {state}")

    # Document Addition Section
    st.sidebar.header("Add New Document")
    doc_title = st.sidebar.text_input("Document Title")
    doc_content = st.sidebar.text_area("Document Content")
    doc_source = st.sidebar.text_input("Source (optional)")

    if st.sidebar.button("Add Document"):
        if doc_title and doc_content:
            with st.sidebar.spinner("Adding document..."):
                success, message = add_document(doc_title, doc_content, doc_source)
                if success:
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)
        else:
            st.sidebar.warning("Please fill in both title and content")

    # Sample Document Section
    st.sidebar.markdown("---")
    st.sidebar.header("Add Sample Document")
    if st.sidebar.button("Add Sample Kubernetes Doc"):
        sample_doc = {
            "title": "Kubernetes Introduction",
            "content": """Kubernetes is an open-source container orchestration platform that automates the deployment, 
            scaling, and management of containerized applications. It groups containers into logical units for easy 
            management and discovery. Some key Kubernetes features include automatic bin packing, self-healing, 
            horizontal scaling, and service discovery and load balancing.""",
            "source": "sample"
        }
        with st.sidebar.spinner("Adding sample document..."):
            success, message = add_document(
                sample_doc["title"],
                sample_doc["content"],
                sample_doc["source"]
            )
            if success:
                st.sidebar.success("Sample document added!")
            else:
                st.sidebar.error(message)

    # Query Section
    st.header("Ask Questions")
    query = st.text_area("Enter your question")
    col1, col2 = st.columns([3, 1])
    top_k = col2.slider("Number of results", min_value=1, max_value=5, value=3)

    if col1.button("Search"):
        if query:
            with st.spinner("Searching for answers..."):
                success, results = query_documents(query, top_k)
                if success:
                    if not results:
                        st.info("No relevant documents found.")
                    else:
                        for i, result in enumerate(results, 1):
                            with st.expander(
                                    f"Result {i}: {result['document_title']} "
                                    f"(Score: {result['similarity_score']:.2f})"
                            ):
                                st.markdown(result['content'])
                else:
                    st.error(str(results))
        else:
            st.warning("Please enter a question")

    # Help Section
    with st.expander("ℹ️ Help"):
        st.markdown("""
        ### How to use this system:
        1. **Add documents** using the sidebar form
        2. **Ask questions** in the main area
        3. View **multiple results** ranked by relevance

        ### Tips:
        - Add detailed documents for better results
        - Be specific in your questions
        - Adjust the number of results using the slider
        """)


if __name__ == "__main__":
    logger.info(f"Starting frontend service in {settings.ENVIRONMENT} environment")
    main()