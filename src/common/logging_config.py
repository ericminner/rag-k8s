import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class CustomJSONFormatter(logging.Formatter):
    """
    Custom JSON formatter that structures logs for cloud-native environments.
    Compatible with AWS CloudWatch, EFK stack, and other logging aggregators.
    """

    def __init__(self):
        super().__init__()
        self.service_name = os.getenv('SERVICE_NAME', 'unknown-service')
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.aws_region = os.getenv('AWS_REGION')
        self.cluster_name = os.getenv('CLUSTER_NAME')

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name,
            'environment': self.environment,
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'path': record.pathname,
            'line_number': record.lineno,
            'function': record.funcName,
        }

        # Add AWS-specific context if available
        if self.aws_region:
            log_obj['aws'] = {
                'region': self.aws_region,
                'cluster': self.cluster_name
            }

        # Add extra fields from record
        if hasattr(record, 'extras'):
            log_obj['extras'] = record.extras

        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': traceback.format_exception(*record.exc_info)
            }

        # Add request context if available
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id

        return json.dumps(log_obj)


def setup_logging(
        service_name: str,
        log_level: str = "INFO",
        request_id: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the service with proper formatting and cloud-ready structure.

    Args:
        service_name: Name of the service
        log_level: Logging level (default: INFO)
        request_id: Optional request ID for request tracking

    Returns:
        logging.Logger: Configured logger instance
    """
    # Set environment variables for the formatter
    os.environ['SERVICE_NAME'] = service_name

    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers = []

    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomJSONFormatter())
    logger.addHandler(console_handler)

    # Create a filter to add request_id to all records
    class RequestIdFilter(logging.Filter):
        def __init__(self, request_id: Optional[str]):
            super().__init__()
            self.request_id = request_id

        def filter(self, record):
            record.request_id = self.request_id
            return True

    if request_id:
        logger.addFilter(RequestIdFilter(request_id))

    return logger


class LoggerMiddleware:
    """
    FastAPI middleware for request logging and request ID injection.
    """

    def __init__(self, app, logger: logging.Logger):
        self.app = app
        self.logger = logger

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = datetime.utcnow()
        request_id = f"{start_time.timestamp()}-{os.urandom(4).hex()}"

        # Add request context to logger
        self.logger.addFilter(RequestIdFilter(request_id))

        # Log request
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": scope.get("method"),
                "path": scope.get("path")
            }
        )

        try:
            response = await self.app(scope, receive, send)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Log response
            self.logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "duration": duration,
                    "status_code": response.status_code if hasattr(response, "status_code") else None
                }
            )
            return response

        except Exception as e:
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise