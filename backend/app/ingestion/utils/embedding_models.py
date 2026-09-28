"""Compatibility wrapper around the application's singleton embedding service."""
from app.services.embedding_service import get_embedding_model

__all__ = ["get_embedding_model"]
