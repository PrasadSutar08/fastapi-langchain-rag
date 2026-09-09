from fastapi import APIRouter, Depends, Request

from app.api.deps import CurrentUser
from app.core.config import settings
from app.schemas.search_schema import SearchRequest, SearchResponse, SearchResult
from app.services.redis_service import redis_service
from app.services.search_service import search_service

router = APIRouter(prefix="/search", tags=["Search"])


def search_rate_limit(request: Request):
    return redis_service.rate_limit(
        request=request,
        route_name="search",
        limit=settings.SEARCH_RATE_LIMIT,
        window_seconds=settings.SEARCH_RATE_LIMIT_WINDOW_SECONDS,
    )


@router.post("", response_model=SearchResponse, dependencies=[Depends(search_rate_limit)])
def search_documents(request: SearchRequest, current_user: CurrentUser):
    results = search_service.search(
        query=request.query,
        user_id=current_user.id,
        top_k=request.top_k,
        search_type=request.search_type,
        document_id=request.document_id,
        filename=request.filename,
        is_superuser=current_user.is_superuser,
    )
    return SearchResponse(
        query=request.query,
        results=[SearchResult(**result) for result in results],
    )
