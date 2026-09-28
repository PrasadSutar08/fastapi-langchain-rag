from fastapi import APIRouter

from app.api.routes import conversation, document, login, qa, search

api_router = APIRouter()
api_router.include_router(login.router, tags=["Authentication"])
api_router.include_router(qa.router)
api_router.include_router(conversation.router, prefix="/conversations", tags=["Conversations"])
api_router.include_router(document.router)
api_router.include_router(search.router)
