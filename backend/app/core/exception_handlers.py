from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import logger


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(
        f"HTTP error | method={request.method} | path={request.url.path} "
        f"| status={exc.status_code}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
        },
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        f"Validation error | method={request.method} | path={request.url.path}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation Error",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        f"Unhandled exception | method={request.method} | path={request.url.path}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal Server Error",
        },
    )
