from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import SyncSessionDep
from app.core import security
from app.core.config import settings
from app.crud import user_crud
from app.models.user_model import Token, UserRegister
from app.services.redis_service import redis_service

router = APIRouter()


def login_rate_limit(request: Request):
    return redis_service.rate_limit(
        request, "login", settings.LOGIN_RATE_LIMIT,
        settings.AUTH_RATE_LIMIT_WINDOW_SECONDS,
    )


def register_rate_limit(request: Request):
    return redis_service.rate_limit(
        request, "register", settings.REGISTER_RATE_LIMIT,
        settings.AUTH_RATE_LIMIT_WINDOW_SECONDS,
    )


@router.post(
    "/login/access-token",
    response_model=Token,
    dependencies=[Depends(login_rate_limit)],
)
def login_access_token(
    session: SyncSessionDep,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    user = user_crud.authenticate(
        session=session,
        email=form_data.username.strip().lower(),
        password=form_data.password,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(403, "Inactive user.")

    return Token(
        access_token=security.create_access_token(
            user.id,
            timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        )
    )


@router.post(
    "/login/register",
    status_code=201,
    dependencies=[Depends(register_rate_limit)],
)
def register_user(
    user_in: UserRegister,
    session: SyncSessionDep,
):
    email = user_in.email.strip().lower()
    if user_crud.get_user_by_email(session=session, email=email):
        raise HTTPException(
            status_code=409,
            detail="A user with this email already exists.",
        )

    user_in.email = email
    user = user_crud.create_registered_user(
        session=session,
        user_in=user_in,
    )
    return {
        "message": "User registered successfully.",
        "data": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
        },
    }
