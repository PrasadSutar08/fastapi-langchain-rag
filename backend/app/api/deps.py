from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import Session

from app.core import security
from app.core.config import logger, settings
from app.core.db import get_session, get_sync_session
from app.models.user_model import TokenPayload, User

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SyncSessionDep = Annotated[Session, Depends(get_sync_session)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]


async def get_current_user(
    session: SessionDep,
    token: TokenDep,
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY_ACCESS_API,
            algorithms=[security.ALGORITHM],
        )
        token_data = TokenPayload(**payload)
        if not token_data.sub:
            raise credentials_exception
        user_id = int(token_data.sub)
    except (JWTError, ValidationError, ValueError, TypeError) as exc:
        logger.warning(f"JWT validation failed | error={exc}")
        raise credentials_exception from exc

    user = await session.get(User, user_id)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user.",
        )

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_current_active_superuser(current_user: CurrentUser) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user does not have enough privileges.",
        )
    return current_user


CurrentSuperUser = Annotated[User, Depends(get_current_active_superuser)]
