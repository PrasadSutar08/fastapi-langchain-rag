from datetime import timedelta

from jose import jwt

from app.core import security
from app.core.config import settings


def test_access_token_contains_subject():
    token = security.create_access_token(
        subject=123,
        expires_delta=timedelta(minutes=5),
    )
    payload = jwt.decode(
        token,
        settings.SECRET_KEY_ACCESS_API,
        algorithms=[security.ALGORITHM],
    )
    assert payload["sub"] == "123"
    assert "exp" in payload
    assert "iat" in payload


def test_password_hash_roundtrip():
    password = "Correct-Horse-Battery-Staple-123!"
    hashed = security.get_password_hash(password)
    assert hashed != password
    assert security.verify_password(password, hashed)
    assert not security.verify_password("wrong", hashed)
