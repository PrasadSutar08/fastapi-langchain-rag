from __future__ import annotations

import hashlib
import time

import redis
from fastapi import HTTPException, Request

from app.core.config import logger, settings


class RedisService:
    def __init__(self) -> None:
        self.client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as exc:
            logger.warning(f"Redis ping failed | error={exc}")
            return False

    @staticmethod
    def _client_identifier(request: Request) -> str:
        # Do not trust X-Forwarded-For unless the deployment is behind a
        # trusted proxy. By default, use the ASGI peer address.
        client = request.client
        address = client.host if client else "unknown"
        return hashlib.sha256(address.encode()).hexdigest()

    def rate_limit(
        self,
        request: Request,
        route_name: str,
        limit: int,
        window_seconds: int,
    ) -> None:
        if not settings.REDIS_RATE_LIMIT_ENABLED:
            return

        identifier = self._client_identifier(request)
        window = int(time.time()) // window_seconds
        key = f"rl:{route_name}:{identifier}:{window}"

        try:
            count = self.client.incr(key)
            if count == 1:
                self.client.expire(key, window_seconds + 1)
            if count > limit:
                ttl = max(self.client.ttl(key), 1)
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later.",
                    headers={"Retry-After": str(ttl)},
                )
        except HTTPException:
            raise
        except Exception as exc:
            # Availability-first behavior. Readiness still reports Redis failure.
            logger.warning(
                f"Redis rate limiting unavailable; failing open "
                f"| route={route_name} | error={exc}"
            )

    def get(self, key: str) -> str | None:
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        try:
            return bool(self.client.set(key, value, ex=ttl))
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False


redis_service = RedisService()
