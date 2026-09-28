from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    search_type: Literal["similarity", "mmr"] = "mmr"
    document_id: int | None = Field(default=None, gt=0)
    filename: str | None = Field(default=None, min_length=1, max_length=512)

    model_config = ConfigDict(str_strip_whitespace=True)


class SearchResult(BaseModel):
    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    content: str
    score: float | None = None
    source: str | None = None
    chunk_index: int | None = None

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
