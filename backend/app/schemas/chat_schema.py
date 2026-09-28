from pydantic import BaseModel, ConfigDict, Field


class ChatBody(BaseModel):
    conversation_id: int = Field(..., gt=0)
    message: str = Field(..., min_length=1, max_length=10000)

    model_config = ConfigDict(str_strip_whitespace=True)


class ChatSource(BaseModel):
    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    source: str | None = None
    chunk_index: int | None = None


class ChatResponseData(BaseModel):
    conversation_id: int
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    user_message_id: int | None = None
    assistant_message_id: int | None = None


class ChatResponse(BaseModel):
    data: ChatResponseData
