from typing import Literal

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryItem] = Field(default_factory=list)
    session_id: str | None = None


class SourceItem(BaseModel):
    title: str | None = None
    source: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    response: str
    sources_count: int
    sources: list[SourceItem] = Field(default_factory=list)
    session_id: str | None = None


class UploadedFileStatus(BaseModel):
    name: str
    mime_type: str
    status: Literal["accepted", "unsupported", "failed"]
    detail: str | None = None


class UploadResponse(BaseModel):
    session_id: str
    files: list[UploadedFileStatus]
    accepted_count: int
