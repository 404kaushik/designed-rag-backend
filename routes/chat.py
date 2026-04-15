from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai_backend_v2.models.schemas import ChatRequest, ChatResponse, UploadResponse
from ai_backend_v2.services.file_parse_service import parse_uploads
from ai_backend_v2.services.rag_service import process_chat
from ai_backend_v2.services.upload_session_service import upload_session_store
from ai_backend_v2.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    session = upload_session_store.create_or_get(session_id)
    artifacts, statuses = await parse_uploads(files)
    upload_session_store.append_artifacts(session.session_id, artifacts)
    accepted_count = sum(1 for s in statuses if s.status == "accepted")
    return UploadResponse(
        session_id=session.session_id,
        files=statuses,
        accepted_count=accepted_count,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    body = request.message.strip()
    if not body:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        session = upload_session_store.get(request.session_id)
        artifacts = session.artifacts if session else []
        result = process_chat(message=body, history=request.history, artifacts=artifacts)
        return ChatResponse(**result, session_id=request.session_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /chat FAILED")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
