from __future__ import annotations

from ai_backend_v2.models.schemas import ChatHistoryItem
from ai_backend_v2.services.azure_search_service import azure_search_service
from ai_backend_v2.services.bedrock_service import bedrock_service
from ai_backend_v2.services.upload_session_service import UploadedArtifact
from ai_backend_v2.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an enterprise AI assistant for company knowledge.
Answer using only the provided context. If the answer is not in context, say you do not know.
Prioritize factual precision over fluency.

Response quality rules:
- Start with a direct answer section.
- Include a compact markdown table when the question involves comparisons, amounts, dates, statuses, fees, policies, or options.
- If details are missing from context, explicitly mark the missing fields as "Not found in provided sources".
- End with a short "Evidence used" bullet list citing source identifiers."""

MAX_HISTORY_MESSAGES = 20
MAX_TEXT_ARTIFACTS = 3
MAX_TEXT_ARTIFACT_CHARS = 4000
MAX_CHUNK_CHARS = 2200
MAX_CONTEXT_CHARS = 18000


def _history_to_messages(history: list[ChatHistoryItem]) -> list[dict]:
    items = list(history)[-MAX_HISTORY_MESSAGES:] if history else []
    return [{"role": item.role, "content": item.content} for item in items]


def _build_retrieval_context(docs: list[dict]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        title = doc.get("title") or f"Document {idx}"
        source = doc.get("source") or "Unknown source"
        text = (doc.get("text", "") or "").strip()[:MAX_CHUNK_CHARS]
        blocks.append(f"[{idx}] {title}\nSource: {source}\n{text}")
    return "\n\n---\n\n".join(blocks)[:MAX_CONTEXT_CHARS]


def _build_sources(docs: list[dict]) -> list[dict]:
    return [{"title": d.get("title"), "source": d.get("source"), "score": d.get("score")} for d in docs]


def process_chat(
    *,
    message: str,
    history: list[ChatHistoryItem],
    artifacts: list[UploadedArtifact] | None = None,
) -> dict:
    query = message.strip()
    docs = azure_search_service.search(query)
    if not docs:
        return {
            "response": "I couldn't find relevant company knowledge for that request.",
            "sources_count": 0,
            "sources": [],
        }

    retrieval_context = _build_retrieval_context(docs)

    text_artifacts = [a for a in (artifacts or []) if a.kind == "text" and a.text][:MAX_TEXT_ARTIFACTS]
    image_artifacts = [a for a in (artifacts or []) if a.kind == "image" and a.image_bytes]

    upload_context = "\n\n".join(
        f"Uploaded file ({artifact.name}):\n{artifact.text[:MAX_TEXT_ARTIFACT_CHARS]}"
        for artifact in text_artifacts
    )

    prompt = (
        "You must synthesize a complete, structured answer from the best matching evidence.\n\n"
        f"Company knowledge context:\n{retrieval_context}\n\n"
        f"{f'Uploaded context:\n{upload_context}\n\n' if upload_context else ''}"
        f"User question:\n{query}\n\n"
        "Output format:\n"
        "1) Direct answer\n"
        "2) Key details table (when applicable)\n"
        "3) Evidence used (source list)\n"
    )

    history_messages = _history_to_messages(history)
    if image_artifacts:
        content_blocks: list[dict] = []
        for artifact in image_artifacts:
            content_blocks.append(bedrock_service.image_block_from_bytes(artifact.image_bytes or b"", artifact.mime_type))
            content_blocks.append({"type": "text", "text": f"Uploaded image: {artifact.name}"})
        content_blocks.append({"type": "text", "text": prompt})
        messages = history_messages + [{"role": "user", "content": content_blocks}]
    else:
        messages = history_messages + [{"role": "user", "content": prompt}]

    llm_response = bedrock_service.generate_with_messages(messages, system=SYSTEM_PROMPT)
    return {
        "response": llm_response,
        "sources_count": len(docs),
        "sources": _build_sources(docs),
    }
