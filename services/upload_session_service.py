"""Ephemeral in-memory upload session store."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UploadedArtifact:
    name: str
    mime_type: str
    kind: str  # "image" | "text"
    text: str | None = None
    image_bytes: bytes | None = None


@dataclass
class UploadSession:
    session_id: str
    artifacts: list[UploadedArtifact] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class UploadSessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, UploadSession] = {}

    def create_or_get(self, session_id: str | None = None) -> UploadSession:
        with self._lock:
            self._evict_expired_locked()
            sid = session_id or str(uuid.uuid4())
            session = self._sessions.get(sid)
            if session is None:
                session = UploadSession(session_id=sid)
                self._sessions[sid] = session
            session.updated_at = time.time()
            return session

    def append_artifacts(self, session_id: str, artifacts: list[UploadedArtifact]) -> UploadSession:
        with self._lock:
            self._evict_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                session = UploadSession(session_id=session_id)
                self._sessions[session_id] = session
            session.artifacts.extend(artifacts)
            session.updated_at = time.time()
            return session

    def get(self, session_id: str | None) -> UploadSession | None:
        if not session_id:
            return None
        with self._lock:
            self._evict_expired_locked()
            session = self._sessions.get(session_id)
            if session:
                session.updated_at = time.time()
            return session

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [sid for sid, sess in self._sessions.items() if now - sess.updated_at > self.ttl_seconds]
        for sid in expired:
            del self._sessions[sid]


upload_session_store = UploadSessionStore(ttl_seconds=settings.upload_session_ttl_seconds)
