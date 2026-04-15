from __future__ import annotations

import requests
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class AzureSearchService:
    def __init__(self) -> None:
        self.content_field = settings.azure_ai_search_content_field
        self.title_field = settings.azure_ai_search_title_field
        self.source_field = settings.azure_ai_search_source_field
        self.use_vector = settings.azure_ai_search_use_vector
        self.vector_field = settings.azure_ai_search_vector_field
        self.default_top_k = settings.azure_ai_search_top_k
        self.semantic_config_name = (settings.azure_ai_search_semantic_config_name or "").strip() or None

        self.client = SearchClient(
            endpoint=settings.azure_ai_search_endpoint,
            index_name=settings.azure_ai_search_index_name,
            credential=AzureKeyCredential(settings.azure_ai_search_api_key),
        )
        self.available_fields = self._load_index_fields()
        self.resolved_content_field = self._resolve_content_field()
        self.resolved_title_field = self._resolve_title_field()
        self.resolved_source_field = self._resolve_source_field()
        self.resolved_semantic_config = self._resolve_semantic_config_name()
        logger.info(
            "AzureSearchService initialised – index=%s vector=%s semantic=%s content=%s title=%s source=%s",
            settings.azure_ai_search_index_name,
            self.use_vector,
            self.resolved_semantic_config or "off",
            self.resolved_content_field,
            self.resolved_title_field or "none",
            self.resolved_source_field or "none",
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = top_k or self.default_top_k
        logger.info("Azure search START – top_k=%d query=%s", limit, query[:120])

        vector_queries = None
        if self.use_vector:
            vector_queries = [
                VectorizableTextQuery(
                    text=query,
                    fields=self.vector_field,
                    k_nearest_neighbors=limit,
                )
            ]

        select_fields = [self.resolved_content_field]
        if self.resolved_title_field:
            select_fields.append(self.resolved_title_field)
        if self.resolved_source_field:
            select_fields.append(self.resolved_source_field)

        search_kwargs: dict = {
            "search_text": query,
            "top": limit,
            "select": list(dict.fromkeys(select_fields)),
        }
        if vector_queries:
            search_kwargs["vector_queries"] = vector_queries
        if self.resolved_semantic_config:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = self.resolved_semantic_config
            search_kwargs["query_caption"] = "extractive"
            search_kwargs["query_answer"] = "extractive"

        try:
            results = self.client.search(**search_kwargs)
        except HttpResponseError as exc:
            if "semantic" in str(exc).lower() and self.resolved_semantic_config:
                logger.warning(
                    "Semantic query failed for config '%s'. Retrying without semantic.",
                    self.resolved_semantic_config,
                )
                for key in ("query_type", "semantic_configuration_name", "query_caption", "query_answer"):
                    search_kwargs.pop(key, None)
                results = self.client.search(**search_kwargs)
            else:
                raise

        docs: list[dict] = []
        for item in results:
            docs.append(
                {
                    "text": item.get(self.resolved_content_field, "") or "",
                    "title": item.get(self.resolved_title_field) if self.resolved_title_field else None,
                    "source": item.get(self.resolved_source_field) if self.resolved_source_field else None,
                    "score": float(item.get("@search.score", 0.0)),
                }
            )

        logger.info("Azure search END – retrieved=%d", len(docs))
        return docs

    def _load_index_fields(self) -> set[str]:
        names: set[str] = set()
        try:
            preview = self.client.search(search_text="*", top=1)
            for item in preview:
                names.update(item.keys())
                break
        except Exception:
            logger.warning("Could not infer index fields at startup; using configured field names")
        return names

    def _resolve_semantic_config_name(self) -> str | None:
        if self.semantic_config_name:
            return self.semantic_config_name
        discovered = self._discover_first_semantic_config_name()
        if discovered:
            logger.info("Auto-detected semantic config: %s", discovered)
        return discovered

    def _discover_first_semantic_config_name(self) -> str | None:
        index_url = (
            f"{settings.azure_ai_search_endpoint.rstrip('/')}/indexes/"
            f"{settings.azure_ai_search_index_name}?api-version=2024-07-01"
        )
        try:
            resp = requests.get(
                index_url,
                headers={"api-key": settings.azure_ai_search_api_key},
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            semantic_block = payload.get("semantic") or payload.get("semanticSearch") or {}
            configs = semantic_block.get("configurations", [])
            if configs:
                return configs[0].get("name")
        except Exception:
            logger.info("No semantic configuration auto-detected; semantic mode disabled.")
        return None

    def _resolve_content_field(self) -> str:
        candidates = [self.content_field, "chunk", "content", "text", "body"]
        return self._first_existing(candidates, required=True) or self.content_field

    def _resolve_title_field(self) -> str | None:
        candidates = [self.title_field, "title", "name", "file_name"]
        return self._first_existing(candidates, required=False)

    def _resolve_source_field(self) -> str | None:
        candidates = [self.source_field, "source", "parent_id", "path", "url", "chunk_id"]
        return self._first_existing(candidates, required=False)

    def _first_existing(self, candidates: list[str], required: bool) -> str | None:
        if not self.available_fields:
            return candidates[0] if candidates else None
        for name in candidates:
            if name in self.available_fields:
                return name
        if required:
            raise ValueError(
                f"No matching field found in index for candidates={candidates}. "
                "Set AZURE_AI_SEARCH_CONTENT_FIELD to a valid field name."
            )
        return None


azure_search_service = AzureSearchService()
