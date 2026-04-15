from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    azure_ai_search_endpoint: str
    azure_ai_search_api_key: str
    azure_ai_search_index_name: str
    azure_ai_search_content_field: str = "chunk"
    azure_ai_search_title_field: str = "title"
    azure_ai_search_source_field: str = "parent_id"
    azure_ai_search_semantic_config_name: str | None = None
    azure_ai_search_use_vector: bool = True
    azure_ai_search_vector_field: str = "text_vector"
    azure_ai_search_top_k: int = 10

    upload_session_ttl_seconds: int = 3600
    cors_origins: str = "https://designedaiv2.netlify.app"

    model_config = SettingsConfigDict(
        env_file="ai_backend_v2/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
