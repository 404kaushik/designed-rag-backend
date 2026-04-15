from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_backend_v2.config import settings
from ai_backend_v2.routes import chat, health
from ai_backend_v2.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG Enterprise AI Backend V2",
    description="Single-chatbot RAG backend using Azure AI Search and AWS Bedrock",
    version="2.0.0",
)

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(health.router)

logger.info("FastAPI app v2 initialised – routes: /chat, /chat/upload, /health")
