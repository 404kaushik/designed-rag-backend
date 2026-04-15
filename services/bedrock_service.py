import base64
import json

import boto3

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BedrockService:
    def __init__(self) -> None:
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        self.llm_model_id = settings.bedrock_llm_model_id
        logger.info("BedrockService initialised – LLM=%s", self.llm_model_id)

    def generate_with_messages(self, messages: list[dict], system: str | None = None) -> str:
        payload: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            payload["system"] = [{"type": "text", "text": system}]

        response = self.client.invoke_model(
            modelId=self.llm_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        result = json.loads(response["body"].read())
        output_text = result["content"][0]["text"]

        input_tokens = result.get("usage", {}).get("input_tokens", "?")
        output_tokens = result.get("usage", {}).get("output_tokens", "?")
        logger.info("LLM call – in=%s out=%s len=%d", input_tokens, output_tokens, len(output_text))

        return output_text

    @staticmethod
    def image_block_from_bytes(image_bytes: bytes, mime_type: str) -> dict:
        media_type = "image/jpeg" if mime_type.lower() == "image/jpg" else mime_type.lower()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }


bedrock_service = BedrockService()
