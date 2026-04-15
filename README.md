# AI Backend V2

FastAPI backend for a single company-knowledge chatbot using Azure AI Search for retrieval and AWS Bedrock for generation.

## Features

- Single `/chat` route (no command prefixes, no ingest endpoint)
- Azure AI Search retrieval from your existing index
- Bedrock (Claude) response generation
- Optional upload context via `/chat/upload` with session-based follow-up

## Setup

1. Copy env template:

```bash
cp ai_backend_v2/.env.example ai_backend_v2/.env
```

2. Fill required values in `ai_backend_v2/.env`:
   - AWS Bedrock credentials/model
   - Azure AI Search endpoint/api key/index name

3. Install deps:

```bash
pip install -r ai_backend_v2/requirements.txt
```

4. Run server from workspace root:

```bash
uvicorn ai_backend_v2.main:app --reload --host 0.0.0.0 --port 8001
```

## API

### `GET /health`

Returns:

```json
{ "status": "ok" }
```

### `POST /chat/upload`

Multipart form-data:
- `files`: one or more files
- `session_id` (optional)

Returns session and accepted file statuses.

### `POST /chat`

Request:

```json
{
  "message": "What is our transfer out fee policy?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "session_id": "optional-session-id"
}
```

Response:

```json
{
  "response": "Answer from company knowledge...",
  "sources_count": 5,
  "sources": [
    { "title": "Policy Doc", "source": "blob/path/file.pdf", "score": 0.93 }
  ],
  "session_id": "optional-session-id"
}
```

## Notes

- You do not need local Qdrant for this backend.
- Hybrid retrieval is enabled by default (`search_text + vector`) for richer results.
- If `AZURE_AI_SEARCH_SEMANTIC_CONFIG_NAME` is blank, the backend will auto-detect a semantic config from the index when available.
- Keep `AZURE_AI_SEARCH_VECTOR_FIELD` aligned with your index vector field (for your current index: `text_vector`).
