import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..models.openai import CompletionRequest, CompletionResponse, Choice, Message, Usage
from ..core.handler import handle_openai_request

app = FastAPI(title="OpenAI BAML Adapter", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def create_chat_completion(request: CompletionRequest, http_request: Request):
    """
    Handle OpenAI-compatible chat completion requests with tool calling support.
    """
    try:
        # Extract headers and pass to handler
        print("REQUEST")
        headers = dict(http_request.headers)

        # Pretty-print the request
        # print(f"Method: {http_request.method}")
        # print(f"URL: {http_request.url}")
        # print(f"Path: {http_request.url.path}")
        # print(f"Query params: {http_request.query_params}")
        # print(f"Headers: {dict(http_request.headers)}")
        # if http_request.method in ["POST", "PUT", "PATCH"]:
        #     body_bytes = await http_request.body()
        #     try:
        #         body = body_bytes.decode('utf-8') if body_bytes else None
        #     except UnicodeDecodeError:
        #         body = f"<binary data: {len(body_bytes)} bytes>"

        response = await handle_openai_request(request, headers)
        return response
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))