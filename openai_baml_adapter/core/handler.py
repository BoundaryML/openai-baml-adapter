import os
import time
import uuid
from typing import List, Any, Optional, Dict

from openai import AsyncOpenAI
from ..baml_client.baml_client.sync_client import BamlSyncClient
from ..models.openai import (
    CompletionRequest, 
    CompletionResponse, 
    Choice, 
    Message, 
    Usage,
    ToolCall,
    FunctionCall
)
from .parse import parse_openai_tools


async def handle_openai_request(request: CompletionRequest, headers: Dict[str, str]) -> CompletionResponse:
    """
    Process OpenAI tool-calling request and return a completion response.
    
    If PASSTHROUGH header is present and truthy, forward to OpenAI.
    Otherwise, process through BAML (not implemented yet).
    
    Args:
        request: OpenAI completion request with tools
        headers: HTTP headers from the request
        
    Returns:
        OpenAI completion response
    """
    # Check for PASSTHROUGH header
    passthrough = headers.get("passthrough", headers.get("PASSTHROUGH", ""))
    
    if passthrough and passthrough.lower() not in ["false", "0", ""]:
        # Forward to OpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Convert our request model to dict for OpenAI client
        request_dict = request.model_dump(exclude_none=True)
        
        # Call OpenAI
        openai_response = await client.chat.completions.create(**request_dict)
        
        # Convert OpenAI response to our response model
        return CompletionResponse(
            id=openai_response.id,
            object=openai_response.object,
            created=openai_response.created,
            model=openai_response.model,
            choices=[
                Choice(
                    index=choice.index,
                    message=Message(
                        role=choice.message.role,
                        content=choice.message.content,
                        tool_calls=[
                            ToolCall(
                                id=tc.id,
                                type=tc.type,
                                function=FunctionCall(
                                    name=tc.function.name,
                                    arguments=tc.function.arguments
                                )
                            ) for tc in (choice.message.tool_calls or [])
                        ] if choice.message.tool_calls else None
                    ),
                    finish_reason=choice.finish_reason
                ) for choice in openai_response.choices
            ],
            usage=Usage(
                prompt_tokens=openai_response.usage.prompt_tokens,
                completion_tokens=openai_response.usage.completion_tokens,
                total_tokens=openai_response.usage.total_tokens
            ) if openai_response.usage else None
        )
    
    # BAML processing not implemented
    raise NotImplementedError("baml proxy not implemented")


def convert_messages_to_baml(messages: List[Message]) -> str:
    """
    Convert OpenAI messages to BAML prompt format.
    
    TODO: Implement actual conversion logic
    """
    pass


def convert_baml_to_openai_response(baml_response: Any) -> Message:
    """
    Convert BAML response to OpenAI message format.
    
    TODO: Implement actual conversion logic
    """
    pass