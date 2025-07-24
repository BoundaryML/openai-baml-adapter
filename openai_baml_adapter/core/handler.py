import json
import os
import time
import uuid
from typing import List, Any, Optional, Dict

from openai import AsyncOpenAI
from ..baml_client.baml_client.sync_client import b
from ..baml_client.baml_client.type_builder import TypeBuilder
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
    
    # BAML processing
    # Initialize BAML client
    
    tb = TypeBuilder()
    # Convert Tool objects to dicts for parse_openai_tools
    print(request.tools)
    tools_dict = [tool.model_dump() for tool in request.tools] if request.tools else []
    parsed_tools = parse_openai_tools(tools_dict, tb)
    
    # Extract just the FieldType objects from the parsed tools
    tool_types = [field_type for field_type, _ in parsed_tools.values()]
    print("tool_types")
    print(tool_types)
    
    # Create union of tool types if any exist
    if tool_types:
        tb.Response.add_property("tool_call", tb.union(tool_types))
    
    # Call BAML function with empty string
    baml_response = b.BamlFunction("", baml_options={"tb": tb})
    
    # Process BAML response and convert to OpenAI format
    message = Message(role="assistant", content=None)
    
    # Debug print to see the actual response structure
    print(f"BAML response type: {type(baml_response)}")
    print(f"BAML response: {baml_response}")
    
    # Check if BAML response is a dict or has tool_call attribute
    tool_call_data = None
    if isinstance(baml_response, dict) and "tool_call" in baml_response:
        tool_call_data = baml_response["tool_call"]
    elif hasattr(baml_response, "tool_call"):
        tool_call_data = baml_response.tool_call
    
    if tool_call_data:
        # Handle both dict and object cases
        if isinstance(tool_call_data, dict):
            function_name = tool_call_data.get("function_name")
            # Create args dict without function_name
            args_dict = {k: v for k, v in tool_call_data.items() if k != "function_name"}
        else:
            # Object case
            function_name = getattr(tool_call_data, "function_name", None)
            args_dict = {}
            for field_name in dir(tool_call_data):
                if not field_name.startswith("_") and field_name != "function_name":
                    value = getattr(tool_call_data, field_name)
                    if value is not None:
                        args_dict[field_name] = value
        
        if function_name:
            # Create the tool call
            message.tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=function_name,
                        arguments=json.dumps(args_dict)
                    )
                )
            ]
        else:
            # No function name found, return a regular message
            message.content = "No tool was called"
    else:
        # No tool_call in response
        message.content = "No tool was called"
    
    # Create the OpenAI response
    return CompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason="tool_calls" if message.tool_calls else "stop"
            )
        ],
        usage=Usage(
            prompt_tokens=100,  # TODO: Get actual token counts from BAML
            completion_tokens=50,
            total_tokens=150
        )
    )


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