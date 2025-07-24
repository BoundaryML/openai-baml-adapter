import os
import pytest
from fastapi.testclient import TestClient
from openai_baml_adapter.api.main import app

client = TestClient(app)


def test_chat_completions_with_tools():
    """Test the /v1/chat/completions endpoint with Greet and GetWeather tools."""
    
    # Define the tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Greet",
                "description": "Greet a person by name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the person to greet"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "GetWeather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude of the location"
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude of the location"
                        }
                    },
                    "required": ["latitude", "longitude"]
                }
            }
        }
    ]
    
    # Create the request
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Greet John and get the weather for San Francisco (37.7749, -122.4194)"
            }
        ],
        "tools": tools,
        "temperature": 0.7
    }
    
    # Make the request
    response = client.post("/v1/chat/completions", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["object"] == "chat.completion"
    assert "id" in response_data
    assert "created" in response_data
    assert "model" in response_data
    assert "choices" in response_data
    assert len(response_data["choices"]) > 0
    
    # Check the first choice
    choice = response_data["choices"][0]
    assert "message" in choice
    assert "role" in choice["message"]
    assert choice["message"]["role"] == "assistant"
    
    # Since we provided tools, check if tool_calls are present in stub
    if "tool_calls" in choice["message"]:
        tool_calls = choice["message"]["tool_calls"]
        assert len(tool_calls) > 0
        assert "id" in tool_calls[0]
        assert "type" in tool_calls[0]
        assert tool_calls[0]["type"] == "function"
        assert "function" in tool_calls[0]
        assert "name" in tool_calls[0]["function"]
        assert tool_calls[0]["function"]["name"] in ["Greet", "GetWeather"]


def test_chat_completions_without_tools():
    """Test the /v1/chat/completions endpoint without tools."""
    
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "temperature": 0.7
    }
    
    response = client.post("/v1/chat/completions", json=request_data)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["object"] == "chat.completion"
    assert "choices" in response_data
    
    choice = response_data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert "content" in choice["message"]
    assert choice["message"]["content"] is not None


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_completions_baml_path():
    """Test the /v1/chat/completions endpoint without PASSTHROUGH header (BAML path)."""
    
    # Define the tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Greet",
                "description": "Greet a person by name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the person to greet"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "GetWeather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude of the location"
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude of the location"
                        }
                    },
                    "required": ["latitude", "longitude"]
                }
            }
        }
    ]
    
    # Create the request
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Greet John and get the weather for San Francisco (37.7749, -122.4194)"
            }
        ],
        "tools": tools,
        "temperature": 0.7
    }
    
    # Make the request without PASSTHROUGH header
    response = client.post("/v1/chat/completions", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["object"] == "chat.completion"
    assert "id" in response_data
    assert "created" in response_data
    assert "model" in response_data
    assert "choices" in response_data
    assert len(response_data["choices"]) > 0
    
    # Check the first choice
    choice = response_data["choices"][0]
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    
    # Should have either tool_calls or content
    message = choice["message"]
    assert "tool_calls" in message or "content" in message


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_chat_completions_with_passthrough():
    """Test the /v1/chat/completions endpoint with PASSTHROUGH header (OpenAI forwarding)."""
    
    # Create the request
    request_data = {
        "model": "gpt-3.5-turbo",  # Using a cheaper model for testing
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello from OpenAI' and nothing else"
            }
        ],
        "temperature": 0,
        "max_tokens": 10
    }
    
    # Make the request with PASSTHROUGH header
    response = client.post(
        "/v1/chat/completions", 
        json=request_data,
        headers={"PASSTHROUGH": "true"}
    )
    
    # Verify response
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["object"] == "chat.completion"
    assert "id" in response_data
    assert "created" in response_data
    assert "model" in response_data
    assert "choices" in response_data
    assert len(response_data["choices"]) > 0
    
    # Check the response content
    choice = response_data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert "content" in choice["message"]
    assert choice["message"]["content"] is not None
    
    # Check usage is present
    assert "usage" in response_data
    assert "prompt_tokens" in response_data["usage"]
    assert "completion_tokens" in response_data["usage"]
    assert "total_tokens" in response_data["usage"]