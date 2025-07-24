#!/usr/bin/env python3
"""Debug script to test the BAML path and show full stack trace."""

import requests
import json
import traceback

def test_baml_path():
    url = "http://localhost:8000/v1/chat/completions"
    
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
    
    print("Sending request to:", url)
    print("Request data:", json.dumps(request_data, indent=2))
    
    try:
        response = requests.post(url, json=request_data)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # If it's a 500 error, the server logs should have the full traceback
        if response.status_code == 500:
            print("\n⚠️  Check the server logs for the full stack trace!")
            
    except Exception as e:
        print(f"\nError making request: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_baml_path()