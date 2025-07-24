# OpenAI BAML Adapter

This is a fastapi server serving the openai completions API.

It is useful when running benchmarks to compare BAML's performance
against OpenAI-style native tool calling.

The `/v1/chat/completions` handler looks for JSONSchema tools
and converts them to BAML types. It then makes a BAML function
call to GPT-4o-mini, parses the response with BAML's SAP parser,
and converts the response into the OpenAI format.

## Installing

```bash
uv build
```

## Running

```
uv run uvicorn openai_baml_adapter.api.main:app --reload --host 0.0.0.0 --port 8000
```

Note that you will need OPENAI_API_KEY set in your environment.

## Testing

```
uv run pytest
```

Note that you will need OPENAI_API_KEY set in your environment.


## Manual testing

Run the server in one terminal as described above. In another terminal,
send a curl request with tools:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-4",
      "messages": [
        {
          "role": "user",
          "content": "Greet John and get the weather for San Francisco (37.7749, -122.4194)"
        }
      ],
      "tools": [
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
      ],
      "temperature": 0.7
    }'
```