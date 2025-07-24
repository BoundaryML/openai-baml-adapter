class Response {
  @@dynamic
}

class Message {
  role string
  content string
}

function BamlFunction(messages: Message[]) -> Response {
  client GPT4oMini
  prompt #"
    {% for message in messages %}
    {{ _.role(message.role) }}
    {{ message.content }}
    {% endfor %}

    {{ _.role("system") }}
    {{ctx.output_format}}
  "#
}

test Test {
  functions [BamlFunction]
  type_builder {
    class Greet {
      greeting string
    }
    class Depart {
      departure_time string
      message string
    }
    dynamic class Response {
      tool_call Greet | Depart
    }
  }

  args {
    prompt #"Greet me? My name is Greg"#
  }
}