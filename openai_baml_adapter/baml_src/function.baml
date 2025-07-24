class Response {
  @@dynamic
}

function BamlFunction(prompt: string) -> Response {
  client GPT4oMini
  prompt #"
    {{ prompt }}

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