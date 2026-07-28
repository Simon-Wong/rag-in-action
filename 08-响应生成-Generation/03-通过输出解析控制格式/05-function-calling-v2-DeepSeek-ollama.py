from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()


llm_ollama = OpenAI(
    base_url="http://192.168.0.119:11434/v1",
    api_key="ollama"                       
)


def send_messages(messages):
    response = client.chat.completions.create(
        model="qwen3:14b",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

client = llm_ollama

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of an location, the user shoud supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
print(f"User>\t {messages[0]['content']}")

message = send_messages(messages)
print("\nModel's Function Call Response:")
print(f"Content: {message.content}")
print(f"Tool Calls: {json.dumps(message.tool_calls[0].function.model_dump(), indent=2)}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})#这里伪造一个数据
message = send_messages(messages)
print(f"\nModel's Final Response: {message.content}")

'''
thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-all
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/05-function-calling-v2-DeepSeek-ollama.py
User>    How's the weather in Hangzhou?

Model's Function Call Response:
Content: 
Tool Calls: {
  "arguments": "{\"location\":\"Hangzhou\"}",
  "name": "get_weather"
}

Model's Final Response: The weather in Hangzhou is currently **24℃**. Let me know if you'd like additional details! 😊

'''