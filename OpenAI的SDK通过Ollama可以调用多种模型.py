from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.119:11434/v1",
    api_key="ollama"                       
)

response = client.chat.completions.create(
    model="qwen3:0.6b",   # 必须与 ollama list 中的名字完全一致
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)

print("="*50)
client = OpenAI(
    base_url="http://192.168.0.119:11434/v1",
    api_key="ollama"                       
)

response = client.chat.completions.create(
    model="deepseek-r1:1.5b",   # 必须与 ollama list 中的名字完全一致
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)


print("="*50)
from langchain_openai import ChatOpenAI
client = ChatOpenAI(model="deepseek-r1:1.5b",
                        base_url="http://192.168.0.119:11434/v1", 
                        api_key="ollama",
                        temperature=0.5,
                        timeout=200,)
response = client.invoke(input=[{"role": "user", "content": "你好"}]
)
print(response.content)

for chunk in client.stream(input=[{"role": "user", "content": "你好"}]):
    print(chunk.content, end="", flush=True)
print("\n")