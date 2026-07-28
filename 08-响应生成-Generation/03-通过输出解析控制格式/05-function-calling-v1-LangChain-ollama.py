from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "qwen3:14b"
LLM_TIMEOUT = 200.0

# embed_model_ollama = OllamaEmbeddings(
#     model=EMBED_MODEL_NAME,
#     base_url=OLLAMA_BASE_URL,
# )

llm_ollama = ChatOllama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=LLM_TIMEOUT
)


# 定义工具模式
class get_weather(BaseModel):
    """获取天气信息"""
    location: str = Field(..., description="城市名称")
    temperature: float = Field(..., description="温度")

# 初始化本地 Ollama 模型（确保已通过 ollama pull deepseek-r1:14b 下载）
llm = llm_ollama

# 绑定工具
llm_with_tools = llm.bind_tools([get_weather])

# 发送请求
response = llm_with_tools.invoke("请告诉我上海的天气")

# 解析输出
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"工具名称: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
else:
    print("没有工具调用")

'''
thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-all
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/05-function-calling-v1-LangChain-ollama.py
工具名称: get_weather
参数: {'location': '上海'}
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''