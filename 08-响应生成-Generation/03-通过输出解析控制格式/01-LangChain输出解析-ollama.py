from langchain_core.output_parsers import JsonOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "deepseek-r1:14b"
LLM_TIMEOUT = 200.0

# embed_model_ollama = OllamaEmbeddings(
#     model=EMBED_MODEL_NAME,
#     base_url=OLLAMA_BASE_URL,
# )

llm_ollama = Ollama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=LLM_TIMEOUT,
)

# 定义输出格式
parser = JsonOutputParser()
prompt = PromptTemplate.from_template("请返回JSON格式的用户信息：{query}，注意不要擅自添加其他信息")
# 调用大模型并解析
llm = llm_ollama # ChatDeepSeek(model="deepseek-chat")
output = llm(prompt.format(query="用户ID 123"))
# 从 AIMessage 中提取内容
parsed_output = parser.parse(output)
print(parsed_output)

'''
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/01-LangChain输出解析.py
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/01-LangChain输出解析.py:17: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm_ollama = Ollama(
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/01-LangChain输出解析.py:28: LangChainDeprecationWarning: The method `BaseLLM.__call__` was deprecated in langchain-core 0.1.7 and will be removed in 1.0. Use :meth:`~invoke` instead.
  output = llm(prompt.format(query="用户ID 123"))
{'ID': 123}
'''