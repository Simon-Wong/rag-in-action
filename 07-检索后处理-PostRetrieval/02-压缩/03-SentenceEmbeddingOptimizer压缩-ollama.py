from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer



from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

embed_ollama = OllamaEmbedding(
    model_name="nomic-embed-text",   # 确保与 Ollama 下载的模型名完全一致
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
)

llm_ollama = Ollama(
    model="qwen3:14b",                # 确保与 Ollama 下载的模型名完全一致qwen3.5:9b
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
    request_timeout=120.0,           # (可选) 设置请求超时，单位秒
)

Settings.embed_model = embed_ollama
Settings.llm = llm_ollama


# 加载文档
documents = SimpleDirectoryReader("90-文档-Data/山西文旅").load_data()  
index = VectorStoreIndex.from_documents(documents)
# 不使用优化的查询
print("不使用优化：")
query_engine0 = index.as_query_engine()
response = query_engine0.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
# 使用优化（百分比截断）
print("\n使用优化（percentile_cutoff=0.5）：")
query_engine_pcf = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)])
response = query_engine_pcf.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
# 使用优化（阈值截断）
print("\n使用优化（threshold_cutoff=0.5）：")
query_engine_tcf = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.5)])#0.7崩溃
response = query_engine_tcf.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")


print("\n\n使用优化（threshold_cutoff=0.7）：")
optimizer_3 = SentenceEmbeddingOptimizer(threshold_cutoff=0.7)
query_engine_tcf2 = index.as_query_engine(node_postprocessors=[optimizer_3])

try:
    response = query_engine_tcf2.query("山西省的主要旅游景点有哪些？")
    print(f"答案：{response}")
except ValueError as e:
    if "returned zero sentences" in str(e):
        print(f"过滤太严格，删光了所有上下文！已回退到默认检索结果。")
        # 回退到没有优化器的查询
        response = query_engine0.query("山西省的主要旅游景点有哪些？")
        print(f"回退答案：{response}")
    else:
        raise e
    

'''
thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-llamaindex
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-llamaindex/bin/python /home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/02-压缩/03-SentenceEmbeddingOptimizer压缩-ollama.py
/home/thbytwo/miniforge3/envs/venv-rag-llamaindex/lib/python3.10/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.3.0) or chardet (7.4.3)/charset_normalizer (3.4.1) doesn't match a supported version!
  warnings.warn(
不使用优化：
答案：山西省的主要旅游景点包括云冈石窟和五台山。云冈石窟位于大同市，以昙曜五窟等早期洞窟群闻名，展现了北魏时期佛教艺术的辉煌成就。五台山则以佛教圣地著称，拥有众多寺庙和自然景观，是华北地区的重要宗教与文化遗址。

使用优化（percentile_cutoff=0.5）：
答案：山西省的主要旅游景点包括云冈石窟。云冈石窟位于大同市，依山开凿，以昙曜五窟等早期洞窟闻名，其中第十六至二十窟为最早开凿的五个洞窟，主像高大，雕刻精美，展现了北魏时期的佛教艺术成就。此外，山西的自然与人文景观丰富，但具体其他景点名称未在提供的资料中提及。

使用优化（threshold_cutoff=0.5）：
答案：山西省的主要旅游景点包括云冈石窟。云冈石窟位于大同市西山的武周山南缘，是北魏时期开凿的重要佛教石窟艺术遗址，其中昙曜五窟（第十六至二十窟）是最早开凿的洞窟群，以宏伟的佛像雕刻和精美的壁画闻名。此外，山西省还拥有丰富的历史文化遗产和自然景观，但具体其他景点名称未在提供的材料中提及。


使用优化（threshold_cutoff=0.7）：
过滤太严格，删光了所有上下文！已回退到默认检索结果。
回退答案：山西省的主要旅游景点包括云冈石窟和五台山。云冈石窟位于大同市，以其独特的石窟艺术和历史价值著称，特别是武州山南缘的昙曜五窟，展现了北魏时期的佛教造像艺术。五台山则以佛教圣地和自然景观闻名，是华北地区的重要文化和宗教中心。
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''