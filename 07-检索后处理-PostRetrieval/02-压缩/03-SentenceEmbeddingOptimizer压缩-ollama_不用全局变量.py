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

# Settings.embed_model = embed_ollama
# Settings.llm = llm_ollama


# 加载文档
documents = SimpleDirectoryReader("90-文档-Data/山西文旅").load_data()  
index = VectorStoreIndex.from_documents(documents, embed_model=embed_ollama)
# 不使用优化的查询
print("不使用优化：")
query_engine0 = index.as_query_engine(llm=llm_ollama)
response = query_engine0.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
# 使用优化（百分比截断）
print("\n使用优化（percentile_cutoff=0.5）：")
query_engine_pcf = index.as_query_engine(llm=llm_ollama,node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5,embed_model=embed_ollama)])
response = query_engine_pcf.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
# 使用优化（阈值截断）
print("\n使用优化（threshold_cutoff=0.5）：")
query_engine_tcf = index.as_query_engine(llm=llm_ollama,node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.5,embed_model=embed_ollama)])#0.7崩溃
response = query_engine_tcf.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")


print("\n\n使用优化（threshold_cutoff=0.7）：")
optimizer_3 = SentenceEmbeddingOptimizer(threshold_cutoff=0.7,embed_model=embed_ollama)
query_engine_tcf2 = index.as_query_engine(llm=llm_ollama,node_postprocessors=[optimizer_3])

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
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-llamaindex/bin/python /home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/02-压缩/03-SentenceEmbeddingOptimizer压缩-ollama_不用 全局变量.py
/home/thbytwo/miniforge3/envs/venv-rag-llamaindex/lib/python3.10/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.3.0) or chardet (7.4.3)/charset_normalizer (3.4.1) doesn't match a supported version!
  warnings.warn(
不使用优化：
答案：山西省的主要旅游景点包括云冈石窟。云冈石窟位于大同市，依附于武周山（亦称武州山）南缘，是北魏时期开凿的重要石窟群。其中最具代表性的“昙曜五窟”（第十六至二十窟）以宏伟的佛教造像闻名，例如第十六窟的释迦牟尼立像高13.5米，第十七窟的交脚弥勒像高15.6米，展现了精湛的雕刻技艺和独特的艺术风格。这些洞窟的开凿与北魏政权对佛教的推崇密切相关，是研究中国古代宗教艺术和历史的重要遗迹。

使用优化（percentile_cutoff=0.5）：
答案：山西省的主要旅游景点包括云冈石窟，该石窟位于大同市西郊的武周山南缘，是北魏时期开凿的重要佛教艺术宝库，以昙曜五窟等早期洞窟群和精美的佛教造像闻名。此外，山西省还拥有丰富的历史文化资源，如太原作为省会城市的历史遗迹，以及与古代王朝、名人相关的文化景观。

使用优化（threshold_cutoff=0.5）：
答案：山西省的主要旅游景点包括云冈石窟。云冈石窟位于大同市，开凿于武周山南缘，是北魏时期的重要佛教艺术遗迹。其中，昙曜五窟（第十六至二十窟）是云冈石窟最早开凿的五个洞窟，以宏伟的造像和精美的雕刻闻名。例如，第十六窟的释迦牟尼像高13.5米，第十七窟的三世佛雕像中，中央的弥勒坐像高达15.6米，展现了北魏时期石窟艺术的巅峰成就。


使用优化（threshold_cutoff=0.7）：
过滤太严格，删光了所有上下文！已回退到默认检索结果。
回退答案：山西省的主要旅游景点包括云冈石窟和五台山。云冈石窟位于大同市，是著名的石窟艺术宝库，以昙曜五窟等早期开凿的洞窟和精美的佛教造像闻名。五台山作为中国佛教四大名山之一，是重要的宗教文化圣地，以其自然景观和历史悠久的寺庙群吸引游客。
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''