from datetime import datetime, timedelta
import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


from langchain_community.embeddings import OllamaEmbeddings
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.2:3b"
LLM_TIMEOUT = 200.0
embed_model_ollama = OllamaEmbeddings(
    model=EMBED_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)

"""
时效加权重排算法实现

TimeWeightedVectorStoreRetriever是一种考虑时间因素的检索器，它将文档的时效性作为重要的排序因子。

核心原理：
1. 时间衰减：文档的相关性分数会随着时间的推移而衰减
2. 访问更新：每次访问文档时更新其"最后访问时间"
3. 综合评分：结合语义相似度和时间新鲜度进行综合排序

数学模型：
- final_score = semantic_score * time_decay_factor
- time_decay_factor = exp(-decay_rate * time_since_last_access)

技术特点：
- 时效性感知：优先返回最近访问或创建的文档
- 动态更新：文档的时间权重会根据访问情况动态调整
- 衰减控制：可调节的衰减率适应不同应用需求

适用场景：
- 新闻检索：最新的新闻更重要
- 知识库维护：最近更新的文档更可靠
- 趋势分析：关注最新的数据和信息
- 实时系统：需要考虑信息时效性的应用

参数说明：
- decay_rate：衰减率，控制时间对相关性的影响程度
- k：返回的文档数量
- last_accessed_at：文档的最后访问时间
"""

print("初始化时效加权重排系统...")

# 1. 配置嵌入模型
print("配置 嵌入模型...")
embeddings_model = embed_model_ollama #OpenAIEmbeddings()
print("  模型:  Embeddings")
print("  维度: 768维向量")
print("  注意: 需要配置OPENAI_API_KEY环境变量")

# 2. 初始化FAISS向量存储
print(f"\n初始化向量存储系统...")
print("  创建FAISS索引（L2距离）...")
index = faiss.IndexFlatL2(768)  # OpenAI嵌入的维度是768
print(f"    索引类型: IndexFlatL2")
print(f"    向量维度: 768")

print("  配置文档存储...")
vectorstore = FAISS(
    embeddings_model,           # 嵌入模型
    index,                      # FAISS索引
    InMemoryDocstore({}),       # 内存文档存储
    {}                          # 索引到文档ID的映射
)
print("  向量存储系统初始化完成")

# 3. 创建时效加权检索器
print(f"\n创建时效加权检索器...")
print("  检索器配置参数:")
decay_rate = 0.5
k_value = 1
print(f"    - decay_rate: {decay_rate} (衰减率，值越大时间影响越强)")
print(f"    - k: {k_value} (返回文档数量)")
print("  衰减机制说明:")
print("    - 衰减公式: score = semantic_score * exp(-decay_rate * hours_passed)")
print("    - 衰减率0.5意味着每小时文档权重衰减约39%")

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=decay_rate,      # 衰减率：控制时间对相关性的影响强度
    k=k_value                   # 返回的文档数量
)
print("  时效加权检索器创建完成")

# 4. 准备测试文档
print(f"\n准备测试文档...")

# 第一个文档：设置为昨天访问
yesterday = datetime.now() - timedelta(days=1)
print(f"  设置文档1的访问时间为昨天: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")

print("  添加第一个文档（昨天访问）...")
doc1 = Document(
    page_content="hello world",
    metadata={"last_accessed_at": yesterday, "doc_id": "doc_1", "topic": "greeting"}
)
retriever.add_documents([doc1])
print(f"    内容: {doc1.page_content}")
print(f"    访问时间: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")

# 第二个文档：当前时间访问（默认）
current_time = datetime.now()
print(f"\n  第二个文档将使用当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

print("  添加第二个文档（当前时间）...")
doc2 = Document(
    page_content="hello foo",
    metadata={"doc_id": "doc_2", "topic": "greeting"}
)
retriever.add_documents([doc2])
print(f"    内容: {doc2.page_content}")
print("    访问时间: 当前时间（默认）")

# 5. 执行检索和时效性分析
print(f"\n执行时效加权检索...")
query = "hello world"
print(f"查询: {query}")

print(f"\n  时效加权检索过程:")
print("    1. 计算查询与每个文档的语义相似度")
print("    2. 根据文档最后访问时间计算时间衰减因子")
print("    3. 综合语义分数和时间因子得出最终分数")
print("    4. 按最终分数降序排列文档")

print(f"\n  正在执行检索...")
results = retriever.get_relevant_documents(query)

# 6. 分析和展示结果
print(f"\n时效加权检索结果分析:")
print(f"{'='*60}")
print(f"检索结果")
print(f"{'='*60}")
print(f"查询: {query}")
print(f"返回文档数: {len(results)}")

for i, doc in enumerate(results, 1):
    print(f"\n排名 {i}:")
    print(f"   文档内容: {doc.page_content}")
    print(f"   文档ID: {doc.metadata.get('doc_id', '未知')}")
    print(f"   主题: {doc.metadata.get('topic', '未知')}")
    
    # 分析时效性影响
    if 'last_accessed_at' in doc.metadata:
        access_time = doc.metadata['last_accessed_at']
        time_diff = datetime.now() - access_time
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = 1.0 / (1.0 + decay_rate * hours_passed)  # 简化的衰减计算
        
        print(f"   最后访问: {access_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   时间间隔: {time_diff}")
        print(f"   过去小时数: {hours_passed:.2f}")
        print(f"   时间衰减因子: {decay_factor:.4f}")
    else:
        print(f"   最后访问: 当前时间（刚添加）")
        print(f"   时间衰减因子: 1.0000（无衰减）")

print(f"\n结果解释:")
if len(results) > 0:
    first_doc = results[0]
    if "foo" in first_doc.page_content:
        print("  'hello foo' 排在第一位")
        print("  原因: 尽管与查询'hello world'的语义相似度可能较低，")
        print("       但由于是最近添加的文档（时间权重高），总分更高")
    else:
        print("  'hello world' 排在第一位")
        print("  原因: 语义相似度高，足以克服时间衰减的影响")

# 7. 时间模拟实验
print(f"\n时间模拟实验...")
print("  模拟几小时后的检索结果...")

# 使用mock来模拟未来时间
from langchain_core.utils import mock_now
import datetime as dt

# 模拟8小时后的检索
future_time = dt.datetime(2028, 8, 8, 12, 0)  # 模拟未来时间
print(f"  模拟时间: {future_time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"  在模拟时间下执行检索...")
with mock_now(future_time):
    future_results = retriever.get_relevant_documents(query)

print(f"\n模拟时间检索结果:")
print(f"查询: {query}")
print(f"模拟时间: {future_time.strftime('%Y-%m-%d %H:%M:%S')}")

for i, doc in enumerate(future_results, 1):
    print(f"\n排名 {i} (模拟):")
    print(f"   文档内容: {doc.page_content}")
    print(f"   文档ID: {doc.metadata.get('doc_id', '未知')}")
    
    # 在模拟时间下计算衰减
    if 'last_accessed_at' in doc.metadata:
        access_time = doc.metadata['last_accessed_at']
        time_diff = future_time - access_time
        hours_passed = time_diff.total_seconds() / 3600
        print(f"   模拟时间间隔: {time_diff}")
        print(f"   模拟过去小时数: {hours_passed:.2f}")
    else:
        # 对于当前添加的文档，计算从添加到模拟时间的间隔
        time_diff = future_time - current_time
        hours_passed = time_diff.total_seconds() / 3600
        print(f"   模拟时间间隔: {time_diff}")
        print(f"   模拟过去小时数: {hours_passed:.2f}")

print(f"\n时效加权重排总结:")
print("- 时效性感知：优先返回最近访问或创建的文档")
print("- 动态权重：文档重要性随时间动态调整")
print("- 可调控制：通过decay_rate参数控制时间影响强度")
print("- 综合评分：平衡语义相关性和时间新鲜度")
print("- 适用场景：新闻检索、知识库维护、实时系统")
print("- 参数调优：根据应用场景调整衰减率和返回数量")
print("- 注意事项：需要合理设置时间元数据")
print("- 最佳实践：结合其他检索方法形成多阶段检索管道")



'''
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/06-时效加权重排-ollama.py
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/06-时效加权重排-ollama.py:15: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model_ollama = OllamaEmbeddings(
初始化时效加权重排系统...
配置 嵌入模型...
  模型:  Embeddings
  维度: 768维向量
  注意: 需要配置OPENAI_API_KEY环境变量

初始化向量存储系统...
  创建FAISS索引（L2距离）...
    索引类型: IndexFlatL2
    向量维度: 768
  配置文档存储...
  向量存储系统初始化完成

创建时效加权检索器...
  检索器配置参数:
    - decay_rate: 0.5 (衰减率，值越大时间影响越强)
    - k: 1 (返回文档数量)
  衰减机制说明:
    - 衰减公式: score = semantic_score * exp(-decay_rate * hours_passed)
    - 衰减率0.5意味着每小时文档权重衰减约39%
  时效加权检索器创建完成

准备测试文档...
  设置文档1的访问时间为昨天: 2026-07-22 16:39:17
  添加第一个文档（昨天访问）...
    内容: hello world
    访问时间: 2026-07-22 16:39:17

  第二个文档将使用当前时间: 2026-07-23 16:39:17
  添加第二个文档（当前时间）...
    内容: hello foo
    访问时间: 当前时间（默认）

执行时效加权检索...
查询: hello world

  时效加权检索过程:
    1. 计算查询与每个文档的语义相似度
    2. 根据文档最后访问时间计算时间衰减因子
    3. 综合语义分数和时间因子得出最终分数
    4. 按最终分数降序排列文档

  正在执行检索...
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/06-时效加权重排-ollama.py:135: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
  results = retriever.get_relevant_documents(query)
/home/thbytwo/miniforge3/envs/venv-rag-langchain/lib/python3.10/site-packages/langchain/retrievers/time_weighted_retriever.py:83: UserWarning: Relevance scores must be between 0 and 1, got [(Document(id='da2a7976-eadc-4966-a843-e0361b2b61cd', metadata={'last_accessed_at': datetime.datetime(2026, 7, 22, 16, 39, 17, 662073), 'doc_id': 'doc_1', 'topic': 'greeting', 'created_at': datetime.datetime(2026, 7, 23, 16, 39, 17, 662119), 'buffer_idx': 0}, page_content='hello world'), -197.93519657331782), (Document(id='1b234cf1-d39d-475b-909e-79b23849a100', metadata={'doc_id': 'doc_2', 'topic': 'greeting', 'last_accessed_at': datetime.datetime(2026, 7, 23, 16, 39, 17, 824032), 'created_at': datetime.datetime(2026, 7, 23, 16, 39, 17, 824032), 'buffer_idx': 1}, page_content='hello foo'), -311.7440200577808)]
  docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(

时效加权检索结果分析:
============================================================
检索结果
============================================================
查询: hello world
返回文档数: 1

排名 1:
   文档内容: hello world
   文档ID: doc_1
   主题: greeting
   最后访问: 2026-07-23 16:39:17
   时间间隔: 0:00:00.000089
   过去小时数: 0.00
   时间衰减因子: 1.0000

结果解释:
  'hello world' 排在第一位
  原因: 语义相似度高，足以克服时间衰减的影响

时间模拟实验...
  模拟几小时后的检索结果...
  模拟时间: 2028-08-08 12:00:00
  在模拟时间下执行检索...
/home/thbytwo/miniforge3/envs/venv-rag-langchain/lib/python3.10/site-packages/langchain/retrievers/time_weighted_retriever.py:83: UserWarning: Relevance scores must be between 0 and 1, got [(Document(id='da2a7976-eadc-4966-a843-e0361b2b61cd', metadata={'last_accessed_at': datetime.datetime(2026, 7, 22, 16, 39, 17, 662073), 'doc_id': 'doc_1', 'topic': 'greeting', 'created_at': datetime.datetime(2026, 7, 23, 16, 39, 17, 662119), 'buffer_idx': 0}, page_content='hello world'), -197.93519657331782), (Document(id='1b234cf1-d39d-475b-909e-79b23849a100', metadata={'doc_id': 'doc_2', 'topic': 'greeting', 'last_accessed_at': datetime.datetime(2026, 7, 23, 16, 39, 17, 824032), 'created_at': datetime.datetime(2026, 7, 23, 16, 39, 17, 824032), 'buffer_idx': 1}, page_content='hello foo'), -311.7440200577808)]
  docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(

模拟时间检索结果:
查询: hello world
模拟时间: 2028-08-08 12:00:00

排名 1 (模拟):
   文档内容: hello world
   文档ID: doc_1
   模拟时间间隔: 0:00:00
   模拟过去小时数: 0.00

时效加权重排总结:
- 时效性感知：优先返回最近访问或创建的文档
- 动态权重：文档重要性随时间动态调整
- 可调控制：通过decay_rate参数控制时间影响强度
- 综合评分：平衡语义相关性和时间新鲜度
- 适用场景：新闻检索、知识库维护、实时系统
- 参数调优：根据应用场景调整衰减率和返回数量
- 注意事项：需要合理设置时间元数据
- 最佳实践：结合其他检索方法形成多阶段检索管道
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''