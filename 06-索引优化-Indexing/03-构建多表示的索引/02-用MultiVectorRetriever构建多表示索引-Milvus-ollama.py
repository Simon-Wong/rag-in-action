# ===================== 1. 依赖导入 =====================
# 标准库工具
import uuid

# ========== Ollama 部分完全保留原有写法，未修改 ==========
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
# ========================================================

# LangChain 核心组件
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 本地MHTML文档加载工具
from langchain_community.document_loaders import MHTMLLoader

# 多向量检索相关组件
from langchain.storage import InMemoryByteStore  # 存储完整原始文档
from langchain_milvus import Milvus              # 官方新版Milvus，旧版本类名仍为Milvus
from langchain.retrievers.multi_vector import MultiVectorRetriever  # 多向量检索器

# ===================== 2. 全局配置 =====================
# 2.1 Ollama 服务配置（完全保留原有参数）
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.2:3b"
LLM_TIMEOUT = 200.0

# 2.2 本地MHTML文件配置
MHTML_FILE_PATH = "/home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/03-构建多表示的索引/test.mhtml"  # 替换成你的本地mhtml文件名

# 2.3 Milvus 本地文件配置
MILVUS_DB_PATH = "./milvus_demo.db"    # 本地数据库单文件，持久化存盘
COLLECTION_NAME = "multi_vector_summaries"
VECTOR_DIM = 768

# 2.4 检索配置
ID_KEY = "doc_id"

# ===================== 3. 初始化基础组件 =====================
# ========== Ollama 初始化完全保留原有写法，未修改 ==========
# 3.1 初始化嵌入模型
embed_model_ollama = OllamaEmbeddings(
    model=EMBED_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)

# 3.2 初始化大语言模型
llm_ollama = Ollama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=LLM_TIMEOUT,
)
# ========================================================

# ===================== 4. 加载本地MHTML文档 =====================
print("\n[步骤1] 开始加载本地MHTML文档...")
loader = MHTMLLoader(MHTML_FILE_PATH)
docs = loader.load()
print(f"[步骤1] 文档加载完成，共{len(docs)}篇文档")

# ===================== 5. 生成每篇文档的摘要 =====================
print("\n[步骤2] 开始生成文档摘要...")
# 5.1 定义摘要生成的提示词模板
summary_prompt = ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")

# 5.2 构建摘要生成链
summary_chain = summary_prompt | llm_ollama | StrOutputParser()

# 5.3 提取所有文档的正文内容
doc_contents = []
for doc in docs:
    doc_contents.append(doc.page_content)

# 5.4 批量生成摘要
summaries = summary_chain.batch(doc_contents, {"max_concurrency": 1})
print(f"[步骤2] 摘要生成完成，共{len(summaries)}份摘要")

# ===================== 6. 初始化本地Milvus向量库与多向量检索器 =====================
print("\n[步骤3] 初始化本地Milvus向量库与检索器...")

vectorstore = Milvus(
    embedding_function=embed_model_ollama,
    collection_name=COLLECTION_NAME,
    connection_args={"uri": MILVUS_DB_PATH},
    vector_field="vector",
    text_field="text",
    primary_field="pk",
    auto_id=True,
    index_params={
        "index_type": "AUTOINDEX",   # 本地模式自动选择 FLAT 或 IVF_FLAT
        "metric_type": "L2",         # 距离度量，可选 L2 或 IP
        "params": {}                 # AUTOINDEX 无需额外参数
    }
)

# 6.2 创建文档存储器（存放完整原始长文档）
doc_store = InMemoryByteStore()

# 6.3 初始化多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=doc_store,
    id_key=ID_KEY,
)

# ===================== 7. 写入数据 =====================
print("\n[步骤4] 开始写入数据...")
# 7.1 为每一篇原始文档生成唯一ID
doc_ids = []
for _ in range(len(docs)):
    unique_id = str(uuid.uuid4())
    doc_ids.append(unique_id)

# 7.2 封装摘要文档，写入Milvus本地库
summary_docs = []
for i in range(len(summaries)):
    current_summary = summaries[i]
    current_doc_id = doc_ids[i]
    
    summary_doc = Document(
        page_content=current_summary,
        metadata={ID_KEY: current_doc_id}
    )
    summary_docs.append(summary_doc)

retriever.vectorstore.add_documents(summary_docs)
print(f"[步骤4] 已将{len(summary_docs)}条摘要写入本地Milvus库")

# 7.3 写入完整原始文档到字节存储器
doc_store_data = list(zip(doc_ids, docs))
retriever.docstore.mset(doc_store_data)
print(f"[步骤4] 已将{len(docs)}篇完整文档写入文档存储器")

# ===================== 8. 测试检索效果 =====================
print("\n" + "="*50)
query = "Memory in agents"
print(f"检索问题：{query}")
print("开始检索...")

retrieved_docs = retriever.get_relevant_documents(query, n_results=1)

print("\n检索完成，返回的完整文档片段：")
print("-"*30)
for doc in retrieved_docs:
    print(doc.page_content[:500] + "...")



'''
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/03-构建多表示的索引/02-用MultiVectorRetriever构建多表示索引-Milvus-ollama.py
/home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/03-构建多表示的索引/02-用MultiVectorRetriever构建多表示索引-Milvus-ollama.py:44: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model_ollama = OllamaEmbeddings(
/home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/03-构建多表示的索引/02-用MultiVectorRetriever构建多表示索引-Milvus-ollama.py:50: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm_ollama = Ollama(

[步骤1] 开始加载本地MHTML文档...
[步骤1] 文档加载完成，共1篇文档

[步骤2] 开始生成文档摘要...
[步骤2] 摘要生成完成，共1份摘要

[步骤3] 初始化本地Milvus向量库与检索器...

[步骤4] 开始写入数据...
[步骤4] 已将1条摘要写入本地Milvus库
[步骤4] 已将1篇完整文档写入文档存储器

==================================================
检索问题：Memory in agents
开始检索...
/home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/03-构建多表示的索引/02-用MultiVectorRetriever构建多表示索引-Milvus-ollama.py:142: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
  retrieved_docs = retriever.get_relevant_documents(query, n_results=1)

检索完成，返回的完整文档片段：
------------------------------




LLM Powered Autonomous Agents | Lil'Log



























Lil'Log

















|






Posts




Archive




Search




Tags




FAQ









      LLM Powered Autonomous Agents
    
Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng


 


Table of Contents



Agent System Overview

Component One: Planning

Task Decomposition

Self-Reflection


Component Two: Memory

Types of Memory

Maximum Inner Product Search (MIPS)


Component Three: Tool Use

Cas...
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''