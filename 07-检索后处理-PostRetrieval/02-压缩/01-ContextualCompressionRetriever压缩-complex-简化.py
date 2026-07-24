# 导入所需的库
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, LLMListwiseRerank, EmbeddingsFilter

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "qwen3:14b"
LLM_TIMEOUT = 200.0

embed_model_ollama = OllamaEmbeddings(
    model=EMBED_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)

llm_ollama = ChatOllama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=LLM_TIMEOUT
)


documents = [
    Document(
        page_content="五台山是中国四大佛教名山之一，以文殊菩萨道场闻名。",
        metadata={"source": "山西旅游指南"}
    ),
    Document(
        page_content="云冈石窟是中国三大石窟之一，以精美的佛教雕塑著称。",
        metadata={"source": "山西旅游指南"}
    ),
    Document(
        page_content="山西平遥古城是中国保存最完整的古代县城之一，被列为世界文化遗产。",
        metadata={"source": "山西旅游指南"}
    )
]

# 创建BM25检索器
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # 设置返回前3个结果

# 创建4种压缩器
# 1. EmbeddingsFilter：向量相似度过滤
embed_filter = EmbeddingsFilter(embeddings=embed_model_ollama, similarity_threshold=0.5)

# 2. LLMChainFilter：基于大模型过滤文档（保留或丢弃整篇）
# 创建一个强制的提示词模板，只允许模型输出 YES 或 NO
filter_prompt = PromptTemplate.from_template(
    "判断下面的文档是否与用户的查询相关。请只回答 'YES' 或 'NO'，不要包含任何其他文字、标点符号或解释。\n\n查询: {question}\n文档: {context}"
)
llm_filter = LLMChainFilter.from_llm(llm_ollama, prompt=filter_prompt)

# 3. LLMListwiseRerank：基于大模型重排和选出Top N个
rerank = LLMListwiseRerank.from_llm(llm_ollama, top_n=2) # 只取重排后最相关的2个

# 4. LLMChainExtractor：基于大模型提取核心内容
extractor = LLMChainExtractor.from_llm(llm_ollama)


# 将它们组合成管道，按顺序执行
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[embed_filter,  extractor]
)

# 创建ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)


# 执行查询、重排和压缩
query = "山西有哪些著名的旅游景点？"
compressed_docs = compression_retriever.invoke(query)

# 输出压缩结果
print(f"查询：{query}\n")
print("经过4种压缩策略组合重排后的结果：")
for i, doc in enumerate(compressed_docs, 1):
    print(f"{i}. {doc.page_content}")

# # --- 调试代码开始 ---
# # 1. 分别测试每一步
# p1 = DocumentCompressorPipeline(transformers=[embed_filter])
# r1 = ContextualCompressionRetriever(base_compressor=p1, base_retriever=retriever)
# print(f"第1步【embed_filter】后，剩余文档数: {len(r1.invoke(query))}")

# p2 = DocumentCompressorPipeline(transformers=[embed_filter, llm_filter])
# r2 = ContextualCompressionRetriever(base_compressor=p2, base_retriever=retriever)
# print(f"第2步【加上 llm_filter】后，剩余文档数: {len(r2.invoke(query))}")

# p3 = DocumentCompressorPipeline(transformers=[embed_filter, llm_filter, rerank])
# r3 = ContextualCompressionRetriever(base_compressor=p3, base_retriever=retriever)
# print(f"第3步【加上 rerank】后，剩余文档数: {len(r3.invoke(query))}")

# p4 = DocumentCompressorPipeline(transformers=[embed_filter, llm_filter, rerank, extractor])
# r4 = ContextualCompressionRetriever(base_compressor=p4, base_retriever=retriever)
# print(f"第4步【最终结果】剩余文档数: {len(r4.invoke(query))}")
# # --- 调试代码结束 ---