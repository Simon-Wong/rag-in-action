from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.deepseek import DeepSeek 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Embedding
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# 1. 配置 Embedding 模型
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",   # 确保与 Ollama 下载的模型名完全一致
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
)

# 2. 配置 LLM 模型
llm = Ollama(
    model="llama3.2:3b",                # 确保与 Ollama 下载的模型名完全一致qwen3.5:9b
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
    request_timeout=120.0,           # (可选) 设置请求超时，单位秒
)

# 配置全局设置
Settings.llm = llm
Settings.embed_model = embed_model
# 准备游戏知识文本
game_knowledge = """《灭神纪∙猢狲》的战斗系统设计精妙绝伦。玩家可以在战斗中自由切换多种战斗形态，每种形态都有其独特优势。金刚形态下……魔佛形态则……"""
# 创建Document对象
documents = [Document(text=game_knowledge)]
# 创建分层节点解析器并处理文档
# 使用HierarchicalNodeParser创建文本层次结构
# chunk_sizes表示不同层级的文本块大小
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[256, 128, 64]  # 从根节点到叶子节点的块大小
)
nodes = node_parser.get_nodes_from_documents(documents)
# 获取叶子节点（最小粒度的文本块）和根节点
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)
# 构建存储和索引
# 创建文档存储并添加所有节点
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
# 创建存储上下文
storage_context = StorageContext.from_defaults(docstore=docstore)
# 为叶子节点创建向量索引
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)
# 创建基础检索器和自动合并检索器
base_retriever = base_index.as_retriever(similarity_top_k=6)
auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True  # 显示合并过程
)
# 准备测试问题
test_questions = [
    # "游戏中金刚形态和魔佛形态有什么区别？",
    "金箍棒在不同形态下有什么特点？",
    # "游戏的难度设计是怎样的？"
]
print("=== 自动合并检索器的结果 ===")
for question in test_questions:
    print(f"\n问题：{question}")
    # 使用自动合并检索器检索
    merge_nodes = auto_merging_retriever.retrieve(question)
    print(f"检索到 {len(merge_nodes)} 个合并后的节点：")
    for node in merge_nodes:
        print(f"\n相似度：{node.score}")
        print(f"内容：{node.node.text}")
        print("-" * 50)
print("\n=== 基础检索器的结果（对比）===")
for question in test_questions:
    print(f"\n问题：{question}")
    # 使用基础检索器检索
    base_nodes = base_retriever.retrieve(question)
    print(f"检索到 {len(base_nodes)} 个基础节点：")
    for node in base_nodes:
        print(f"\n相似度：{node.score}")
        print(f"内容：{node.node.text}")
        print("-" * 50)

'''
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-llamaindex/bin/python /home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/02-构建有层次的 索引/05-分层合并的示例-ollama.py
/home/thbytwo/miniforge3/envs/venv-rag-llamaindex/lib/python3.10/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.3.0) or chardet (7.4.3)/charset_normalizer (3.4.1) doesn't match a supported version!
  warnings.warn(
=== 自动合并检索器的结果 ===

问题：金箍棒在不同形态下有什么特点？
> Merging 2 nodes into parent node.
> Parent node id: c4769ad2-1e51-4c62-b1b7-ba0483eacea2.
> Parent node text: 《灭神纪∙猢狲》的战斗系统设计精妙绝伦。玩家可以在战斗中自由切换多种战斗形态，每种形态都有其独特优势。金刚形态下……魔佛形态则……

> Merging 1 nodes into parent node.
> Parent node id: 939fedb1-4549-4439-aefb-4922e55948e4.
> Parent node text: 《灭神纪∙猢狲》的战斗系统设计精妙绝伦。玩家可以在战斗中自由切换多种战斗形态，每种形态都有其独特优势。金刚形态下……魔佛形态则……

检索到 1 个合并后的节点：

相似度：0.7208339059335035
内容：《灭神纪∙猢狲》的战斗系统设计精妙绝伦。玩家可以在战斗中自由切换多种战斗形态，每种形态都有其独特优势。金刚形态下……魔佛形态则……
--------------------------------------------------

=== 基础检索器的结果（对比）===

问题：金箍棒在不同形态下有什么特点？
检索到 2 个基础节点：

相似度：0.8021960592350466
内容：玩家可以在战斗中自由切换多种战斗形态，每种形态都有其独特优势。金刚形态下……魔佛形态则……
--------------------------------------------------

相似度：0.6394717526319604
内容：《灭神纪∙猢狲》的战斗系统设计精妙绝伦。
--------------------------------------------------
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ 
'''