# ===================== 依赖库导入 =====================
# numpy：用于向量运算、相似度计算
import numpy as np
# KMeans：用于文本嵌入的语义聚类，实现相似文本自动分组
from sklearn.cluster import KMeans
# langchain_community 社区版 Ollama 组件，与你的环境完全兼容
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
# PromptTemplate：用于构造标准化的提示词模板
from langchain_core.prompts import PromptTemplate

# ===================== 1. 基础组件初始化（全局统一配置） =====================
# 嵌入模型：用于将文本转换为向量，实现语义相似度匹配
# 统一使用新的 Ollama 服务地址，模型保持专用嵌入模型
embed_model = OllamaEmbeddings(
    model="nomic-embed-text",          # 专用文本嵌入模型，负责生成向量
    base_url="http://192.168.0.119:11434",  # 统一替换为你的新 Ollama 服务地址
)

# 大语言模型：用于生成聚类摘要，提升上层节点的语义概括能力
# 替换为 llama3.2:3b，模型能力更强，摘要质量更高，检索定位更准确
llm = Ollama(
    model="llama3.2:3b",               # 替换为更强的生成模型，替代原 deepseek-r1:1.5b
    base_url="http://192.168.0.119:11434",  # 统一服务地址
    timeout=120.0,                     # 请求超时时间，避免大模型生成过久导致报错
    temperature=0.3,                   # 生成温度，越低摘要越稳定、越少发散
)

# 摘要生成提示词模板：约束大模型的摘要生成格式和要求
SUMMARY_PROMPT = PromptTemplate.from_template("""
请将以下多段文本浓缩成一段简洁的摘要，保留核心主题和关键信息，去除冗余细节。
要求：不超过100字，语义连贯，不要额外解释。
文本内容：
{texts}
摘要：
""")
# 构建摘要生成链：提示词 → 大模型 → 输出结果
summary_chain = SUMMARY_PROMPT | llm

# ===================== 2. 测试文档（RAPTOR 树的叶子层） =====================
# 叶子节点：最底层的原始文本分块，对应书中的「原始文档分块」
# 共9个分块，天然分为3个语义主题：花果山、水帘洞、东海龙宫，用于验证聚类效果
leaf_chunks = [
    # 花果山主题 3个分块
    "花果山位于东胜神洲傲来国境内，是一块灵气汇聚的仙山，相传是开天辟地时就存在的灵脉。",
    "花果山上终年不谢的奇花异草遍布，山泉瀑布四季长流，还有茂密的千年古树森林，生态极佳。",
    "花果山里有仙果园种植灵果、平坦的练功场、猴族休憩区三个特殊区域，各有不同功能。",
    # 水帘洞主题 3个分块
    "水帘洞入口是一道高30丈的天然瀑布，隐藏在花果山之巅，普通人很难发现洞口位置。",
    "水帘洞内部是错综复杂的洞穴系统，分为修炼大厅、藏宝室、议事厅三个核心功能区。",
    "水帘洞的藏宝室有天然防护阵法加持，普通妖法无法攻破，专门存放法宝和丹药。",
    # 东海龙宫主题 3个分块
    "东海龙宫是建在东海海底的宏伟宫殿群，整体用珊瑚和夜明珠搭建，占地数十里。",
    "龙宫的龙王宝库储存着无数珍宝，包括夜明珠、定海神针等上古神器，是龙宫核心重地。",
    "龙宫兵器库收藏了各式水系法器和神兵利器，是水族将领领取装备的地方。",
]

# ===================== 3. 基础工具函数 =====================
def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    批量生成文本的嵌入向量
    :param texts: 输入的文本列表
    :return: 二维numpy数组，每行对应一段文本的向量
    """
    # 调用嵌入模型批量生成向量，转换为numpy数组方便后续运算
    embeds = embed_model.embed_documents(texts)
    return np.array(embeds)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> list[list[int]]:
    """
    对文本向量做KMeans语义聚类，把语义相近的文本归为同一类
    :param embeddings: 文本向量数组
    :param n_clusters: 目标聚类数量
    :return: 列表，每个元素是一个聚类对应的文本索引列表
    """
    # 边界处理：聚类数不能超过文本总数，否则报错
    n_clusters = min(n_clusters, len(embeddings))
    # 文本数≤1时无需聚类，直接返回全部索引
    if n_clusters <= 1:
        return [list(range(len(embeddings)))]
    
    # 初始化KMeans聚类器，固定随机种子保证结果可复现
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # 执行聚类，返回每个向量对应的聚类标签
    labels = kmeans.fit_predict(embeddings)
    
    # 按标签分组，整理为「聚类ID → 对应文本索引列表」的结构
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    # 只返回索引列表的集合，用于后续取对应文本生成摘要
    return list(clusters.values())


def generate_cluster_summary(cluster_texts: list[str]) -> str:
    """
    为一个聚类内的所有文本生成统一的摘要，作为上一层的节点内容
    :param cluster_texts: 同一个聚类内的所有文本片段
    :return: 浓缩后的摘要文本
    """
    # 把聚类内的多段文本拼接成一个整体
    combined = "\n".join(cluster_texts)
    # 调用摘要链生成概括文本，去除首尾空白
    result = summary_chain.invoke({"texts": combined})
    return result.strip()

# ===================== 4. 核心函数：自底向上递归构建 RAPTOR 多层索引树 =====================
def build_raptor_index(
    current_chunks: list[str],
    current_level: int = 0,
    max_levels: int = 3,
    cluster_ratio: float = 0.4
) -> dict:
    """
    递归构建RAPTOR多层树状索引，对应书中「嵌入→聚类→摘要→递归向上」的核心流程
    构建方向：从最底层叶子节点开始，逐层向上聚类生成摘要，直到达到最大层数
    :param current_chunks: 当前层级的所有文本块
    :param current_level: 当前层级编号，0代表最底层叶子层
    :param max_levels: 最大递归层数，即索引树的总高度
    :param cluster_ratio: 聚类压缩比例，上层聚类数 = 当前层块数 × 该比例
    :return: 嵌套字典结构，包含当前层信息 + 下一层的完整结构
    """
    # 步骤1：为当前层所有文本块生成嵌入向量
    current_embeds = get_embeddings(current_chunks)
    
    # 步骤2：递归终止条件（对应书中的递归终止逻辑）
    # 达到最大层数 或 当前文本块太少无法继续聚类，终止递归，返回当前层结构
    if current_level >= max_levels or len(current_chunks) <= 2:
        return {
            "level": current_level,       # 当前层级编号
            "chunks": current_chunks,     # 当前层的所有文本内容
            "embeddings": current_embeds, # 当前层文本对应的向量
            "child_clusters": None,       # 叶子层没有下层聚类，置空
            "next_level": None            # 没有上一层，置空（递归终点）
        }
    
    # 步骤3：计算当前层的聚类数量，执行语义聚类
    # 聚类数最少为2，保证每一层都有信息压缩
    n_clusters = max(2, int(len(current_chunks) * cluster_ratio))
    # 对当前层向量做聚类，得到每个聚类对应的文本索引
    cluster_indices = cluster_embeddings(current_embeds, n_clusters)
    
    # 步骤4：为每个聚类生成摘要，作为上一层的输入文本块
    upper_chunks = []
    for indices in cluster_indices:
        # 取出当前聚类对应的所有原始文本
        cluster_texts = [current_chunks[i] for i in indices]
        # 生成摘要，作为上一层的一个节点
        summary = generate_cluster_summary(cluster_texts)
        upper_chunks.append(summary)
    
    # 打印构建日志，方便观察每一层的压缩效果
    print(f"[第{current_level}层] 共{len(current_chunks)}个文本块，聚为{len(upper_chunks)}个摘要节点，进入上一层")
    
    # 步骤5：递归调用，构建更上一层的索引
    # 把当前层生成的摘要作为输入，层级+1，继续向上聚类
    upper_level = build_raptor_index(
        current_chunks=upper_chunks,
        current_level=current_level + 1,
        max_levels=max_levels,
        cluster_ratio=cluster_ratio
    )
    
    # 步骤6：返回当前层完整结构，包含与上层、下层的关联信息
    return {
        "level": current_level,               # 当前层级编号
        "chunks": current_chunks,             # 当前层所有文本内容
        "embeddings": current_embeds,         # 当前层文本向量
        "child_clusters": cluster_indices,    # 当前层每个聚类对应的下层文本索引（父子节点关联）
        "next_level": upper_level             # 上一层的完整结构（嵌套递归）
    }

# ===================== 5. 检索函数：自顶向下分层检索 =====================
def raptor_retrieve(raptor_index: dict, query: str, top_k: int = 2) -> list[str]:
    """
    RAPTOR 分层检索逻辑：从最顶层摘要开始，逐层向下定位最相关的聚类，最终命中叶子层原始细节
    对应书中「从宏观到微观的信息导航」特性
    :param raptor_index: 构建好的RAPTOR索引树
    :param query: 用户查询问题
    :param top_k: 返回最相关的叶子层文本数量
    :return: 最相关的原始文本块列表
    """
    # 生成查询问题的向量，用于相似度计算
    query_embed = np.array(embed_model.embed_query(query))
    
    # 步骤1：先定位到索引树的最顶层（根节点层）
    current_level = raptor_index
    while current_level["next_level"] is not None:
        current_level = current_level["next_level"]
    
    # 步骤2：自顶向下逐层匹配，找到每一层最相关的聚类
    print(f"[第{current_level['level']}层（顶层总览）] 开始主题匹配")
    # 只要当前层还有下层聚类，就继续向下定位
    while current_level["child_clusters"] is not None:
        # 计算查询与当前层所有摘要节点的语义相似度（内积计算）
        embeds = current_level["embeddings"]
        similarities = np.dot(embeds, query_embed)
        # 找到相似度最高的1个聚类索引
        best_idx = int(np.argmax(similarities))
        # 打印当前层命中的摘要，观察检索路径
        print(f"  命中最相关摘要：{current_level['chunks'][best_idx][:60]}...")
        # 进入下一层，继续精准匹配
        current_level = current_level["next_level"]
    
    # 步骤3：到达最底层叶子层，计算与所有原始块的相似度，返回Top-K结果
    leaf_embeds = current_level["embeddings"]
    leaf_sims = np.dot(leaf_embeds, query_embed)
    # 按相似度从高到低排序，取前top_k个
    top_indices = np.argsort(leaf_sims)[::-1][:top_k]
    # 返回对应的原始文本
    return [current_level["chunks"][i] for i in top_indices]

# ===================== 运行示例 =====================
if __name__ == "__main__":
    # 第一部分：构建RAPTOR多层索引
    print("=== 开始构建RAPTOR多层索引 ===")
    # 构建2层摘要+1层叶子，总共3层结构；可修改max_levels调整树的深度
    raptor_tree = build_raptor_index(leaf_chunks, max_levels=2)
    print("=== 索引构建完成 ===\n")
    
    # 打印整棵树的各层内容，直观查看分层效果
    print("\n=== RAPTOR 索引树各层内容概览 ===")
    level = raptor_tree
    while level is not None:
        print(f"\n第 {level['level']} 层（共{len(level['chunks'])}个节点）：")
        for i, chunk in enumerate(level["chunks"]):
            # 只打印前60字，避免输出过长
            print(f"  [{i}] {chunk[:60]}...")
        # 移动到上一层
        level = level["next_level"]
    
    # 第二部分：测试分层检索效果
    print("\n" + "="*50)
    query = "水帘洞里存放宝物的地方有什么防护？"
    print(f"用户查询：{query}")
    print("--- 分层检索路径 ---")
    results = raptor_retrieve(raptor_tree, query, top_k=2)
    print("--- 最终检索结果 ---")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")

'''
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/02-构建有层次的索引/06-RAPTOR-ollama.py
/home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/02-构建有层次的索引/06-RAPTOR-ollama.py:15: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model = OllamaEmbeddings(
/home/thbytwo/testGit/rag-in-action/06-索引优化-Indexing/02-构建有层次的索引/06-RAPTOR-ollama.py:22: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm = Ollama(
=== 开始构建RAPTOR多层索引 ===
[第0层] 共9个文本块，聚为3个摘要节点，进入上一层
[第1层] 共3个文本块，聚为2个摘要节点，进入上一层
=== 索引构建完成 ===


=== RAPTOR 索引树各层内容概览 ===

第 0 层（共9个节点）：
  [0] 花果山位于东胜神洲傲来国境内，是一块灵气汇聚的仙山，相传是开天辟地时就存在的灵脉。...
  [1] 花果山上终年不谢的奇花异草遍布，山泉瀑布四季长流，还有茂密的千年古树森林，生态极佳。...
  [2] 花果山里有仙果园种植灵果、平坦的练功场、猴族休憩区三个特殊区域，各有不同功能。...
  [3] 水帘洞入口是一道高30丈的天然瀑布，隐藏在花果山之巅，普通人很难发现洞口位置。...
  [4] 水帘洞内部是错综复杂的洞穴系统，分为修炼大厅、藏宝室、议事厅三个核心功能区。...
  [5] 水帘洞的藏宝室有天然防护阵法加持，普通妖法无法攻破，专门存放法宝和丹药。...
  [6] 东海龙宫是建在东海海底的宏伟宫殿群，整体用珊瑚和夜明珠搭建，占地数十里。...
  [7] 龙宫的龙王宝库储存着无数珍宝，包括夜明珠、定海神针等上古神器，是龙宫核心重地。...
  [8] 龙宫兵器库收藏了各式水系法器和神兵利器，是水族将领领取装备的地方。...

第 1 层（共3个节点）：
  [0] 花果山是东胜神洲傲来国境内的一块灵气汇聚的仙山。该地拥有终年不谢的奇花异草、茂密的千年古树森林和生态极佳的环境。花果山设...
  [1] 水帘洞内部设有三大核心功能区：修炼大厅、藏宝室和议事厅。藏宝室内拥有天然防护阵法，加持使普通妖法无法攻破，存放着法宝和丹...
  [2] 东海龙宫是位于东海海底的一座宏伟宫殿群，主体结构由珊瑚和夜明珠搭建而成。宫内的龙王宝库储存着无数珍贵的神器，如夜明珠和定...

第 2 层（共2个节点）：
  [0] 花果山是东胜神洲的一块特殊地带，拥有终年不谢的奇花异草、茂密的千年古树森林和生态极佳的环境。该地设有三个区域：仙果园种植...
  [1] 东海龙宫是一个宏伟的宫殿群，位于东海海底。其内部设有三大核心功能区：修炼大厅、藏宝室和议事厅。藏宝室内拥有天然防护阵法，...

==================================================
用户查询：水帘洞里存放宝物的地方有什么防护？
--- 分层检索路径 ---
[第2层（顶层总览）] 开始主题匹配
--- 最终检索结果 ---
1. 花果山是东胜神洲的一块特殊地带，拥有终年不谢的奇花异草、茂密的千年古树森林和生态极佳的环境。该地设有三个区域：仙果园种植灵果、练功场和猴族休憩区。水帘洞入口是一道高30丈的天然瀑布，隐藏在山巅，普通人很难发现。
2. 东海龙宫是一个宏伟的宫殿群，位于东海海底。其内部设有三大核心功能区：修炼大厅、藏宝室和议事厅。藏宝室内拥有天然防护阵法，加持使普通妖法无法攻破，存放着法宝和丹药。
'''