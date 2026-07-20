# ===================== 依赖库导入 =====================
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# Milvus 向量数据库相关依赖
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# LangChain + Ollama 组件，与你的环境完全对齐
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ===================== 全局配置项（统一修改入口） =====================
# 1. Ollama 服务配置（统一使用你的新地址与模型）
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"  # 嵌入模型，输出维度768维
LLM_MODEL_NAME = "llama3.2:3b"         # 生成摘要的大模型

# 2. Milvus 数据库配置
MILVUS_HOST = "localhost"  # Milvus 服务地址，本地部署填localhost
MILVUS_PORT = "19530"      # Milvus 默认服务端口
COLLECTION_PREFIX = "raptor_level_"  # RAPTOR每层集合的命名前缀
VECTOR_DIM = 768           # 嵌入向量维度，必须和nomic-embed-text的输出维度一致

# 3. RAPTOR 索引参数
MAX_LEVELS = 2             # 最大层数（0为叶子层，共3层结构）
CLUSTER_RATIO = 0.4        # 每层聚类压缩比例

# ===================== 基础组件初始化 =====================
# 嵌入模型：文本转向量
embed_model = OllamaEmbeddings(
    model=EMBED_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)

# 大模型：生成聚类摘要
llm = Ollama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=120.0,
    temperature=0.3,
)

# 摘要生成提示词模板
SUMMARY_PROMPT = PromptTemplate.from_template("""
请将以下多段文本浓缩成一段简洁的摘要，保留核心主题和关键信息，去除冗余细节。
要求：不超过100字，语义连贯，不要额外解释。
文本内容：
{texts}
摘要：
""")
summary_chain = SUMMARY_PROMPT | llm

# ===================== Milvus 持久化工具函数 =====================
def connect_milvus():
    """连接 Milvus 数据库，全局只需连接一次"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print("[Milvus] 连接成功")
    except Exception as e:
        print(f"[Milvus] 连接失败：{str(e)}")
        raise


def create_level_collection(level: int):
    """
    为指定层级创建 Milvus 集合（表）
    每个层级对应一个独立集合，方便分层检索
    :param level: 层级编号，0为叶子层
    """
    collection_name = f"{COLLECTION_PREFIX}{level}"
    
    # 如果集合已存在，先删除旧集合（重建索引时用）
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"[Milvus] 已删除旧集合：{collection_name}")
    
    # 定义集合字段结构
    fields = [
        # 主键：自增ID
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 文本内容：存储当前层的原始文本/摘要文本
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        # 向量字段：存储文本对应的嵌入向量，维度必须和模型输出一致
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    
    # 创建集合
    schema = CollectionSchema(fields, description=f"RAPTOR索引第{level}层")
    Collection(collection_name, schema)
    print(f"[Milvus] 已创建第{level}层集合：{collection_name}")


def insert_docs_to_milvus(level: int, texts: list[str], embeddings: np.ndarray):
    """
    将文本和对应向量写入指定层级的 Milvus 集合，持久化存盘
    :param level: 层级编号
    :param texts: 文本列表
    :param embeddings: 对应文本的向量数组
    """
    collection_name = f"{COLLECTION_PREFIX}{level}"
    collection = Collection(collection_name)
    
    # 整理插入数据，顺序必须和字段定义一致
    data = [
        texts,
        embeddings.tolist()  # numpy数组转列表，适配Milvus格式
    ]
    
    # 插入数据
    collection.insert(data)
    # 立即刷盘，确保数据持久化到磁盘
    collection.flush()
    
    # 为向量字段创建索引，提升检索速度
    index_params = {
        "metric_type": "IP",  # 内积相似度，和向量计算方式一致
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    # 加载集合到内存，准备后续检索
    collection.load()
    
    print(f"[Milvus] 第{level}层已写入{len(texts)}条数据，索引创建完成")


def search_milvus(level: int, query_embed: np.ndarray, top_k: int = 1) -> list[str]:
    """
    在指定层级的 Milvus 集合中做向量相似度检索
    :param level: 检索的层级
    :param query_embed: 查询问题的向量
    :param top_k: 返回最相关的结果数
    :return: 最相关的文本列表
    """
    collection_name = f"{COLLECTION_PREFIX}{level}"
    collection = Collection(collection_name)
    
    # 检索参数
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # 执行向量检索
    results = collection.search(
        data=[query_embed.tolist()],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"]  # 返回结果中包含文本内容
    )
    
    # 提取返回的文本内容
    return [hit.entity.get("text") for hit in results[0]]


def get_max_exist_level() -> int:
    """查询当前Milvus中已存在的最大RAPTOR层级，用于加载已有索引"""
    level = 0
    while utility.has_collection(f"{COLLECTION_PREFIX}{level}"):
        level += 1
    return level - 1  # 返回最大有效层级

# ===================== RAPTOR 核心工具函数 =====================
def get_embeddings(texts: list[str]) -> np.ndarray:
    """批量生成文本的嵌入向量"""
    embeds = embed_model.embed_documents(texts)
    return np.array(embeds)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> list[list[int]]:
    """对向量做KMeans语义聚类，返回每个聚类的文本索引列表"""
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters <= 1:
        return [list(range(len(embeddings)))]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # 易懂写法：按标签分组归集索引
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    return list(clusters.values())


def generate_cluster_summary(cluster_texts: list[str]) -> str:
    """为一个聚类的所有文本生成摘要，作为上一层节点内容"""
    combined = "\n".join(cluster_texts)
    result = summary_chain.invoke({"texts": combined})
    return result.strip()

# ===================== 核心：递归构建RAPTOR持久化索引 =====================
def build_raptor_persistent_index(current_chunks: list[str], current_level: int = 0):
    """
    自底向上递归构建RAPTOR多层索引，每一层都持久化写入Milvus
    :param current_chunks: 当前层的文本块
    :param current_level: 当前层级编号
    """
    # 步骤1：生成当前层文本的向量
    current_embeds = get_embeddings(current_chunks)
    
    # 步骤2：创建当前层的Milvus集合，写入数据持久化
    create_level_collection(current_level)
    insert_docs_to_milvus(current_level, current_chunks, current_embeds)
    
    # 步骤3：递归终止条件：达到最大层数 或 文本块太少无法聚类
    if current_level >= MAX_LEVELS or len(current_chunks) <= 2:
        print(f"[RAPTOR构建] 到达第{current_level}层，递归终止")
        return
    
    # 步骤4：对当前层向量做聚类
    n_clusters = max(2, int(len(current_chunks) * CLUSTER_RATIO))
    cluster_indices = cluster_embeddings(current_embeds, n_clusters)
    
    # 步骤5：每个聚类生成摘要，作为上一层的文本块
    upper_chunks = []
    for indices in cluster_indices:
        cluster_texts = [current_chunks[i] for i in indices]
        summary = generate_cluster_summary(cluster_texts)
        upper_chunks.append(summary)
    
    print(f"[RAPTOR构建] 第{current_level}层完成，共{len(current_chunks)}块，生成{len(upper_chunks)}个摘要进入上一层")
    
    # 步骤6：递归构建上一层
    build_raptor_persistent_index(upper_chunks, current_level + 1)

# ===================== 核心：RAPTOR分层检索（从Milvus读取） =====================
def raptor_persistent_retrieve(query: str, top_k: int = 2) -> list[str]:
    """
    自顶向下分层检索，从Milvus持久化索引中读取数据
    流程：顶层总摘要 → 中层主题摘要 → 叶子层原始细节
    :param query: 用户查询问题
    :param top_k: 最终返回的叶子层结果数
    :return: 最相关的原始文本块列表
    """
    # 生成查询问题的向量
    query_embed = np.array(embed_model.embed_query(query))
    
    # 步骤1：找到当前已构建的最高层级
    max_level = get_max_exist_level()
    if max_level < 0:
        raise Exception("未找到任何RAPTOR索引，请先执行构建")
    
    print(f"[检索] 从第{max_level}层（顶层）开始分层匹配")
    current_level = max_level
    
    # 步骤2：从顶层开始逐层向下定位
    while current_level > 0:
        # 当前层检索最相关的1个摘要
        top_summary = search_milvus(current_level, query_embed, top_k=1)[0]
        print(f"  第{current_level}层命中：{top_summary[:60]}...")
        # 进入下一层继续匹配
        current_level -= 1
    
    # 步骤3：到达叶子层（第0层），返回Top-K最相关的原始文本
    print(f"  到达第0层（叶子层），返回Top-{top_k}原始结果")
    leaf_results = search_milvus(0, query_embed, top_k=top_k)
    return leaf_results

# ===================== 测试文档（叶子层原始内容） =====================
leaf_chunks = [
    # 花果山主题 3块
    "花果山位于东胜神洲傲来国境内，是一块灵气汇聚的仙山，相传是开天辟地时就存在的灵脉。",
    "花果山上终年不谢的奇花异草遍布，山泉瀑布四季长流，还有茂密的千年古树森林，生态极佳。",
    "花果山里有仙果园种植灵果、平坦的练功场、猴族休憩区三个特殊区域，各有不同功能。",
    # 水帘洞主题 3块
    "水帘洞入口是一道高30丈的天然瀑布，隐藏在花果山之巅，普通人很难发现洞口位置。",
    "水帘洞内部是错综复杂的洞穴系统，分为修炼大厅、藏宝室、议事厅三个核心功能区。",
    "水帘洞的藏宝室有天然防护阵法加持，普通妖法无法攻破，专门存放法宝和丹药。",
    # 东海龙宫主题 3块
    "东海龙宫是建在东海海底的宏伟宫殿群，整体用珊瑚和夜明珠搭建，占地数十里。",
    "龙宫的龙王宝库储存着无数珍宝，包括夜明珠、定海神针等上古神器，是龙宫核心重地。",
    "龙宫兵器库收藏了各式水系法器和神兵利器，是水族将领领取装备的地方。",
]

# ===================== 运行示例 =====================
if __name__ == "__main__":
    # 1. 连接Milvus数据库
    connect_milvus()
    
    # ========== 模式1：首次运行，构建持久化索引（只需运行一次） ==========
    print("\n=== 开始构建RAPTOR持久化索引 ===")
    build_raptor_persistent_index(leaf_chunks, current_level=0)
    print("=== 索引构建完成，已全部持久化到Milvus ===")
    
    # ========== 模式2：已有索引，直接检索（后续运行可注释上面的构建代码） ==========
    print("\n" + "="*50)
    query = "水帘洞里存放宝物的地方有什么防护？"
    print(f"用户查询：{query}")
    print("--- 检索路径 ---")
    results = raptor_persistent_retrieve(query, top_k=2)
    print("--- 最终检索结果 ---")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")


'''
看看得了，没验证
'''