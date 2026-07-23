# 导入相关的库
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.load import dumps, loads


from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.2:3b"
LLM_TIMEOUT = 200.0

embed_model_ollama = OllamaEmbeddings(
    model=EMBED_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)

llm_ollama = Ollama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=LLM_TIMEOUT,
)


"""
RRF（Reciprocal Rank Fusion）重排算法实现

RRF是一种简单而有效的多检索结果融合算法，它通过将多个检索查询的结果进行排名融合，
来提高检索的准确性和覆盖面。

核心思想：
1. 对于同一个用户问题，生成多个不同角度的查询
2. 分别对每个查询进行检索
3. 使用RRF算法将多个检索结果列表融合成一个统一的排序列表
4. RRF算法为每个文档分配分数：score = 1/(rank + k)，其中rank是该文档在某个结果列表中的排名

优势：
- 提高检索的覆盖面：多个查询可以从不同角度检索相关文档
- 降低单一查询的偏差：通过多查询融合减少单一查询的局限性
- 简单高效：算法复杂度低，易于实现和理解
"""

# 文档目录配置
doc_dir = "90-文档-Data/山西文旅"

def load_documents(directory):
    """
    文档加载函数
    
    功能：读取指定目录中的所有文档（支持PDF、TXT格式）
    
    参数：
        directory (str): 文档所在目录路径
    
    返回：
        list: 加载的文档列表，每个文档包含内容和元数据
    
    说明：
        - 遍历目录中的所有文件
        - 根据文件扩展名选择合适的加载器
        - 支持PDF和TXT格式文件
        - 跳过不支持的文件格式
    """
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if filename.endswith(".pdf"):
            # 使用PyPDFLoader加载PDF文件
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            # 使用TextLoader加载TXT文件
            loader = TextLoader(filepath)
        else:
            continue  # 跳过不支持的文件类型
        
        # 加载文档并添加到列表中
        documents.extend(loader.load())
    return documents

# 第一步：加载文档
print("正在加载文档...")
docs = load_documents(doc_dir)
print(f"成功加载 {len(docs)} 个文档")

# 第二步：文本切块（分割）
print("\n正在进行文本切块...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # 每个文本块的最大字符数
    chunk_overlap=50     # 相邻文本块之间的重叠字符数，确保上下文连续性
)
splits = text_splitter.split_documents(docs)
print(f"文档已切分为 {len(splits)} 个文本块")

# 第三步：创建向量索引
print("\n正在创建向量索引...")
# 使用HuggingFace的轻量级嵌入模型
embed_model = embed_model_ollama
# 使用Chroma向量数据库存储文档向量
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
# 创建检索器
retriever = vectorstore.as_retriever()
print("向量索引创建完成")

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    RRF（Reciprocal Rank Fusion）算法实现
    
    功能：将多个检索结果列表融合成一个统一的排序列表
    
    参数：
        results (list[list]): 多个检索结果列表，每个列表包含按相关性排序的文档
        k (int): RRF算法的调节参数，默认值60（经验值）
    
    返回：
        list: 融合后的(文档, 分数)元组列表，按分数降序排序
    
    算法原理：
        1. 对于每个检索结果列表中的每个文档
        2. 计算该文档的RRF分数：score = 1 / (rank + k)
        3. 如果同一文档出现在多个列表中，累加其分数
        4. 按最终分数对所有文档进行排序
    
    优势：
        - rank越小（排名越靠前），分数越高
        - k参数防止分母为0，并调节不同排名之间的差距
        - 多次出现的文档会获得更高的累积分数
    """
    print(f"RRF算法处理 {len(results)} 个检索结果列表...")
    
    fused_scores = {}  # 存储每个文档的累积分数
    
    # 遍历每个检索结果列表
    for list_idx, docs in enumerate(results):
        print(f"  处理第 {list_idx + 1} 个结果列表，包含 {len(docs)} 个文档")
        
        # 遍历该列表中的每个文档
        for rank, doc in enumerate(docs):
            # 将文档序列化为字符串作为唯一标识
            doc_str = dumps(doc)
            
            # 如果该文档首次出现，初始化分数
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            # 计算RRF分数并累加
            rrf_score = 1 / (rank + k)
            fused_scores[doc_str] += rrf_score
            
            # 调试信息：显示文档在当前列表中的排名和分数
            if rank < 3:  # 只显示前3个文档的详细信息
                print(f"    文档 {rank+1}: RRF分数 = 1/({rank}+{k}) = {rrf_score:.4f}")
    
    # 按分数降序排序，返回(文档, 分数)元组列表
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    print(f"RRF融合完成，共 {len(reranked_results)} 个唯一文档")
    return reranked_results

# 第四步：多查询生成
print("\n💭 配置多查询生成器...")
template = """你是一个帮助用户生成多个搜索查询的助手。

请根据以下问题生成4个不同角度的相关搜索查询，这些查询应该：
1. 从不同的角度理解原问题
2. 使用不同的关键词和表达方式
3. 覆盖问题的不同方面
4. 每个查询单独一行，不要编号、不要序号、不要前缀词（如“查询1:”）
5. 只输出查询语句本身，不要添加任何解释、评价或额外文字
6. 查询之间用换行分隔

原问题：{question}
请生成4个相关的搜索查询："""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = llm_ollama

# 创建查询生成链
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))  # 按行分割生成的查询
)
print("多查询生成器配置完成")

# 第五步：测试示例
print("\n开始RRF重排测试...")
questions = [
    "山西有哪些著名的旅游景点？",
    "云冈石窟的历史背景是什么？",
    "五台山的文化和宗教意义是什么？"
]

# 对每个问题进行RRF检索和重排
for idx, question in enumerate(questions, 1):
    print(f"\n{'='*50}")
    print(f"第 {idx} 个问题：{question}")
    print('='*50)
    
    # 第一步：生成多个查询
    print("\n1️生成多个相关查询...")
    queries = generate_queries.invoke({"question": question})
    # 过滤空查询
    queries = [q.strip() for q in queries if q.strip()]
    print(f"生成了 {len(queries)} 个查询：")
    for i, query in enumerate(queries, 1):
        print(f"  查询 {i}: {query}")
    
    # 第二步：对每个查询进行检索
    print(f"\n2️对每个查询进行向量检索...")
    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  检索查询 {i}: {query}")
        docs = retriever.invoke(query)
        all_results.append(docs)
        print(f"    检索到 {len(docs)} 个相关文档")
    
    # 第三步：使用RRF算法融合结果
    print(f"\n3️使用RRF算法融合检索结果...")
    reranked_docs = reciprocal_rank_fusion(all_results)
    
    # 第四步：展示最终结果
    print(f"\n4️最终RRF重排结果（显示前3个）：")
    print(f"总共融合了 {len(reranked_docs)} 个唯一文档")
    
    for i, (doc, score) in enumerate(reranked_docs[:3], 1):
        print(f"\n排名 {i} (RRF分数: {score:.4f}):")
        # 截取前200个字符避免输出过长
        content_preview = doc.page_content[:200].replace('\n', ' ').strip()
        print(f"   内容预览: {content_preview}...")
        
        # 显示文档来源信息（如果有）
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', '未知来源')
            print(f"   来源: {source}")

print(f"\nRRF重排测试完成！")
print("\nRRF算法总结：")
print("- 多角度查询生成：从不同角度理解用户问题")
print("- 多检索结果融合：整合多个检索结果的优势")
print("- 排名优化：通过RRF算法重新排序文档")
print("- 提高召回率：减少单一查询的遗漏")
print("- 提升相关性：多次出现的文档获得更高权重")


'''
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/01-RRF重排-ollama.py
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/01-RRF重排-ollama.py:21: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model_ollama = OllamaEmbeddings(
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/01-RRF重排-ollama.py:26: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm_ollama = Ollama(
正在加载文档...
成功加载 120 个文档

正在进行文本切块...
文档已切分为 279 个文本块

正在创建向量索引...
向量索引创建完成

💭 配置多查询生成器...
多查询生成器配置完成

开始RRF重排测试...

==================================================
第 1 个问题：山西有哪些著名的旅游景点？
==================================================

1️生成多个相关查询...
生成了 4 个查询：
  查询 1: 浏览山西旅游信息
  查询 2: 山西最著名的景点
  查询 3: 山西省的风景区
  查询 4: 山西旅游必游点

2️对每个查询进行向量检索...
  检索查询 1: 浏览山西旅游信息
    检索到 4 个相关文档
  检索查询 2: 山西最著名的景点
    检索到 4 个相关文档
  检索查询 3: 山西省的风景区
    检索到 4 个相关文档
  检索查询 4: 山西旅游必游点
    检索到 4 个相关文档

3️使用RRF算法融合检索结果...
RRF算法处理 4 个检索结果列表...
  处理第 1 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 2 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 3 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 4 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/01-RRF重排-ollama.py:164: LangChainBetaWarning: The function `loads` is in beta. It is actively being worked on, so the API may change.
  (loads(doc), score)
RRF融合完成，共 7 个唯一文档

4️最终RRF重排结果（显示前3个）：
总共融合了 7 个唯一文档

排名 1 (RRF分数: 0.0495):
   内容预览: Website www.shanxigov.cn (htt p://www.shanxigov.cn/) (in Chinese) Shanxi "Shanxi" in Chinese characters Chinese 山西 Postal Shansi Literal meaning "West of the (Taihang) Mountains" Transcriptions Standa...
   来源: 90-文档-Data/山西文旅/山西-en.pdf

排名 2 (RRF分数: 0.0492):
   内容预览: 交通信息：  大同市西门外有公交车和旅游车可搭乘前往。大同火车站坐4路（1元，30分钟，经过西马路－大北街－大西街，出租车可能就8元左右）到市区西部的新开里汽车站3路（1.5元，30分钟）直达云冈景区东门。可以在红旗大饭店旁坐火车站到燕山的中巴，中途下车，5元。      旅游提示：...
   来源: 90-文档-Data/山西文旅/云冈石窟2.txt

排名 3 (RRF分数: 0.0484):
   内容预览: 交通信息：  大同市西门外有公交车和旅游车可搭乘前往。大同火车站坐4路（1元，30分钟，经过西马路－大北街－大西街，出租车可能就8元左右）到市区西部的新开里汽车站3路（1.5元，30分钟）直达云冈景区东门。可以在红旗大饭店旁坐火车站到燕山的中巴，中途下车，5元。      旅游提示：...
   来源: 90-文档-Data/山西文旅/云冈石窟.txt

==================================================
第 2 个问题：云冈石窟的历史背景是什么？
==================================================

1️生成多个相关查询...
生成了 5 个查询：
  查询 1: 云冈石窟的历史背景是什么？
  查询 2: 云冈石窟的发展史
  查询 3: 云冈石窟文化遗产
  查询 4: 云冈石窟地理位置背景
  查询 5: 云冈石窟保护和管理

2️对每个查询进行向量检索...
  检索查询 1: 云冈石窟的历史背景是什么？
    检索到 4 个相关文档
  检索查询 2: 云冈石窟的发展史
    检索到 4 个相关文档
  检索查询 3: 云冈石窟文化遗产
    检索到 4 个相关文档
  检索查询 4: 云冈石窟地理位置背景
    检索到 4 个相关文档
  检索查询 5: 云冈石窟保护和管理
    检索到 4 个相关文档

3️使用RRF算法融合检索结果...
RRF算法处理 5 个检索结果列表...
  处理第 1 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 2 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 3 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 4 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 5 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
RRF融合完成，共 5 个唯一文档

4️最终RRF重排结果（显示前3个）：
总共融合了 5 个唯一文档

排名 1 (RRF分数: 0.0825):
   内容预览: 云冈石窟 云冈石窟位于中国北部山西省大同市西郊17公里处的武周山南麓，石窟依山开凿，东西绵延1公里。存有主要洞窟45个，大小窟龛252个，石雕造像51000余躯，为中国规模最大的古代石窟群之一，与敦煌莫高窟、洛阳龙门石窟和天水麦积山石窟并称为中国四大石窟艺术宝库。 1961年被国务院公布为全国首批重点文物保护单位，2001年12月14日被联合国教科文组织列入世界遗产名录，2007年5月8日被国家旅...
   来源: 90-文档-Data/山西文旅/云冈石窟.txt

排名 2 (RRF分数: 0.0817):
   内容预览: 云冈石窟 a。b。c。云冈石窟位于中国北部山西省大同市西郊17公里处的武周山南麓，石窟依山开凿，东西绵延1公里;存有主要洞窟45个，大小窟龛252个，石雕造像51000余躯，为中国规模最大的古代石窟群之一，与敦煌莫高窟、洛阳龙门石窟和天水麦积山石窟并称为中国四大石窟艺术宝库; 1961年被国务院公布为全国首批重点文物保护单位，2001年12月14日被联合国教科文组织列入世界遗产名录，2007年5月...
   来源: 90-文档-Data/山西文旅/云冈石窟2.txt

排名 3 (RRF分数: 0.0645):
   内容预览: 云冈五华洞 位于云冈石窟中部的第 9——13窟。这五窟因清代施泥彩绘云冈石窟景观而得名。五华洞雕饰绮丽，丰富多彩，是研究北魏历史、艺术、音乐、舞蹈、书法和建筑的珍贵资料，为云冈石窟群的重要组成部分。  塔洞 云冈东部窟群，指云冈石窟东端1——4，均为塔洞。第1、2窟为同期开的一组，凿于孝文帝迁洛前，窟内中央雕造方形塔柱，四面开龛造像。第一窟主像是弥勒，塔南面下层雕释迦多宝像，上层雕释迦像。浮雕五层...
   来源: 90-文档-Data/山西文旅/云冈石窟2.txt

==================================================
第 3 个问题：五台山的文化和宗教意义是什么？
==================================================

1️生成多个相关查询...
生成了 4 个查询：
  查询 1: 五台山的历史与文化意义
  查询 2: 五台山宗教信仰与文化遗产
  查询 3: 五台山文化影响和社会意义
  查询 4: 五台山在东北地区的神话传说

2️对每个查询进行向量检索...
  检索查询 1: 五台山的历史与文化意义
    检索到 4 个相关文档
  检索查询 2: 五台山宗教信仰与文化遗产
    检索到 4 个相关文档
  检索查询 3: 五台山文化影响和社会意义
    检索到 4 个相关文档
  检索查询 4: 五台山在东北地区的神话传说
    检索到 4 个相关文档

3️使用RRF算法融合检索结果...
RRF算法处理 4 个检索结果列表...
  处理第 1 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 2 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 3 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
  处理第 4 个结果列表，包含 4 个文档
    文档 1: RRF分数 = 1/(0+60) = 0.0167
    文档 2: RRF分数 = 1/(1+60) = 0.0164
    文档 3: RRF分数 = 1/(2+60) = 0.0161
RRF融合完成，共 7 个唯一文档

4️最终RRF重排结果（显示前3个）：
总共融合了 7 个唯一文档

排名 1 (RRF分数: 0.0664):
   内容预览: 武州山 武周山，亦名武州山，在大同城西山中。宋《太平寰宇记》引《冀州图》云：“武周山在郡西北，东西数百里，南北五十里。山之南面，千仞壁立。”云冈石窟即因武周山南缘斩山开凿。  昙曜五窟 第十六至二十窟，是云冈石窟最早开业凿的五个洞窟，通称“昙曜五窟。”十六窟为平面呈椭圆形。正中主像释迦像，高13.5米，立于莲花座上，周壁雕有千佛和佛龛。第11窟第十七窟，主像是三世佛，正中为交弥勒坐像，高15.6米...
   来源: 90-文档-Data/山西文旅/云冈石窟2.txt

排名 2 (RRF分数: 0.0653):
   内容预览: 武州山 武周山，亦名武州山，在大同城西山中。宋《太平寰宇记》引《冀州图》云：“武周山在郡西北，东西数百里，南北五十里。山之南面，千仞壁立。”云冈石窟即因武周山南缘斩山开凿。  昙曜五窟 第十六至二十窟，是云冈石窟最早开业凿的五个洞窟，通称“昙曜五窟。”十六窟为平面呈椭圆形。正中主像释迦像，高13.5米，立于莲花座上，周壁雕有千佛和佛龛。第11窟第十七窟，主像是三世佛，正中为交弥勒坐像，高15.6米...
   来源: 90-文档-Data/山西文旅/云冈石窟.txt

排名 3 (RRF分数: 0.0489):
   内容预览: Mount Hengshan (Heng Shan), in Hunyuan County, is one of the "Five Great Peaks" of China, and is also a major Taoist site. Not far from Hengshan, the Hanging Temple is located on the side of a cliff a...
   来源: 90-文档-Data/山西文旅/山西-en.pdf

RRF重排测试完成！

RRF算法总结：
- 多角度查询生成：从不同角度理解用户问题
- 多检索结果融合：整合多个检索结果的优势
- 排名优化：通过RRF算法重新排序文档
- 提高召回率：减少单一查询的遗漏
- 提升相关性：多次出现的文档获得更高权重

'''