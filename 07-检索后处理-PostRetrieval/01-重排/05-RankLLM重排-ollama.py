from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import torch


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

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
import re
from pydantic import Field
from typing import Any
from typing import Sequence, Optional
from langchain.callbacks.manager import Callbacks

"""
RankLLM重排算法实现

RankLLM是一种基于大语言模型（LLM）的重排方法，利用LLM强大的语言理解能力进行文档重排。

核心原理：
1. 利用LLM的深度语言理解能力判断查询与文档的相关性
2. 通过prompt engineering引导LLM进行排序决策
3. 结合LLM的推理能力，能够处理复杂的语义关系

技术特点：
- 语义理解深度：基于LLM的强大语言理解能力
- 推理能力强：能够进行复杂的逻辑推理和语义匹配
- 灵活性高：可以通过prompt调整适应不同领域和任务
- 解释性好：LLM可以提供排序的理由和解释

与其他方法对比：
- vs BERT类模型：语义理解更深入，能够处理更复杂的推理
- vs 传统重排：能够理解上下文和隐含信息
- vs 嵌入模型：不仅考虑相似度，还考虑逻辑关系

适用场景：
- 对精度要求极高的应用
- 需要复杂推理的查询
- 领域专业性强的文档检索
- 需要可解释性的重排任务

注意事项：
- 计算成本较高（调用LLM API）
- 延迟相对较大
- 需要合理设计prompt
"""

print("初始化RankLLM重排系统...")

# 1. 文档加载和预处理
print("加载和预处理文档...")
doc_path = "90-文档-Data/山西文旅/云冈石窟.txt"
print(f"文档路径: {doc_path}")

print("  使用TextLoader加载文档...")
documents = TextLoader(doc_path).load()
print(f"  成功加载文档，原始文档数量: {len(documents)}")

print("  开始文档分割...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个文档块500个字符
    chunk_overlap=100     # 块之间重叠100个字符，保持上下文连续性
)
texts = text_splitter.split_documents(documents)
print(f"  分割后文档块数量: {len(texts)}")

# 为每个文档块添加唯一ID
print("  为文档块添加唯一标识...")
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
    text.metadata["chunk_size"] = len(text.page_content)
print("  文档预处理完成")

# 2. 创建向量检索器
print(f"\n创建FAISS向量检索器...")
print("  加载嵌入模型...")
embed_model = embed_model_ollama
print("  构建FAISS向量索引...")
retriever = FAISS.from_documents(texts, embed_model).as_retriever(
    search_kwargs={"k": 20}  # 第一阶段检索Top-20文档
)
print(f"  向量检索器创建完成，将返回Top-20候选文档")

# 3. GPU内存优化（如果使用GPU）
print(f"\n优化GPU内存使用...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("  已清理GPU缓存")
else:
    print("  当前使用CPU模式")

# 4. 配置RankLLM重排器
print(f"\n配置自定义重排器...")


# ---------- 2. 自定义基于 Ollama 的重排器 ----------
class OllamaReranker(BaseDocumentCompressor):
    """
    使用 Ollama 本地大模型进行文档重排
    工作原理：对每个文档，让 LLM 输出一个 0-10 的相关性分数，然后按分数排序
    """
    # ----- 声明 Pydantic 字段（必须） -----
    llm: Any                     # 允许任何类型的 LLM 实例
    top_n: int = Field(default=3, ge=1)  # 默认为 3，且至少为 1

    def __init__(self, llm, top_n: int = 3):
        # 通过 super() 传递字段值，Pydantic 会验证并存储
        super().__init__(llm=llm, top_n=top_n)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        对文档列表进行重排，返回 top_n 个最相关的文档（已排序）
        """
        # 在方法内部构建 prompt 和 chain，避免存储为类属性（省去 Pydantic 字段定义）
        prompt_template = PromptTemplate(
            input_variables=["query", "document"],
            template="""你是一个相关性评估专家。请判断以下文档与查询的相关程度，只输出一个 0 到 10 之间的整数分数（0=完全无关，10=完美匹配）。

查询：{query}

文档：{document}

分数（仅输出数字，不要包含其他文字）："""
        )
        chain = prompt_template | self.llm

        scored_docs = []
        for doc in documents:
            content = doc.page_content[:2000]
            response = chain.invoke({"query": query, "document": content})
            match = re.search(r"\d+", response.strip())
            score = int(match.group()) if match else 0
            score = max(0, min(10, score))
            doc.metadata["rerank_score"] = score
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.top_n]]

# ---------- 3. 用自定义重排器替换原 RankLLMRerank ----------
compressor = OllamaReranker(
    llm=llm_ollama,
    top_n=3
)

print("  RankLLM重排器配置完成")

# 5. 创建上下文压缩检索器
print(f"\n创建上下文压缩检索器...")
print("  组合向量检索器和RankLLM重排器...")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,     # 使用RankLLM作为压缩器（重排器）
    base_retriever=retriever        # 使用FAISS作为基础检索器
)
print("  检索链条构建完成：FAISS检索 → RankLLM重排")

# 6. 执行查询和重排
print(f"\n开始执行查询和重排...")
query = "云冈石窟有哪些著名的造像？"
print(f"查询问题: {query}")

print(f"\n第一阶段 - FAISS向量检索:")
print("  基于语义相似度检索候选文档...")

print(f"\n第二阶段 - RankLLM重排:")
print("  调用GPT模型进行深度语义重排...")
print("  正在处理中（LLM推理需要一些时间）...")

try:
    compressed_docs = compression_retriever.invoke(query)
    print(f"  RankLLM重排完成")
    print(f"  最终返回 {len(compressed_docs)} 个高质量文档")

    # 7. 格式化输出重排结果
    def pretty_print_docs(docs):
        """
        美化文档输出函数
        
        功能：以易读的格式展示重排后的文档结果
        
        参数：
            docs (list): 重排后的文档列表
        """
        print(f"\n{'='*60}")
        print(f"RankLLM重排最终结果")
        print(f"{'='*60}")
        print(f"查询: {query}")
        print(f"重排后文档（按相关性降序）:")
        
        result_parts = []
        for i, doc in enumerate(docs, 1):
            doc_info = f"\n排名 {i}:\n"
            doc_info += f"   文档内容:\n{doc.page_content}\n"
            
            # 显示文档元数据
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_info += f"   文档ID: {doc.metadata.get('id', '未知')}\n"
                doc_info += f"   内容长度: {doc.metadata.get('chunk_size', len(doc.page_content))} 字符\n"
                if 'source' in doc.metadata:
                    doc_info += f"   来源文件: {doc.metadata['source']}\n"
            
            result_parts.append(doc_info)
        
        return "\n" + ("-" * 100) + "\n".join(result_parts)

    # 输出美化的结果
    formatted_result = pretty_print_docs(compressed_docs)
    print(formatted_result)

except Exception as e:
    print(f"  RankLLM重排失败: {str(e)}")
    print("  可能的原因:")
    print("    - 文档内容格式问题")
    print("  建议检查:")
    print("    - 文档文件是否存在")

# 8. 资源清理
print(f"\n清理系统资源...")
try:
    # 清理RankLLM模型（如果需要）
    if 'compressor' in locals():
        del compressor
        print("  已释放RankLLM模型资源")
    
    # 再次清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  已清理GPU缓存")
    
    print("  资源清理完成")
except Exception as e:
    print(f"  资源清理时出现警告: {str(e)}")

print(f"\nRankLLM重排总结:")
print("- 深度理解：基于LLM的强大语言理解能力")
print("- 推理能力：能够进行复杂的逻辑推理和语义匹配")
print("- 高精度：利用最先进的语言模型技术")
print("- 可解释：LLM可以提供排序的理由和依据")
print("- 高成本：需要调用LLM API，成本相对较高")
print("- 高延迟：LLM推理时间相对较长")
print("- 最佳实践：适用于对精度要求极高的重要查询")
print("- 优化建议：合理设计prompt以提升重排效果")


'''

(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/05-RankLLM重排-ollama.py
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/05-RankLLM重排-ollama.py:17: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model_ollama = OllamaEmbeddings(
/home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/01-重排/05-RankLLM重排-ollama.py:22: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm_ollama = Ollama(
初始化RankLLM重排系统...
加载和预处理文档...
文档路径: 90-文档-Data/山西文旅/云冈石窟.txt
  使用TextLoader加载文档...
  成功加载文档，原始文档数量: 1
  开始文档分割...
  分割后文档块数量: 4
  为文档块添加唯一标识...
  文档预处理完成

创建FAISS向量检索器...
  加载嵌入模型...
  构建FAISS向量索引...
  向量检索器创建完成，将返回Top-20候选文档

优化GPU内存使用...
  当前使用CPU模式

配置自定义重排器...
  RankLLM重排器配置完成

创建上下文压缩检索器...
  组合向量检索器和RankLLM重排器...
  检索链条构建完成：FAISS检索 → RankLLM重排

开始执行查询和重排...
查询问题: 云冈石窟有哪些著名的造像？

第一阶段 - FAISS向量检索:
  基于语义相似度检索候选文档...

第二阶段 - RankLLM重排:
  调用GPT模型进行深度语义重排...
  正在处理中（LLM推理需要一些时间）...
  RankLLM重排完成
  最终返回 3 个高质量文档

============================================================
RankLLM重排最终结果
============================================================
查询: 云冈石窟有哪些著名的造像？
重排后文档（按相关性降序）:

----------------------------------------------------------------------------------------------------
排名 1:
   文档内容:
云冈石窟
云冈石窟位于中国北部山西省大同市西郊17公里处的武周山南麓，石窟依山开凿，东西绵延1公里。存有主要洞窟45个，大小窟龛252个，石雕造像51000余躯，为中国规模最大的古代石窟群之一，与敦煌莫高窟、洛阳龙门石窟和天水麦积山石窟并称为中国四大石窟艺术宝库。 1961年被国务院公布为全国首批重点文物保护单位，2001年12月14日被联合国教科文组织列入世界遗产名录，2007年5月8日被国家旅游局评为首批国家5A级旅游景区。


云冈五华洞
位于云冈石窟中部的第 9——13窟。这五窟因清代施泥彩绘云冈石窟景观而得名。五华洞雕饰绮丽，丰富多彩，是研究北魏历史、艺术、音乐、舞蹈、书法和建筑的珍贵资料，为云冈石窟群的重要组成部分。
   文档ID: 0
   内容长度: 318 字符
   来源文件: 90-文档-Data/山西文旅/云冈石窟.txt


排名 2:
   文档内容:
塔洞
云冈东部窟群，指云冈石窟东端1——4，均为塔洞。第1、2窟为同期开的一组，凿于孝文帝迁洛前，窟内中央雕造方形塔柱，四面开龛造像。第一窟主像是弥勒，塔南面下层雕释迦多宝像，上层雕释迦像。浮雕五层小塔，是研究北魏建筑的形象资料。第二窟是释迦像，塔南面下层雕释迦多宝像，上层雕三世佛。两窟南壁窟门两侧都雕有维摩、文殊。第三窟为云冈石窟中规模最大的洞窟，前立壁高约25米，传为昙曜译经楼。


武州山
武周山，亦名武州山，在大同城西山中。宋《太平寰宇记》引《冀州图》云：“武周山在郡西北，东西数百里，南北五十里。山之南面，千仞壁立。”云冈石窟即因武周山南缘斩山开凿。

昙曜五窟
第十六至二十窟，是云冈石窟最早开业凿的五个洞窟，通称“昙曜五窟。”十六窟为平面呈椭圆形。正中主像释迦像，高13.5米，立于莲花座上，周壁雕有千佛和佛龛。第11窟第十七窟，主像是三世佛，正中为交弥勒坐像，高15.6米。东、西两壁各雕龛，东为坐像，西为立像。明窗东侧的北魏太和十三年(公元489年)佛龛，是以后补刻的。

景点地址
云冈石窟位于中国北部山西省大同市西郊17公里处的武周山南麓

最佳旅游时间
5月——10月
   文档ID: 1
   内容长度: 498 字符
   来源文件: 90-文档-Data/山西文旅/云冈石窟.txt


排名 3:
   文档内容:
景点地址
云冈石窟位于中国北部山西省大同市西郊17公里处的武周山南麓

最佳旅游时间
5月——10月

开放时间
8:30——17:30（4月1日至10月15日）、8:30——17:00（10月16日至次年3月31日）；云冈博物馆开放时间：9:30——17:00（4月1日至10月15日）、9:30——16:20（10月16日至次年3月31日）

其他提示
 
景区门票：

旺季（4月1日至10月15日）票价：125元，半价票60元。 淡季（10月16日至次年3月31日）票价：80元，半价票40元。    

交通信息：

大同市西门外有公交车和旅游车可搭乘前往。大同火车站坐4路（1元，30分钟，经过西马路－大北街－大西街，出租车可能就8元左右）到市区西部的新开里汽车站3路（1.5元，30分钟）直达云冈景区东门。可以在红旗大饭店旁坐火车站到燕山的中巴，中途下车，5元。    

旅游提示：
   文档ID: 2
   内容长度: 399 字符
   来源文件: 90-文档-Data/山西文旅/云冈石窟.txt


清理系统资源...
  已释放RankLLM模型资源
  资源清理完成

RankLLM重排总结:
- 深度理解：基于LLM的强大语言理解能力
- 推理能力：能够进行复杂的逻辑推理和语义匹配
- 高精度：利用最先进的语言模型技术
- 可解释：LLM可以提供排序的理由和依据
- 高成本：需要调用LLM API，成本相对较高
- 高延迟：LLM推理时间相对较长
- 最佳实践：适用于对精度要求极高的重要查询
- 优化建议：合理设计prompt以提升重排效果
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''