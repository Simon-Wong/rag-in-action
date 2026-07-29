'''
一、输入层集成（Input-layer Integration）
1. 核心定义
输入层集成是最经典、工业界最通用的 RAG 模式：在生成开始前，把检索到的所有参考文档和用户问题拼接在一起，整体作为输入传给大模型，模型基于参考资料一次性生成最终答案。
对应书中描述：「将检索内容与原始查询一同传递给生成器」，典型代表是 REALM 模型，也就是我们日常用的标准 RAG。
2. 通俗类比
相当于开卷考试：拿到题目后，先把所有相关参考书、笔记都摆在桌面，看完所有资料后，一次性写完答案。
3. 工作流程
用户输入问题
检索器从知识库召回所有相关文档
把「问题 + 全部参考文档」拼接成完整提示词
大模型根据提示词一次性生成最终答案
4. 优缺点
优点：实现最简单，逻辑直观，调试方便，是绝大多数 RAG 系统的默认方案
缺点：参考资料过多时容易超出模型上下文窗口；无关资料会形成噪声，干扰生成质量
'''
'''
二、输出层集成（Output-layer Integration）
1. 核心定义
输出层集成是事后修正的模式：大模型先只根据自身知识生成一份初稿答案，然后再把检索到的参考资料和初稿一起输入，让模型基于资料对初稿进行校对、纠错、补充，最终输出优化后的答案。
对应书中描述：「先生成输出，随后将输出与检索结果相结合，采用加权整合的方法来优化」，典型代表是 kNN-LM。
2. 通俗类比
相当于闭卷先写答案，交卷前再翻开参考书，把写错、写漏的地方修改补充一遍，形成最终答案。
3. 工作流程
用户输入问题
大模型不看参考资料，直接生成初稿答案
检索器召回相关文档
把「初稿答案 + 参考资料 + 原始问题」一起传给模型，让模型修正初稿
输出最终优化后的答案
4. 优缺点
优点：不干预模型原生生成逻辑，灵活度高；可以有效减少幻觉，修正初稿的事实错误
缺点：最终质量依赖初稿质量；当初稿和检索资料冲突时，模型可能出现取舍不当的问题
'''
'''
三、中间层集成（Middle-layer Integration）
1. 核心定义
中间层集成是生成过程中动态介入的模式：在生成的不同阶段，多次调用检索获取资料，让模型边生成、边查资料、边调整内容，实现知识的深度融合。
书中提到的原生模型级方案（如 RETRO）需要修改 Transformer 内部结构，工程落地难度极高；我们通常用分步迭代生成来模拟同等效果，不需要修改模型本身。
2. 通俗类比
相当于写学术论文：先写大纲，然后写每个章节的时候，都去查对应的参考文献，补充细节后再接着写下一部分，全程边查边写。
3. 工作流程（工程模拟版）
用户输入问题
第一步：先生成回答的大纲框架
针对大纲的每个要点，分别检索对应的参考资料
基于每个要点的专属资料，分步生成详细内容
把所有部分拼接整合，输出最终的完整答案
4. 优缺点
优点：生成过程中多次利用检索信息，信息利用率高，长文本、复杂问题的生成质量显著更好
缺点：实现逻辑复杂，需要多轮调用模型，生成速度慢；原生模型级的中间层集成需要改模型结构，成本极高
'''


# ===================== 1. 依赖导入 =====================
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===================== 2. 公共基础配置（完全沿用你的环境） =====================
OLLAMA_BASE_URL = "http://192.168.0.119:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

# 初始化嵌入模型
embed_model = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# 初始化大模型
llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    timeout=120.0,
    temperature=0.3
)

# ===================== 3. 构建 FAISS 向量检索库（真实语义检索） =====================
# 3.1 准备知识库文本
knowledge_texts = [
    "RAPTOR是一种递归抽象处理的多层级索引结构，通过聚类+摘要自底向上构建树状索引。",
    "输入层集成是将检索内容与原始查询一同传给生成器，是传统RAG的标准模式。",
    "输出层集成是先生成答案初稿，再用检索结果加权修正，事后校准提升准确率。",
    "中间层集成是在生成器内部中间层引入检索信息，代表模型为RETRO。",
    "MultiVectorRetriever是多向量检索器，用摘要做检索，返回完整原始文档。",
    "RecursiveRetriever可以实现分层递归检索，先粗筛再深入细节。",
    "FAISS是Facebook开源的向量相似度搜索库，支持高效的最近邻检索。",
    "RAG全称检索增强生成，通过外部知识库补充大模型知识，减少幻觉。"
]

# 3.2 转成Document对象
knowledge_docs = []
for text in knowledge_texts:
    doc = Document(page_content=text)
    knowledge_docs.append(doc)

# 3.3 构建FAISS向量索引（内存版，真实语义检索）
print("正在构建FAISS向量索引...")
faiss_db = FAISS.from_documents(knowledge_docs, embed_model)
# 创建检索器，每次返回Top3最相关的文档
retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
print("FAISS索引构建完成\n")

# ===================== 4. 输入层集成 =====================
def input_layer_rag(query: str) -> str:
    """
    输入层集成：检索到的所有资料一次性放进提示词，模型一次性生成答案
    对应传统标准RAG模式
    """
    # 第一步：用FAISS做真实语义检索，获取相关文档
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # 把检索到的文档拼接成参考资料文本
    context_str = ""
    for i, doc in enumerate(retrieved_docs, 1):
        context_str += f"{i}. {doc.page_content}\n"
    
    # 第二步：构建提示词，把资料和问题一起传给模型
    prompt = PromptTemplate.from_template("""
请根据以下参考资料，回答用户的问题。
要求：答案必须基于参考资料，不要编造信息。

参考资料：
{context}

用户问题：{question}

回答：
""")
    
    # 构建链路并生成答案
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context_str,
        "question": query
    })
    
    print("="*50)
    print("【输入层集成 结果】")
    print(f"检索到相关文档：{len(retrieved_docs)}条")
    print(f"最终答案：{answer.strip()}")
    print("="*50 + "\n")
    return answer.strip()

# ===================== 5. 输出层集成 =====================
def output_layer_rag(query: str) -> str:
    """
    输出层集成：先生成初稿，再用检索资料修正优化
    对应事后校准的模式
    """
    # 第一步：闭卷生成初稿（不使用任何检索资料）
    first_draft_prompt = PromptTemplate.from_template("""
请直接回答用户的问题，凭借你的已有知识作答。
用户问题：{question}
初稿答案：
""")
    draft_chain = first_draft_prompt | llm | StrOutputParser()
    first_draft = draft_chain.invoke({"question": query}).strip()
    
    # 第二步：用FAISS检索相关参考资料
    retrieved_docs = retriever.get_relevant_documents(query)
    context_str = ""
    for i, doc in enumerate(retrieved_docs, 1):
        context_str += f"{i}. {doc.page_content}\n"
    
    # 第三步：用资料修正初稿，生成最终答案
    revise_prompt = PromptTemplate.from_template("""
请根据下方的参考资料，对初稿答案进行修正和补充。
要求：
1. 修正初稿中与参考资料不符的内容
2. 补充初稿中缺失的关键信息
3. 保持语句通顺，输出最终的完整答案

参考资料：
{context}

初稿答案：
{draft}

最终修正后的答案：
""")
    
    revise_chain = revise_prompt | llm | StrOutputParser()
    final_answer = revise_chain.invoke({
        "context": context_str,
        "draft": first_draft
    }).strip()
    
    print("="*50)
    print("【输出层集成 结果】")
    print(f"生成的初稿：{first_draft}")
    print(f"检索到相关文档：{len(retrieved_docs)}条")
    print(f"修正后最终答案：{final_answer}")
    print("="*50 + "\n")
    return final_answer

# ===================== 6. 中间层集成（分步迭代模拟版） =====================
def middle_layer_rag(query: str) -> str:
    """
    中间层集成模拟：分步生成，每个步骤都检索对应资料再续写
    模拟生成过程中动态引入检索信息的效果
    """
    # 第一步：先生成回答的大纲框架
    outline_prompt = PromptTemplate.from_template("""
请针对用户的问题，生成3个核心要点的回答大纲，只输出要点，不要展开解释。
用户问题：{question}
回答大纲：
1.
""")
    outline_chain = outline_prompt | llm | StrOutputParser()
    outline_raw = outline_chain.invoke({"question": query}).strip()
    
    # 拆分每个大纲点
    outline_points = []
    for line in outline_raw.split("\n"):
        line = line.strip()
        if line:
            outline_points.append(line)
    
    # 第二步：针对每个大纲点，分别检索资料，生成详细内容
    detail_sections = []
    for idx, point in enumerate(outline_points, 1):
        # 针对当前要点做精准语义检索
        point_docs = retriever.get_relevant_documents(point)
        point_context = ""
        for i, doc in enumerate(point_docs, 1):
            point_context += f"{i}. {doc.page_content}\n"
        
        # 基于该部分的专属资料，生成详细内容
        detail_prompt = PromptTemplate.from_template("""
请根据参考资料，详细阐述下面这个要点，写成一段通顺的文字。
要点：{point}
参考资料：{context}
详细内容：
""")
        detail_chain = detail_prompt | llm | StrOutputParser()
        detail = detail_chain.invoke({
            "point": point,
            "context": point_context
        }).strip()
        detail_sections.append(detail)
    
    # 第三步：整合所有部分，形成最终完整答案
    final_content = "\n\n".join(detail_sections)
    
    print("="*50)
    print("【中间层集成 结果】")
    print("生成的大纲：")
    for p in outline_points:
        print(f"  - {p}")
    print(f"\n分步生成后的最终答案：\n{final_content}")
    print("="*50 + "\n")
    return final_content

# ===================== 运行测试 =====================
if __name__ == "__main__":
    # 测试输入层集成
    input_layer_rag("什么是输入层集成和输出层集成？")
    
    # 测试输出层集成
    output_layer_rag("什么是中间层集成？")
    
    # 测试中间层集成
    middle_layer_rag("RAG的三种检索集成方式有什么区别？")


    '''
    thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-langchain
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/05-检索结果集成方式/04-all.py
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/05-检索结果集成方式/04-all.py:15: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model = OllamaEmbeddings(
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/05-检索结果集成方式/04-all.py:21: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm = Ollama(
正在构建FAISS向量索引...
FAISS索引构建完成

/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/05-检索结果集成方式/04-all.py:61: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
  retrieved_docs = retriever.get_relevant_documents(query)
==================================================
【输入层集成 结果】
检索到相关文档：3条
最终答案：根据参考资料，输入层集成是将检索内容与原始查询一同传给生成器，是传统RAG的标准模式。输出层集成是先生成答案初稿，再用检索结果加权修正，事后校准提升准确率。
==================================================

==================================================
【输出层集成 结果】
生成的初稿：中间层集成（Middle Layer Integration）是指将不同系统或应用程序之间的数据和功能进行整合，使得它们能够互相通信和交换信息。这种集成通常涉及到使用中间层技术，例如API（Application Programming Interface）、微服务等，以实现系统之间的协调和集成。
检索到相关文档：3条
修正后最终答案：根据参考资料，对初稿进行修正和补充：

中间层集成（Middle Layer Integration）是指在生成器内部，将检索信息引入中间层，用于提高模型的精度和有效性。这种集成代表了RETRO（Retrospective）模型的特征。

中间层集成的主要目的是将检索结果加权修正，事后校准提升准确率。这是通过将检索内容与原始查询一同传给生成器来实现的，这样可以更好地理解和处理复杂的数据。

在这种集成中，输入层集成（Input Layer Integration）是将检索内容与原始查询一起传递给生成器，是传统RAG（Retrospective Active Generation）的标准模式。通过这一方式，可以有效地实现系统之间的协调和集成。

综上所述，中间层集成是提高模型精度和有效性的关键组成部分，它代表了RETRO模型的特征，并且可以通过输入层集成来实现。
==================================================

==================================================
【中间层集成 结果】
生成的大纲：
  - 1. 基于类别的检索集成
  - 2. 基于内容的检索集成
  - 3. 基于行为的检索集成

分步生成后的最终答案：
基于类别的检索集成是机器学习和自然语言处理领域的一个重要概念，它涉及将检索结果与原始查询进行整合，以提高答案的准确性和相关性。这种方法可以通过多层次的集成来实现，包括输出层、输入层和中间层。

首先，我们需要了解输出层集成（Output Layer Integration）的原理。这种方法是先生成答案初稿，然后使用检索结果加权修正，以提高答案的准确性。例如，在传统的RAG（Retrieval-Augmented Generation）模型中，生成器会根据检索结果对答案进行修正和校准，这样可以更好地匹配原始查询的意图。

其次，我们需要了解输入层集成（Input Layer Integration）的原理。这种方法是将检索内容与原始查询一同传给生成器，这是传统RAG的标准模式。通过这样做，可以让生成器在生成答案时考虑到检索结果对答案的影响。

最后，我们需要了解中间层集成（Middle Layer Integration）的原理。这种方法是在生成器内部中间层引入检索信息，代表模型为RETRO（Retrieval-Augmented Textualization）。通过这样做，可以让生成器在生成答案时考虑到检索结果对答案的影响，同时也可以利用检索结果来进行文本化。

综上所述，基于类别的检索集成是机器学习和自然语言处理领域的一个重要概念，它涉及将检索结果与原始查询进行整合，以提高答案的准确性和相关性。通过输出层、输入层和中间层集成，可以实现更好的答案生成和修正，进而提高模型的性能。

基于内容的检索集成是RAG（Retrieval-Augmentation-Generation）系统中的一个关键组件，它通过将检索内容与原始查询一同传给生成器来实现。这种方法是传统RAG的标准模式，输入层集成。

在这个过程中，检索内容和原始查询被同时传递给生成器，这样可以更好地理解用户的需求和意图。通过这两个信息，生成器可以产生更加准确和有意义的答案初稿。

但是，仅靠输入层集成可能还不足够，因为答案初稿可能存在一些错误或不完整。因此，输出层集成被引入，这是事后校准提升准确率的关键步骤。首先，生成器产生一个答案初稿，然后使用检索结果加权修正这个初稿。

这种方法可以根据检索结果的重要性和相关性来调整答案初稿中的某些部分。这通过提高答案的准确度和完整性来实现。最后，通过事后校准，可以进一步提升答案的准确率和有效性。

中间层集成是RAG系统中另一个关键组件，它是在生成器内部中间层引入检索信息。这种方法可以代表模型为RETRO（Retrieval-Augmentation-Generation），它结合了输入层集成和输出层集成的优势。

通过在生成器内部引入检索信息，可以更好地理解用户的需求和意图，并且可以根据检索结果加权修正答案初稿。这种方法可以提高答案的准确度和完整性，实现更好的事后校准和提升答案的有效性。

基于行为的检索集成是现代自然语言处理（NLP）技术的一个关键概念，它通过将检索结果和原始查询结合起来来改善答案的准确率。这种方法可以分为三层：输出层集成、输入层集成和中间层集成。

首先，输出层集成是一种基于行为的检索集成方法。它的原理是先生成答案初稿，然后使用检索结果加权修正，最后进行事后校准以提升准确率。这意味着生成器在生成答案之前就已经接收到了检索内容，并且可以根据检索结果来调整答案的内容和优先级。

其次，输入层集成是传统RAG（Retrieval-Augmented Generation）的标准模式。这种方法将检索内容与原始查询一起传给生成器，这样生成器就可以在生成答案之前就已经接收到了检索信息。这种方法的优势是能够更好地理解检索结果，并且可以根据检索结果来调整答案的内容和优先级。

最后，中间层集成是一种基于行为的检索集成方法，它是在生成器内部中间层引入检索信息。这种方法代表了RETRO（Retrieval-Augmented Text Generation）模型，这种模型通过在生成器内部中间层引入检索信息来改善答案的准确率。这种方法可以更好地结合检索结果和原始查询，从而产生更精确和相关的答案。

综上所述，基于行为的检索集成是现代NLP技术的一个关键概念，它通过将检索结果和原始查询结合起来来改善答案的准确率。三层的方法可以提供不同的优势和表现，包括输出层集成、输入层集成和中间层集成。
==================================================
    
    '''