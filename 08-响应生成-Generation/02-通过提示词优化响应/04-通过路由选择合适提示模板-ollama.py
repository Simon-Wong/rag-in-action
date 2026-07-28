from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 定义不同业务场景的提示模板
templates = {
    "customer_service": """
    你是一个专业的客服代表。请根据以下客户反馈和历史案例提供解决方案：
    
    历史案例：
    {similar_cases}
    
    当前客户反馈：{customer_feedback}
    
    请提供：
    1. 问题分析
    2. 具体解决方案
    3. 预防措施
    """,
    
    "technical_support": """
    你是一个技术专家。请根据以下技术问题和历史案例提供解决方案：
    
    历史案例：
    {similar_cases}
    
    当前问题描述：{issue_description}
    
    请提供：
    1. 问题诊断
    2. 解决步骤
    3. 技术建议
    """,
    
    "business_analysis": """
    你是一个商业分析师。请根据以下业务问题和历史案例提供分析：
    
    历史案例：
    {similar_cases}
    
    当前问题描述：{business_issue}
    
    请提供：
    1. 问题分析
    2. 影响评估
    3. 改进建议
    """
}

# 示例数据库
case_database = {
    "customer_service": [
        "客户反馈产品包装破损，我们提供了免费换货服务，并改进了包装材料",
        "客户投诉配送延迟，我们优化了物流路线，并提供了补偿方案",
        "客户反映产品质量问题，我们进行了质量检查，并提供了退款服务"
    ],
    "technical_support": [
        "系统频繁崩溃，通过更新服务器配置和优化代码解决了问题",
        "数据库连接超时，通过增加连接池和优化查询语句解决了问题",
        "API响应慢，通过添加缓存和优化算法提高了性能"
    ],
    "business_analysis": [
        "销售额下降，通过市场调研发现是竞争对手价格战导致，调整了营销策略",
        "客户流失率高，通过客户满意度调查发现是服务体验问题，改进了服务流程",
        "运营成本上升，通过流程优化和自动化降低了成本"
    ]
}

# 创建路由函数
def get_prompt_template_by_question(question):
    # 使用LLM进行意图识别
    intent_prompt = f"""
    请分析以下问题属于哪个场景类型:
    问题: {question}
    
    可选场景:
    - customer_service: 客户服务相关问题
    - technical_support: 技术支持相关问题  
    - business_analysis: 业务分析相关问题

    只需返回对应的场景标识符,如 'customer_service'
    """
    
    intent = llm.invoke(intent_prompt).strip()
    
    if intent in templates:
        return intent,PromptTemplate.from_template(templates[intent])
    else:
        raise ValueError(f"未识别的场景类型: {question}, 识别结果: {intent}")

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

# 获取相似案例
def get_similar_cases(scenario, query, k=2):
    # 创建向量数据库
    embeddings = embed_model_ollama# OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_texts(case_database[scenario], embeddings)
    
    # 检索相似案例
    docs = db.similarity_search(query, k=k)
    return "\n".join([f"- {doc.page_content}" for doc in docs])

# 示例使用

# 初始化LLM
llm = llm_ollama

# 测试不同场景的路由选择
scenarios = ["customer_service", "technical_support", "business_analysis"]
test_queries = {
    "customer_service": "产品包装破损，导致商品损坏",
    "technical_support": "系统频繁崩溃，错误代码500", 
    "business_analysis": "销售额连续三个月下降"
}

# 遍历测试每个场景
for scenario in scenarios:
    query = test_queries[scenario]
    print(f"\n{'='*20} {scenario} {'='*20}")
    print(f"输入问题: {query}")
    
    # 获取对应的提示模板
    intent,prompt_template = get_prompt_template_by_question(query)
    print(f"\n识别到的场景: {intent}")
    print("\n选择的提示模板:")
    print(prompt_template.template)
    
    # 获取相似案例
    similar_cases = get_similar_cases(intent, query)
    print("\n检索到的相似案例:")
    print(similar_cases)
    
    # 根据模板中的变量名设置参数
    template_vars = {
        "customer_feedback": query,
        "issue_description": query,
        "business_issue": query,
        "similar_cases": similar_cases
    }
    
    # 生成回复
    response = llm.invoke(prompt_template.format(**template_vars))
    print("\n生成的回复:")
    print(response)

'''
thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-langchain
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/04-通过路由选择合适提示模板-ollama.py
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/04-通过路由选择合适提示模板-ollama.py:100: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  embed_model_ollama = OllamaEmbeddings(
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/04-通过路由选择合适提示模板-ollama.py:105: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
  llm_ollama = Ollama(

==================== customer_service ====================
输入问题: 产品包装破损，导致商品损坏

识别到的场景: customer_service

选择的提示模板:

    你是一个专业的客服代表。请根据以下客户反馈和历史案例提供解决方案：
    
    历史案例：
    {similar_cases}
    
    当前客户反馈：{customer_feedback}
    
    请提供：
    1. 问题分析
    2. 具体解决方案
    3. 预防措施
    

检索到的相似案例:
- 客户反映产品质量问题，我们进行了质量检查，并提供了退款服务
- 客户投诉配送延迟，我们优化了物流路线，并提供了补偿方案

生成的回复:
**问题分析**

目前的客户反馈表明，产品包装破损导致了商品损坏。这可能是由于发运过程中的不稳定性或包装设计上的缺陷。为确保客户满意，我们需要进一步分析情况。

1.  检查包装是否符合标准：我们需要检查现有的包装规格和设计，以确定是否存在任何问题。
2.  运输过程中的监控：我们需要查看运送过程中是否发生异常，包括视频记录等证据。
3.  客户的反馈信息：收集更多关于客户购买体验的详细信息，如包装处理情况、商品使用情况等。

**具体解决方案**

1. 退款并提供替换产品：为客户提供一份新的、不损坏的产品，并承诺尽快交付。
2.  改善包装设计和发运策略：我们需要根据现有的数据进行分析，找出可能导致包装破裂或商品损伤的原因。然后，我们将改进相关的包装和运输过程，以确保客户接受的产品是质量好的。
3.  客户满意度补偿：我们可以提供一份补偿金给客户，以表达对他们的不满的致歉，并希望让他们能够在未来继续与我们的品牌互动。

**预防措施**

1.  进行定期的包装质量检查：确保所有包装都符合产品发运标准，确保每个商品都是完整且安全地运送至客户。
2.  提高运输管理：我们需要更密切监控物流过程，以确保所有运输均有足够的时间来避免任何意外情况。
3.  客户服务培训：对我们的客服代表进行常规培训，以便他们能够提供最好的客户体验，并迅速有效地解决问题。

我们致力于为客户提供最优质的服务和产品。通过不断的改进和客户反馈，我们将努力建立信任和高质量的关系。

==================== technical_support ====================
输入问题: 系统频繁崩溃，错误代码500

识别到的场景: technical_support

选择的提示模板:

    你是一个技术专家。请根据以下技术问题和历史案例提供解决方案：
    
    历史案例：
    {similar_cases}
    
    当前问题描述：{issue_description}
    
    请提供：
    1. 问题诊断
    2. 解决步骤
    3. 技术建议
    

检索到的相似案例:
- 系统频繁崩溃，通过更新服务器配置和优化代码解决了问题
- API响应慢，通过添加缓存和优化算法提高了性能

生成的回复:
**问题诊断**

根据历史案例，我们知道系统频繁崩溃通常是由服务器配置、代码优化或性能问题引起的。现在，系统出现错误代码500，这是一个严重的指示，可能与系统资源耗尽、数据库连接问题或其它关键问题相关。

我们需要深入分析系统的日志、性能监控数据和代码以确定具体原因。以下步骤可以帮助我们进行问题诊断：

1. 检查系统的错误日志：查看错误代码500的日志，了解系统在发生崩溃时的行为。
2. 评估系统的性能：使用性能工具监控系统的CPU、内存和磁盘使用率，以确定是否存在资源耗尽的问题。
3. 检查代码质量：分析代码，找出可能导致系统崩溃的地方，例如循环无限、资源不够或者算法错误。

**解决步骤**

根据问题诊断结果，我们可以采取以下步骤来解决系统频繁崩溃的问题：

1. **优化服务器配置**：
 * 检查和调整服务器硬件配置（CPU、内存、磁盘等）
 * 重新配置服务器的资源分配
 * 使用调优工具（如top、htop）监控服务器性能
2. **代码优化**：
 * 分析并修复可能导致系统崩溃的地方（例如循环无限、资源不够或者算法错误）
 * 优化数据库连接和查询方式
 * 使用性能测试工具（如Apache JMeter）进行代码性能评估
3. **加强安全性**：
 * 检查系统的安全设置，确保关键配置正确
 * 安装和配置防火墙、入侵检测系统等安全软件

**技术建议**

1. **使用监控工具**：使用监控工具（如Prometheus、Grafana）来实时监控系统的性能和资源使用率。
2. **采用容器化**: 使用容器化技术（如Docker）来管理系统的依赖和环境，减少环境相关的问题。
3. **实施自动化测试**：开发和执行自动化测试脚本来验证代码质量和性能。
4. **使用分区方案**: 分离关键资源和数据，以防止资源耗尽问题在一个点上集中。

通过这些步骤，我们可以有效地解决系统频繁崩溃的问题，确保系统的稳定性和可靠性。

==================== business_analysis ====================
输入问题: 销售额连续三个月下降

识别到的场景: business_analysis

选择的提示模板:

    你是一个商业分析师。请根据以下业务问题和历史案例提供分析：
    
    历史案例：
    {similar_cases}
    
    当前问题描述：{business_issue}
    
    请提供：
    1. 问题分析
    2. 影响评估
    3. 改进建议
    

检索到的相似案例:
- 销售额下降，通过市场调研发现是竞争对手价格战导致，调整了营销策略
- 运营成本上升，通过流程优化和自动化降低了成本

生成的回复:
**问题分析**

根据历史案例，我们知道在销售额下降时，通过市场调研发现竞争对手的价格战是主要原因。同样，在运营成本上升时，通过流程优化和自动化减少了成本。

现在面临的销售额连续三个月下降的问题，可能与以上历史案例中出现的问题类似。但是，问题可能更加复杂和深入。需要进行更细致的分析来确定原因。

**影响评估**

可能的影响包括：

*   销售额下降会对整体业务 revenue 和竞争力造成影响。
*   长期销售额下降可能导致员工 morale 的下降，并引起员工离职率的增加。
*   下降的销售额也可能导致利润率下降。

**改进建议**

1.  **市场调研和 competitor analysis：** 进行更深入的市场调研，了解竞争对手的战略和市场趋势。
2.  **价格战分析：** 分析当前的价格战策略，并确定是否需要进行调整或新策略的制定。
3.  **销售策略调整：** 根据市场调研结果和竞争对手的战略，优化销售策略，增加销量并改善销售效率。
4.  **运营成本管理：** 进行成本管理分析，并确定哪些流程和系统需要优化或自动化，以减少不必要的开支并提高效率。
5.  **数据监测：** 设定数据监测系统，实时跟踪销售额、竞争对手的动态和市场趋势，从而快速响应变化并调整策略。

**总结**

解决销售额连续三个月下降的问题需要进行更细致的分析和深入的市场调研。通过价格战分析、销售策略调整、运营成本管理和数据监测等手段来推动销售额的增长，确保业务在不断变化的市场中保持竞争力。
'''