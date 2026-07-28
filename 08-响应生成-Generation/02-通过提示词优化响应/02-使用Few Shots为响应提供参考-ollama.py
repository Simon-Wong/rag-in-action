from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os


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


# 示例数据
examples = [
    {
        "context": "某大型制造企业的供应链系统出现延迟问题，导致生产效率下降15%。经过调查发现主要是由于供应商管理混乱和库存预测不准确导致。",
        "answer": """问题分析报告：
                    核心问题：供应链效率低下
                    影响程度：生产效率降低15%
                    主要原因：
                    - 供应商管理体系不完善
                    - 库存预测系统准确度不足

                    建议方案：
                    1. 优化供应商评估体系
                    2. 引入智能预测系统
                    3. 建立实时监控机制"""
    },
    {
        "context": "某科技公司的员工流失率达到25%，主要集中在研发部门，影响了产品迭代进度。",
        "answer": """问题分析报告：
                    核心问题：高员工流失率
                    影响程度：流失率25%
                    主要原因：
                    - 薪资福利竞争力不足
                    - 职业发展空间受限

                    建议方案：
                    1. 优化薪酬体系
                    2. 完善晋升机制
                    3. 改善工作环境"""
    }
]

# 创建向量数据库
embeddings =embed_model_ollama # OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
example_texts = [ex["context"] for ex in examples]
db = FAISS.from_texts(example_texts, embeddings)

# 用户输入的问题描述
current_issue = """某零售连锁企业的客户投诉率在过去三个月上升40%，主要集中在配送时效和商品质量两个方面，影响了品牌声誉。"""

# 检索最相似的示例
docs = db.similarity_search(current_issue, k=1)
most_similar_example = next(ex for ex in examples if ex["context"] == docs[0].page_content)

# 构建提示词
prompt = """这是一个企业问题分析示例：

示例：
基于以下情况：
{example_context}

{example_answer}

现在，请基于以下问题，按照相同格式生成分析报告：
{input_context}

请保持分析的专业性和可操作性。
"""

# 创建LLM
llm =llm_ollama# OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 格式化提示词并生成回答
formatted_prompt = prompt.format(
    example_context=most_similar_example["context"],
    example_answer=most_similar_example["answer"],
    input_context=current_issue
)

print(formatted_prompt)

response = llm.invoke(formatted_prompt)
print('-'*10)
print(response)


'''
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python "/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/02-使用Few Shots为响应提供参考.py"
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/02-使用Few Shots为响应提供参考.py:2: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import TextLoader
这是一个企业问题分析示例：

示例：
基于以下情况：
某大型制造企业的供应链系统出现延迟问题，导致生产效率下降15%。经过调查发现主要是由于供应商管理混乱和库存预测不准确导致。

问题分析报告：
                    核心问题：供应链效率低下
                    影响程度：生产效率降低15%
                    主要原因：
                    - 供应商管理体系不完善
                    - 库存预测系统准确度不足

                    建议方案：
                    1. 优化供应商评估体系
                    2. 引入智能预测系统
                    3. 建立实时监控机制

现在，请基于以下问题，按照相同格式生成分析报告：
某零售连锁企业的客户投诉率在过去三个月上升40%，主要集中在配送时效和商品质量两个方面，影响了品牌声誉。

请保持分析的专业性和可操作性。

----------
content='问题分析报告：  \n**核心问题：** 客户投诉率显著上升  \n**影响程度：** 投诉率三个月内增长40%，品牌声誉受损  \n**主要原因：**  \n- **配送时效问题：** 物流网络调度不合理、仓储管理效率低、第三方物流协作不畅  \n- **商品质量问题：** 供应商质量管控缺失、入库质检流程不严格、商品存储条件不达标  \n\n**建议方案：**  \n1. **优化配送体系**  \n   - 引入智能物流调度系统，动态调整配送路线与运力分配  \n   - 优化仓储布局，提升拣货与打包效率  \n   - 与第三方物流重新签订服务协议，明确时效考核标准  \n\n2. **强化商品质量管控**  \n   - 建立供应商分级管理制度，实施定期质量审计  \n   - 升级入库质检流程，增加抽样检测频次与标准  \n   - 改善仓储温湿度监控系统，确保商品存储环境符合规范  \n\n3. **建立客户反馈闭环**  \n   - 开发投诉分类追踪系统，实时分析投诉数据趋势  \n   - 设立专项整改小组，针对高频问题制定限时改进计划  \n   - 通过客户回访机制提升问题解决透明度，修复品牌信任度' additional_kwargs={} response_metadata={'model': 'qwen3:14b', 'created_at': '2026-07-28T03:01:17.5757601Z', 'done': True, 'done_reason': 'stop', 'total_duration': 20163564300, 'load_duration': 128243400, 'prompt_eval_count': 197, 'prompt_eval_duration': 43284000, 'eval_count': 626, 'eval_duration': 19899740000, 'logprobs': None, 'model_name': 'qwen3:14b', 'model_provider': 'ollama'} id='lc_run--019fa6ab-106d-78d2-a395-37aed12d7122-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 197, 'output_tokens': 626, 'total_tokens': 823}
'''