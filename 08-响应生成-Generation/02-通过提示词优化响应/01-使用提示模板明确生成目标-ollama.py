from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAI
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

# 1. 加载文档
loader = TextLoader("90-文档-Data/黑悟空/设定.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = embed_model_ollama #OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.from_documents(texts, embeddings)

# 4. 检索相关内容
query = "白骨精的特点和战斗方式是什么？"
docs = db.similarity_search(query)
retrieved_content = docs[0].page_content

# 5. 定义提示模板
template = """
基于以下检索到的资料：
{context}

请详细分析并按照以下格式生成角色分析报告：

人物名称：[提供完整名称]

背景故事：介绍角色的来历和背景，与其他角色的关系，在故事中的定位。
技能特点：介绍角色的主要技能和能力，特殊能力描述，战斗风格特点。
战斗策略：介绍角色的主要攻击方式，防御机制，战斗中的特殊表现，克制和弱点。

请基于资料进行详尽分析，确保内容准确且具有连贯性。
"""

# 创建PromptTemplate和LLM
prompt = PromptTemplate(
    input_variables=["context"],
    template=template
)

llm = llm_ollama #OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 生成文本
formatted_prompt = prompt.format(context=retrieved_content)
response = llm.invoke(formatted_prompt)
print(response)


'''
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python "/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/01-使用提示模板 明确生成目标 copy.py"
/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/02-通过提示词优化响应/01-使用提示模板明确生成目标 copy.py:3: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import TextLoader
content='根据提供的资料，目前未明确提及《黑神话：悟空》中具体角色的完整名称、背景故事及技能细节。但结合游戏标题、章节主题及文化设定，可推测游戏核心角色可能与《西游记》中的经典人物（如孙悟空、唐僧、妖怪等）存在关联，并融合神话、宗教哲学元素进行再创作。以下基于现有信息进行合理推测和分析：\n\n---\n\n### **人物名称**：悟空（推测为主角，基于游戏标题）  \n**背景故事**：  \n悟空可能是游戏的核心角色，其背景可能基于《西游记》中孙悟空的原型，但被赋予更深层的哲学内涵。他可能因反抗天庭、追求自由而被贬，后在取经路上经历磨难，逐渐揭示“黑神话”背后的真相。游戏章节名称（如“火照黑云”“未竟”）暗示其旅程充满宿命与救赎的主题。与其他角色的关系可能涉及师徒情谊（如唐僧）、敌对（如妖怪或天庭势力），以及自我内心的挣扎。  \n\n**技能特点**：  \n- **主要技能**：可能包括七十二变、筋斗云、金箍棒变形等传统能力，但结合游戏设定可能被重新诠释（如金箍棒可化为武器或法器）。  \n- **特殊能力**：可能涉及佛教（如禅定、因果轮回）或道教（如符咒、炼丹）元素，例如通过“白露”章节的隐喻获得短暂隐身能力，或借助“紫鸳”章节的象征性场景释放范围攻击。  \n- **战斗风格**：以近战格斗为主，结合快速闪避与爆发性攻击，可能融入中国武术的“刚柔并济”理念，同时通过动画过场展现其内心挣扎与力量的双重性。  \n\n**战斗策略**：  \n- **攻击方式**：近战物理攻击（如金箍棒劈砍）、远程法术（如火焰或雷电技能），可能根据章节主题切换风格（如“风起黄昏”中使用风属性技能）。  \n- **防御机制**：通过“禅定”状态短暂免疫伤害，或利用环境（如石刻、寺庙场景）触发护盾。  \n- **特殊表现**：在特定章节（如“夜生白露”）可能解锁“分身术”或“因果轮回”机制，允许短暂时间倒回战斗状态。  \n- **克制与弱点**：可能对火属性攻击抗性较低（因“火照黑云”章节暗示其与火焰的关联），但对精神类攻击（如幻术）有特殊抗性。  \n\n---\n\n### **其他潜在角色分析（基于文化设定推测）**  \n1. **佛教/道教NPC**：  \n   - **背景**：可能作为导师或反派出现，代表宗教哲学中的“执念”或“空性”主题。  \n   - **技能**：使用符咒、幻术或召唤神兽，战斗风格偏向远程控制与群体伤害。  \n   - **弱点**：可能对“因果”概念敏感，若玩家通过剧情选择打破其信仰，可造成额外伤害。  \n\n2. **妖怪/反派角色**：  \n   - **背景**：可能与“黑神话”核心秘密相关，如被封印的上古妖王，或因执念堕落的修行者。  \n   - **技能**：高防御与再生能力，擅长使用环境陷阱（如利用“鹳雀楼”场景制造落石攻击）。  \n   - **战斗策略**：需通过探索章节动画（如“曲度紫鸳”）解锁其弱点，例如其“执念”可被特定法器或剧情选择破除。  \n\n---\n\n### **总结**  \n《黑神话：悟空》的角色设计深度结合了中国传统文化、宗教哲学与神话叙事，但具体角色信息需依赖游戏实际内容。上述分析基于现有资料与《西游记》原型的合理推测，实际游戏中的角色设定可能更加复杂且富有隐喻性。建议关注后续官方资料或游戏实机演示以获取更精准的信息。' additional_kwargs={} response_metadata={'model': 'qwen3:14b', 'created_at': '2026-07-28T02:49:15.9042086Z', 'done': True, 'done_reason': 'stop', 'total_duration': 51843876000, 'load_duration': 9067712600, 'prompt_eval_count': 257, 'prompt_eval_duration': 530366999, 'eval_count': 1252, 'eval_duration': 42231125000, 'logprobs': None, 'model_name': 'qwen3:14b', 'model_provider': 'ollama'} id='lc_run--019fa69f-988f-70a0-b1f5-3b29bdf3a2eb-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 257, 'output_tokens': 1252, 'total_tokens': 1509}

'''