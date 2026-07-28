from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

embed_ollama = OllamaEmbedding(
    model_name="nomic-embed-text",   # 确保与 Ollama 下载的模型名完全一致
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
)

llm_ollama = Ollama(
    model="qwen3:14b",                # 确保与 Ollama 下载的模型名完全一致qwen3.5:9b
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
    request_timeout=120.0,           # (可选) 设置请求超时，单位秒
)

Settings.embed_model = embed_ollama
Settings.llm = llm_ollama


# 定义游戏信息结构
class GameInfo(BaseModel):
    title: str = Field(description="游戏名称")
    developer: str = Field(description="开发商")
    release_date: str = Field(description="发行日期")
    platforms: List[str] = Field(description="支持平台")
    main_features: List[str] = Field(description="主要特点")
    story_summary: str = Field(description="故事概要")
    reception: str = Field(description="市场反响")

# 载入数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/黑悟空wiki.txt"], encoding="utf-8").load_data()
index = VectorStoreIndex.from_documents(documents)

# 1. 基础解析模式 - 使用COMPACT模式
print("=== 基础解析模式 ===")
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    verbose=True    # 显示详细信息
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("请总结《黑神话：悟空》这款游戏的主要内容")
print(response)

# 2. 结构化解析模式 - 使用REFINE模式
print("\n=== 结构化解析模式 ===")
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    output_cls=GameInfo,  # 指定输出类
    verbose=True
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("请提取《黑神话：悟空》的关键信息")
# 安全地处理响应
if hasattr(response, 'response'):
    print(response.response)
else:
    print(response)

# 3. 表格格式解析 - 使用TREE_SUMMARIZE模式
print("\n=== 游戏特点表格解析 ===")
table_prompt = PromptTemplate(
    template="请将以下游戏特点以表格形式展示：\n{query_str}\n格式要求：\n| 类别 | 内容 |\n|------|------|\n"
)
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    summary_template=table_prompt,
    verbose=True
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("请用表格形式总结《黑神话：悟空》的主要特点")
print(response)

# 4. 分点解析模式 - 使用COMPACT_ACCUMULATE模式
print("\n=== 游戏亮点分点解析 ===")
bullet_prompt = PromptTemplate(
    template="请将以下游戏亮点以分点形式展示：\n{query_str}\n格式要求：\n1. \n2. \n3. "
)
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT_ACCUMULATE,
    text_qa_template=bullet_prompt,
    verbose=True,
    use_async=True  # 启用异步处理
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("请用分点形式总结《黑神话：悟空》的亮点")
print(response)

# 5. 故事线解析 - 使用SIMPLE_SUMMARIZE模式
print("\n=== 游戏故事线解析 ===")
story_prompt = PromptTemplate(
    template="请将以下游戏故事以时间线形式展示：\n{query_str}\n格式要求：\n- 时间点：事件\n"
)
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    text_qa_template=story_prompt,
    verbose=True
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("请用时间线形式总结《黑神话：悟空》的故事发展")
print(response)


'''
(venv-rag-llamaindex) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-llamaindex/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/02-LlamaIndex输出 解析-ollama.py
/home/thbytwo/miniforge3/envs/venv-rag-llamaindex/lib/python3.10/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.3.0) or chardet (7.4.3)/charset_normalizer (3.4.1) doesn't match a supported version!
  warnings.warn(
=== 基础解析模式 ===
《黑神话：悟空》是一款以中国神话和传统文化为背景的动作角色扮演游戏。游戏故事发生在《西游记》之后，玩家扮演“天命人”——一只来自花果山的灵明石猴，踏上寻找孙悟空遗失的六件根器的旅程，目标是解救并复活孙悟空。游戏融合了佛教、道教哲学思想以及中国各地的文化地标，如大足石刻、小西天等，通过丰富的场景设计展现东方美学。

在剧情中，玩家将穿越多个区域，与各路妖王、神佛对抗，逐步揭开孙悟空陨落的真相。关键情节包括与黑熊精的对决、黄风大圣的挑战，以及最终与二郎神、四大天王的激战。游戏的结局取决于玩家是否成功收集所有根器：若获得第六件根器“意见欲”，孙悟空将复活，天命人继承其意志；若未能获得，则天命人沦为天庭的工具，孙悟空的执念则持续存在。

游戏通过战斗、探索和叙事结合的方式，探讨自由、束缚与轮回的主题，同时以高质量的画面和动画过场呈现中国神话的深层内涵。

=== 结构化解析模式 ===
title='《黑神话：悟空》关键信息' developer='游戏科学' release_date='2020年8月20日（首段实机演示视频公布）' platforms=['PC', '主机（推测）'] main_features=['融合中国传统文化与自然地标（如重庆大足石刻、山西小西天等）', '佛教与道教哲学元素的融入', '章节结尾包含二维/三维动画过场', '基于虚幻引擎4开发'] story_summary='故事发生在《西游记》之后，孙悟空成佛后因拒绝受拘束遭天庭追杀，死后魂魄化为六件根器（六根）。玩家扮演‘天命人’寻找根器、复活孙悟空，剧情参考《大话西游》并融入《西游记》原著元素。结局中，天命人需在梅山击败四大天王与二郎神，获取第六件根器‘意见欲’，最终与孙悟空‘执念’化身的残躯决战。若成功集齐六根，孙悟空复活并摆脱束缚；若失败，天命人将沦为天庭工具。' reception='未提及具体评价或销量数据'

=== 游戏特点表格解析 ===
1 text chunks after repacking
| 类别 | 内容 |
|------|------|
| **游戏类型** | 动作冒险RPG |
| **画面表现** | 高画质渲染，细节丰富，采用虚幻引擎5技术，场景设计融合中国神话与写实风格 |
| **剧情设定** | 基于《西游记》改编，加入黑暗奇幻元素，主线围绕“天命人”追寻孙悟空的真相 |
| **战斗系统** | 多样化连招与技能组合，BOSS战设计复杂，强调操作与策略结合 |
| **角色设计** | 角色造型融合传统神话与现代审美，妖怪、天庭角色设计极具辨识度 |
| **玩法特色** | 开放世界探索、解谜元素、隐藏剧情分支、装备与技能树系统 |
| **音乐音效** | 交响乐与传统乐器结合，环境音效沉浸感强，战斗BGM激昂富有张力 |
| **开发团队** | 游戏科学（Game Science），由《鬼谷八荒》原团队核心成员组建 |
| **平台信息** | PC（Steam）及PlayStation 5 |
| **创新点** | 开放世界与线性叙事结合、动作系统深度、剧情多结局设计、跨平台同步存档 |

=== 游戏亮点分点解析 ===
Response 1: 1. **沉浸式东方美学与创新画面表现**  
   以中国神话为背景，融合水墨画风格与3D渲染技术，打造极具视觉冲击力的场景与角色设计，呈现媲美电影级的美术品质。

2. **深度动作战斗系统与策略性玩法**  
   结合快节奏近战连招、技能组合与BOSS战机制，强调操作技巧与策略选择，提供高自由度的战斗体验，挑战玩家反应与战术思维。

3. **颠覆性叙事与角色塑造**  
   以《西游记》为灵感重构剧情，通过多线叙事与角色成长，展现主角悟空的复杂内心世界，结合隐喻与哲学思考，赋予传统神话全新深度。

=== 游戏故事线解析 ===
- **游戏开始前（背景设定）**：  
  天庭与妖界的矛盾激化，孙悟空因大闹天宫被压五指山下五百年，其肉身被封印，灵魂游离于天地之间。

- **第一章：五指山下（游戏初期）**：  
  玩家扮演“无名僧人”在五指山下发现被封印的孙悟空，触发“天命”与“业力”双线剧情，揭示悟空堕落的伏笔。

- **第二章：天庭与妖界（主线冲突）**：  
  悟空挣脱封印后，发现天庭与妖界勾结，利用“灵山”之力操控众生，引发三界动荡，悟空开始调查真相。

- **第三章：灵山之谜（关键转折）**：  
  悟空潜入灵山，发现“灵山”实为天庭与妖界共谋的阴谋核心，揭露“如来”与“太上老君”合谋制造“天命”骗局。

- **第四章：三界之战（高潮部分）**：  
  悟空联合被压迫的妖族与反抗的天庭将领，发动三界之战，对抗天庭与妖界的联合势力，揭露“天命”本质为对众生的控制。

- **第五章：自我救赎（结局）**：  
  悟空在最终决战中直面自身“业力”与“天命”的矛盾，选择以肉身毁灭灵山核心，打破天庭与妖界的平衡，但自身灵魂消散，留下“无名僧人”继续寻找真相。

- **尾声（开放结局）**：  
  游戏结尾暗示“无名僧人”继承悟空意志，开始新的旅程，三界陷入新的混乱，留下续作伏笔。
'''