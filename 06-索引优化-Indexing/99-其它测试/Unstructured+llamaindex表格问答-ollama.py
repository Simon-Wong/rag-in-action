import os
import re
from typing import List
from unstructured.partition.pdf import partition_pdf
import pandas as pd

# 导入 LlamaIndex 相关模块
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import IndexNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# ---------------------------
# 全局设置（请确保已设置 OPENAI_API_KEY 环境变量）
# ---------------------------
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ---------------------------
# 1. 解析 PDF 结构，提取文本和表格
# ---------------------------
file_path = "90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"  # 修改为你的实际文件路径

elements = partition_pdf(
    file_path,
    strategy="hi_res",             # 高精度策略，更准确地识别表格
    extract_tables_in_paragraphs=True,
    include_metadata=True
)

# 分离文本元素和表格元素
text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]

# ---------------------------
# 2. 识别每个表格的年份（基于前一个文本元素中的年份）
# ---------------------------
def extract_year_from_text(text: str):
    """从文本中提取年份（1900-2099）"""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return match.group(0) if match else None

table_data = []          # 存储 (表格DataFrame, 年份)
last_seen_year = None

for element in elements:
    if element.category == "Text":
        year = extract_year_from_text(element.text)
        if year:
            last_seen_year = year
    elif element.category == "Table":
        # 解析表格文本内容为 DataFrame
        rows = element.text.strip().split('\n')
        if len(rows) < 2:
            continue      # 跳过只有表头的空表格
        header = rows[0].split()
        data = [row.split() for row in rows[1:]]
        df = pd.DataFrame(data, columns=header)
        table_data.append({
            "table": df,
            "year": last_seen_year if last_seen_year else "Unknown"
        })

print(f"共提取到 {len(table_data)} 个表格，年份分布：{[t['year'] for t in table_data]}")

# ---------------------------
# 3. 为每个表格创建一个 PandasQueryEngine
# ---------------------------
# 使用 gpt-4 来生成更准确的 pandas 代码（也可用 gpt-3.5-turbo）
llm_for_table = OpenAI(model="gpt-4")

df_query_engines = []
for idx, item in enumerate(table_data):
    engine = PandasQueryEngine(item["table"], llm=llm_for_table, verbose=True)
    df_query_engines.append(engine)

# ---------------------------
# 4. 为每个表格生成一句话摘要，并构建向量索引（用于路由选择）
# ---------------------------
table_summaries = []
for idx, item in enumerate(table_data):
    table_csv = item["table"].to_csv(index=False)
    year = item["year"]
    prompt = f"请用一句话总结{year}年表格的主要内容：\n\n{table_csv}\n\n摘要："
    summary = Settings.llm.complete(prompt).text.strip()
    table_summaries.append(summary)
    print(f"[摘要] {year} 年表格：{summary}")

# 将摘要封装为 IndexNode
summary_nodes = [
    IndexNode(text=summary, index_id=f"table_{idx}")
    for idx, summary in enumerate(table_summaries)
]

# 构建向量索引，用于根据用户问题检索最相关的表格
vector_index = VectorStoreIndex(summary_nodes)

# ---------------------------
# 5. 将 PandasQueryEngine 包装成工具，并用路由器统一调度
# ---------------------------
query_engine_tools = []
for idx, engine in enumerate(df_query_engines):
    year = table_data[idx]["year"]
    tool = QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name=f"table_{year}",
            description=(
                f"提供 {year} 年的亿万富翁排名数据。可以回答关于该年财富数值、"
                f"排名、姓名等问题。请使用此工具查询具体的数字和排序。"
            )
        )
    )
    query_engine_tools.append(tool)

# 使用 LLMSingleSelector 自动选择最合适的工具
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True
)

# ---------------------------
# 6. 测试查询
# ---------------------------
test_queries = [
    "Who was the second richest billionaire in 2023?",
    "How much net worth did the richest person have in 2024?",
    "Tell me the total number of billionaires in 2022."
]

for q in test_queries:
    print("\n" + "="*60)
    print(f"问题：{q}")
    response = router.query(q)
    print(f"回答：{response}")
    print("="*60)