#pip install ddgs

from langchain.tools import tool
from ddgs import DDGS

@tool
def web_search(query: str) -> str:
    """搜索互联网获取信息，参数query为搜索关键词，返回结果摘要。"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "未找到相关结果。"
    
    output = []
    for i, r in enumerate(results, 1):
        output.append(f"{i}. {r['title']}\n   链接：{r['href']}\n   摘要：{r['body']}")
    return "\n\n".join(output)

# 之后所有用到 web_search_tool 的地方，直接用这个 web_search 即可
result = web_search.invoke("2026年世界杯冠军")
print(result)