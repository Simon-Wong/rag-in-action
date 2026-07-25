#pip install ddgs

from ddgs import DDGS

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

result = web_search("2026年世界杯冠军")
print(result)