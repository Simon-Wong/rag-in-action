# 使用WebBaseLoader加载网页
import bs4
from langchain_community.document_loaders import WebBaseLoader
page_url = "https://www.cnblogs.com/wlsandwho/p/18673629"
# loader = WebBaseLoader(web_paths=[page_url])
# docs = []
# docs = loader.load()
# assert len(docs) == 1
# doc = docs[0]
# print(f"{doc.metadata}\n")
# print(doc.page_content.strip()[:3000])


# 只解析文章的主体部分
loader = WebBaseLoader(
    web_paths=[page_url],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="cnblogs_post_body"),
    },
    # bs_get_text_kwargs={"separator": " | ", "strip": True},

    #设置请求头，模拟浏览器访问，解决反爬拦截
    requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    }
)
docs = []
docs = loader.load()
assert len(docs) == 1
doc = docs[0]
print(f"{doc.metadata}\n")
print(doc.page_content)
print('='*20)
print(docs)