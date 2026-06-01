# 使用UnstructuredLoader加载网页
from langchain_unstructured import UnstructuredLoader
page_url = "https://www.cnblogs.com/wlsandwho/p/18673629"
loader = UnstructuredLoader(web_url=page_url)
docs = loader.load()
for doc in docs[:5]:
    print(f'{doc.metadata["category"]}: {doc.page_content}')

