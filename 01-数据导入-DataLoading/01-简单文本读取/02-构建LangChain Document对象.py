from langchain_core.documents import Document
documents = [
    Document(
        page_content="悟空是大师兄.",
        metadata={"source": "师徒四人.txt"},
    ),
    Document(
        page_content="八戒是二师兄.",
        metadata={"source": "师徒四人.txt "},
    ),
]
print(documents)

'''
#如果你想把每一句话分成一个独立的 Document（和你最初的例子一样），只需要加一个分割器：

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 加载文件
loader = TextLoader("师徒四人.txt", encoding="utf-8")
documents = loader.load()

# 按换行符分割，每一行变成一个 Document
splitter = CharacterTextSplitter(separator="\n", chunk_size=100)
split_docs = splitter.split_documents(documents)

# 打印分割后的结果
print("分割后 Document 数量：", len(split_docs))
for i, doc in enumerate(split_docs):
    print(f"\n第{i+1}个Document：")
    print(doc)

'''