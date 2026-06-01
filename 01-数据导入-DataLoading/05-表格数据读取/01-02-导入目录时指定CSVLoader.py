from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader

loader = DirectoryLoader(
    path="90-文档-Data/黑悟空",  # Specify the directory containing your CSV files
    glob="**/*.csv",                # Use a glob pattern to match CSV files
    loader_cls=CSVLoader            # Specify CSVLoader as the loader class
    #loader_cls=lambda path: CSVLoader(path, csv_args={"delimiter": ",","fieldnames": ["种类", "名称", "说明", "等级"],}, autodetect_encoding=True)#可以用lambda进行具体的设置，细节和单独用一样
)

docs = loader.load()
print(f"文档数：{len(docs)}")  # 输出文档总数
print(docs[0])

#这个实际上是加载一个文件夹里的所有csv，并指定加载器类型，没有处理别的文件。
#书上搞错了。我自己写了一个见01-03