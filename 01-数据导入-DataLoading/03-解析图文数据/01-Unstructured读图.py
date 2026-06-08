from langchain_community.document_loaders import UnstructuredImageLoader
image_path = "90-文档-Data/黑悟空/黑悟空英文.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)


loader = UnstructuredImageLoader("90-文档-Data/黑悟空/IMG_2020.JPG")
data = loader.load()
print(data)

loader = UnstructuredImageLoader("90-文档-Data/黑悟空/IMG_3071.JPG")
data = loader.load()
print(data)

loader = UnstructuredImageLoader("90-文档-Data/黑悟空/IMG_3757.JPG")
data = loader.load()
print(data)
