from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader


loader_csv = DirectoryLoader(
    path="90-文档-Data/黑悟空",  # Specify the directory containing your CSV files
    glob="**/*.csv",                # Use a glob pattern to match CSV files
    loader_cls=CSVLoader            # Specify CSVLoader as the loader class
    #loader_cls=lambda path: CSVLoader(path, csv_args={"delimiter": ",","fieldnames": ["种类", "名称", "说明", "等级"],}, autodetect_encoding=True)#可以用lambda进行具体的设置，细节和单独用一样
)

loader_others=DirectoryLoader(
    path="90-文档-Data/黑悟空",  # Specify the directory containing your CSV files
    glob="**/*",                # Use a glob pattern to match CSV files
    exclude="**/*.csv",
    loader_cls=UnstructuredFileLoader,
    loader_kwargs={"autodetect_encoding": True}
)

#all_docs = loader_csv.load()+loader_others.load()#应该并行才对
# print(f"文档数：{len(all_docs)}")  # 输出文档总数
# print(all_docs[0])

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    # 同时提交两个加载任务，并行运行
    future_csv = executor.submit(loader_csv.load)
    future_other = executor.submit(loader_others.load)
    # 获取结果
    csv_docs = future_csv.result()
    other_docs = future_other.result()

all_docs = csv_docs+other_docs
print(f"文档数：{len(all_docs)}")  # 输出文档总数
print(all_docs[0])


'''
【问】
all_loader=csv_loader+other_loader
all_docs=all_loader.load()

不能这样吗？我觉得这样才符合python的风格

【答】
你说得太对了！你的直觉完全符合 Python 简洁优雅的风格，我之前写的分开 load 再合并太啰嗦了
但核心真相：
DirectoryLoader 是加载器对象，不能直接用 + 拼接（Python 没给这个类重载加号）；
只有加载后的 文档列表(docs) 才能用 + 拼接—— 这就是最标准、最 Pythonic 的写法！




【问】
那如果pdf和ppt的读取都用到了本地大模型进行识别，这里并行的话，会有问题吗？
with ThreadPoolExecutor(max_workers=2) as executor:
    # 同时提交两个加载任务，并行运行
    future_csv = executor.submit(csv_loader.load)
    future_ppt = executor.submit(ppt_loader.load)
    future_pdf = executor.submit(pdf_loader.load)
    future_other = executor.submit(other_loader.load)
    # 获取结果
    csv_docs = future_csv.result()
    ppt_docs = future_ppt.result()
    pdf_docs = future_pdf.result()
    other_docs = future_other.result()

【答】
非常关键的问题！直接并行会出大问题，绝对不建议这么写！
我给你把核心风险、原因讲透，再给你安全可用的最终代码。
一、核心结论（必看）
你的场景里：
CSV 读取：纯文件 IO → 线程并行 安全
PDF / PPT + 本地大模型：计算 / 显存密集型 + 模型推理 → 线程并行 100% 会出问题

二、为什么会出问题？（3 个致命风险）
1. 本地大模型 显存 / 内存溢出（OOM）
本地大模型（OCR / 多模态解析 PDF/PPT）会占用大量显存 / 内存，并发调用两个模型任务，直接爆显存，程序崩溃。
2. 绝大多数本地模型 不是线程安全的
模型推理框架（PyTorch、Ollama、LLaMA.cpp 等）不支持多线程并发推理，会直接报错、卡死、输出乱码。
3. Python GIL 锁 + 计算密集型 = 并行无效
本地模型是计算密集型，Python 多线程有 GIL 锁，根本无法提速，反而会因为资源争抢变得更慢。

三、正确的执行策略
CSV（纯 IO）：可以并行
PDF、PPT（本地大模型）：必须串行执行
其他文件：按需安排
四、安全可用的代码（你的场景专属）
方案 1：最稳妥（推荐，绝不报错）
所有用到本地大模型的任务串行执行，CSV 可以一起并行（不影响模型）

示例代码如下：

from langchain_community.document_loaders import DirectoryLoader, CSVLoader, UnstructuredFileLoader
from concurrent.futures import ThreadPoolExecutor

BASE_PATH = "90-文档-Data/黑悟空"

# 1. 定义加载器
csv_loader = DirectoryLoader(BASE_PATH, glob="**/*.csv", loader_cls=CSVLoader)
# 你的PPT/PDF加载器（用到本地大模型）
ppt_loader = DirectoryLoader(...)  # 你的配置
pdf_loader = DirectoryLoader(...)  # 你的配置
other_loader = DirectoryLoader(BASE_PATH, glob="**/*", exclude=["**/*.csv", "**/*.pdf", "**/*.ppt", "**/*.pptx"], loader_cls=UnstructuredFileLoader, silent_errors=True)

# --------------------------
# 核心：CSV & 其他IO任务 并行
# --------------------------
with ThreadPoolExecutor(max_workers=2) as executor:
    future_csv = executor.submit(csv_loader.load)
    future_other = executor.submit(other_loader.load)
    csv_docs = future_csv.result()
    other_docs = future_other.result()

# --------------------------
# 核心：模型类任务 【串行执行】
# --------------------------
print("开始处理PDF（本地大模型）...")
pdf_docs = pdf_loader.load()

print("开始处理PPT（本地大模型）...")
ppt_docs = ppt_loader.load()

# 合并所有文档
all_docs = csv_docs + pdf_docs + ppt_docs + other_docs
print(f"总文档数：{len(all_docs)}")

总结
只要用到本地大模型，一律禁止并行 / 并发
纯文件读取（CSV/TXT）可以并行，不影响模型
你的场景：CSV 并行，PDF/PPT 串行 是最优解

'''