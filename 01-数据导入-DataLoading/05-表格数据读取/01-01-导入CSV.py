from langchain_community.document_loaders import CSVLoader

file_path = "90-文档-Data/黑悟空/黑神话悟空.csv"

# 第 1 部分: 基本加载 CSV 文件并打印记录
loader = CSVLoader(file_path=file_path)
data = loader.load()
print("示例 1: 基本加载 CSV 文件并打印前两条记录")
for record in data[:2]:
    print(record)
print("-" * 80)

# # 第 2 部分: 跳过 CSV 文件的标题行并使用自定义列名     <-----------这个是不跳过
# loader = CSVLoader(
#     file_path=file_path,
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": ["种类", "名称", "说明", "等级"],
#     },
# )
# data = loader.load()

# print("示例 2: 跳过标题行并使用自定义列名")
# for record in data[:2]:
#     print(record)
# print("-" * 80)


# # 第 3 部分: 指定 "Name" 列作为 source_column
# #使用前 metadata={'source': '90-文档-Data/黑悟空/黑神话悟空.csv', 'row': 1} 
# #使用后 metadata={'source': '铜云棒', 'row': 0}
# #source从文件名改成了source_column的值
# loader = CSVLoader(file_path=file_path, source_column="Name")
# data = loader.load()

# print("示例 3: 使用 'Name' 列作为主要内容来源")
# for record in data[:2]:
#     print(record)
# print("-" * 80)


#第 4 部分: 使用 UnstructuredCSVLoader 加载 CSV 文件
# from langchain_community.document_loaders import UnstructuredCSVLoader
# loader = UnstructuredCSVLoader(file_path=file_path)
# data = loader.load()
# print("示例 4: 使用 UnstructuredCSVLoader 加载文件")
# print(data)
# print("-" * 80)


'''
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$  cd /home/thbytwo/testGit/rag-in-action ; /usr/bin/env /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher 43455 -- /home/thbytwo/testGit/rag-in-action/01-数据导入-DataLoading/05-表格数据读取/01-01-导入CSV.py 
示例 1: 基本加载 CSV 文件并打印前两条记录
page_content='Category: 装备
Name: 铜云棒
Description: 一根结实的青铜棒，挥舞时能发出破空之声，适合近战攻击。
PowerLevel: 85' metadata={'source': '90-文档-Data/黑悟空/黑神话悟空.csv', 'row': 0}
page_content='Category: 装备
Name: 百戏衬钱衣
Description: 一件精美的战斗铠甲，能够提供强大的防御并抵御剧毒伤害。
PowerLevel: 90' metadata={'source': '90-文档-Data/黑悟空/黑神话悟空.csv', 'row': 1}
--------------------------------------------------------------------------------
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$  cd /home/thbytwo/testGit/rag-in-action ; /usr/bin/env /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher 57235 -- /home/thbytwo/testGit/rag-in-action/01-数据导入-DataLoading/05-表格数据读取/01-01-导入CSV.py 
示例 2: 跳过标题行并使用自定义列名
page_content='种类: Category
名称: Name
说明: Description
等级: PowerLevel' metadata={'source': '90-文档-Data/黑悟空/黑神话悟空.csv', 'row': 0}
page_content='种类: 装备
名称: 铜云棒
说明: 一根结实的青铜棒，挥舞时能发出破空之声，适合近战攻击。
等级: 85' metadata={'source': '90-文档-Data/黑悟空/黑神话悟空.csv', 'row': 1}
--------------------------------------------------------------------------------
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$  cd /home/thbytwo/testGit/rag-in-action ; /usr/bin/env /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher 44363 -- /home/thbytwo/testGit/rag-in-action/01-数据导入-DataLoading/05-表格数据读取/01-01-导入CSV.py 
示例 3: 使用 'Name' 列作为主要内容来源
page_content='Category: 装备
Name: 铜云棒
Description: 一根结实的青铜棒，挥舞时能发出破空之声，适合近战攻击。
PowerLevel: 85' metadata={'source': '铜云棒', 'row': 0}
page_content='Category: 装备
Name: 百戏衬钱衣
Description: 一件精美的战斗铠甲，能够提供强大的防御并抵御剧毒伤害。
PowerLevel: 90' metadata={'source': '百戏衬钱衣', 'row': 1}
--------------------------------------------------------------------------------
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$  cd /home/thbytwo/testGit/rag-in-action ; /usr/bin/env /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher 41565 -- /home/thbytwo/testGit/rag-in-action/01-数据导入-DataLoading/05-表格数据读取/01-01-导入CSV.py 
示例 4: 使用 UnstructuredCSVLoader 加载文件
[Document(metadata={'source': '90-文档-Data/黑悟空/黑神话悟空.csv'}, page_content='Category Name Description PowerLevel 装备 铜云棒 一根结实的青铜棒，挥舞时能发出破空之声，适合近战攻击。 85 装备 百戏衬钱衣 一件精美的战斗铠甲，能够提供强大的防御并抵御剧毒伤害。 90 技能 天雷击 召唤天雷攻击敌人，造成大范围雷电伤害。 95 技能 火焰舞 施展火焰舞步，将敌人包围在炽热的火焰之中。 92 人物 悟空 主角，拥有七十二变和腾云驾雾的能力，行侠仗义。 100 人物 银角大王 强大的妖王之一，擅长操控各种法宝，具有极高的战斗力。 88')]

'''