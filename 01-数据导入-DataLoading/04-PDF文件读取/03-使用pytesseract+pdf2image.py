# 扫描图片型 PDF，建议用 pytesseract + pdf2image  
# sudo apt-get install tesseract-ocr
# sudo apt-get install tesseract-ocr-chi-sim

import pdf2image
import pytesseract
import os

# 创建 output 目录
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 将 PDF 转换为图片并保存
images = pdf2image.convert_from_path('90-文档-Data/黑悟空/黑神话悟空.pdf')
for i, image in enumerate(images):
    image.save(f'{output_dir}/page_{i+1}.png')

# 使用 pytesseract 提取文本
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang='chi_sim')
    print(f"第 {i+1} 页文本:")
    print(text)
    print("\n") 


'''

(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/01-数据导入-DataLoading/04-PDF文件读取/03-使用pytesseract+pdf2image.py
第 1 页文本:
山

，申 国 故事

GAME SCIENCE

 游戏科学

08. 20

直面天

命



第 2 页文本:
-De

和
|

黑神话: 悟空》是一款基于《西游记》改编的中国神话动作角色扮演游戏，玩家化身

“天命志人”》，在险象环生的西游冒险中追寻传说彰后的秘密。

中



第 3 页文本:



第 4 页文本:
雄奇壮丽，光怪陆离
重走西游故地，再写神话结局

-守

多

有  全
YY
Re 3  2
Ten/

玉     人 SS

1

四




第 5 页文本:
二
一你,也想当神仙?

坦

气洒几请 1 沉

六

用

放 疏过二-寿

页 妈

人

“ 法 主盖库

人
-二

好



(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''