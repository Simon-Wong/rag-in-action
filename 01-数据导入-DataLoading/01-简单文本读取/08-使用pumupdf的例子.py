import pymupdf

doc=pymupdf.open('90-文档-Data/黑悟空/黑神话悟空.pdf')
texts= [page.get_textpage().extractText() for page in doc]

for text in texts:
    print(text)
    print('='*20)
