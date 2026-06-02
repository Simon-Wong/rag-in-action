#pip install marker-pdf

import os
import subprocess

def convert_pdf_to_markdown(input_pdf_path,output_folder,batch_multiplier=2,max_pages=12):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    command=[
        'marker_single',
        input_pdf_path,
        output_folder,
        f'--batch_multiplier={batch_multiplier}',
        f'--max_pages={max_pages}'
    ]

    try:
        subprocess.run(command,check=True)
        print(f"PDF文档转换为Markdown格式成功，文件已保存到{output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"PDF文档转换失败：{e}")

if __name__=="__main__":
    input_pdf_path="90-文档-Data/山西文旅/云冈石窟-en.pdf"
    output_floder="output/云冈石窟-en"

    convert_pdf_to_markdown(input_pdf_path,output_floder)


