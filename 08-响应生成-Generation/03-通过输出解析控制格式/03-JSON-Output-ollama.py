import json
from openai import OpenAI
from dotenv import load_dotenv  
import os

load_dotenv()

client = OpenAI(
    base_url="http://192.168.0.119:11434/v1",
    api_key="ollama"                       
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="qwen3:14b",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))

'''
thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-langchain
(venv-rag-langchain) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-langchain/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/03-JSON-Output-ollama.py
{'question': 'Which is the longest river in the world?', 'answer': 'The Nile River'}

'''