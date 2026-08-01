import os
from dotenv import load_dotenv
load_dotenv() # 加载.env文件中的环境变量
import numpy as np
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate

# 准备评估用的LLM（使用GPT-3.5）
from openai import OpenAI

# from ragas.llms import llm_factory
# llm = llm_factory('llama3.2:3b', client = OpenAI(base_url="http://192.168.0.119:11434/v1",
#                                                 api_key="ollama"))
llm = LangchainLLMWrapper(ChatOpenAI(
    model="deepseek-r1:14b",#"deepseek-r1:1.5b ", #"llama3.2:3b",
    base_url="http://192.168.0.119:11434/v1",
    api_key="ollama"
))


# 准备数据集
# 这个数据集包含了问题、生成的答案以及相关的上下文信息
data = {
    "question": [
        "Who is the main character in Black Myth: Wukong?",
        "What are the special features of the combat system in Black Myth: Wukong?",
        "How is the visual quality of Black Myth: Wukong?",
    ],
    "answer": [
        "The main character in Black Myth: Wukong is Sun Wukong, based on the Chinese classic 'Journey to the West' but with a new interpretation. This version of Sun Wukong is more mature and brooding, showing a different personality from the traditional character.",
        "Black Myth: Wukong's combat system combines Chinese martial arts with Soulslike game features, including light and heavy attack combinations, technique transformations, and magic systems. Notably, Wukong can transform between different weapon forms during combat, such as his iconic staff and nunchucks, and use various mystical abilities.",
        "Black Myth: Wukong is developed using Unreal Engine 5, showcasing stunning visual quality. The game's scene modeling, lighting effects, and character details are all top-tier, particularly in its detailed recreation of traditional Chinese architecture and mythological settings.",
    ],
    "contexts": [
        [
            "Black Myth: Wukong is an action RPG developed by Game Science, featuring Sun Wukong as the protagonist based on 'Journey to the West' but with innovative interpretations. In the game, Wukong has a more composed personality and carries a special mission.",
            "The game is set in a mythological world, telling a new story that presents a different take on the traditional Sun Wukong character."
        ],
        [
            "The game's combat system is heavily influenced by Soulslike games while incorporating traditional Chinese martial arts elements. Players can utilize different weapon forms, including the iconic staff and other transforming weapons.",
            "During combat, players can unleash various mystical abilities, combined with light and heavy attacks and combo systems, creating a fluid and distinctive combat experience. The game also features a unique transformation system."
        ],
        [
            "Black Myth: Wukong demonstrates exceptional visual quality, built with Unreal Engine 5, achieving extremely high graphical fidelity. The game's environments and character models are meticulously crafted.",
            "The lighting effects, material rendering, and environmental details all reach AAA-level standards, perfectly capturing the atmosphere of an Eastern mythological world."
        ]
    ]
}

# 将字典转换为Hugging Face的Dataset对象，方便Ragas处理
dataset = Dataset.from_dict(data)

print("\n=== Ragas评估指标说明 ===")
print("\n1. Faithfulness（忠实度）")
print("- 评估生成的答案是否忠实于上下文内容")
print("- 通过将答案分解为简单陈述，然后验证每个陈述是否可以从上下文中推断得出")
print("- 该指标仅依赖LLM，不需要embedding模型")

# 评估Faithfulness
# 创建Faithfulness评估指标，它只需要一个LLM来进行评估
faithfulness_metric = [Faithfulness(llm=llm)] # 只需要提供生成模型
print("\n正在评估忠实度...")
# 使用evaluate函数对数据集进行评估
faithfulness_result = evaluate(dataset, faithfulness_metric)
# 提取忠实度分数
scores = faithfulness_result['faithfulness']
# 计算平均分
mean_score = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"忠实度评分: {mean_score:.4f}")




print("\n2. AnswerRelevancy（答案相关性）")
print("- 评估生成的答案与问题的相关程度")
print("- 使用embedding模型计算语义相似度")
print("- 我们将比较开源 bge-m3 模型和 qwen3-embedding:0.6b 模型")

# 设置两种embedding模型
# from langchain_community.embeddings import OllamaEmbeddings
# OLLAMA_BASE_URL = "http://192.168.0.119:11434"
# EMBED_MODEL_NAME_bge = "bge-m3:latest"
# EMBED_MODEL_NAME_qwen3 = "qwen3-embedding:0.6b"

# embed_model_ollama_bge = OllamaEmbeddings(
#     model=EMBED_MODEL_NAME_bge,
#     base_url=OLLAMA_BASE_URL,
# )

# from ragas.embeddings import embedding_factory
# embedding_bge=embedding_factory('ollama', model=EMBED_MODEL_NAME_bge, client=embed_model_ollama_bge) 

# embed_model_ollama_qwen3 = OllamaEmbeddings(
#     model=EMBED_MODEL_NAME_qwen3,
#     base_url=OLLAMA_BASE_URL,
# )
# embedding_qwen3=embedding_factory('ollama', model=EMBED_MODEL_NAME_qwen3, client=embed_model_ollama_qwen3) 

from langchain_community.embeddings import OllamaEmbeddings

ollama_embed_bge = OllamaEmbeddings(model="bge-m3:latest", base_url="http://192.168.0.119:11434")
ollama_embed_qwen3 = OllamaEmbeddings(model="qwen3-embedding:0.6b", base_url="http://192.168.0.119:11434")

embedding_bge = LangchainEmbeddingsWrapper(ollama_embed_bge)
embedding_qwen3 = LangchainEmbeddingsWrapper(ollama_embed_qwen3)

# 创建答案相关性评估指标
# 分别为两种embedding模型创建AnswerRelevancy评估指标
relevancy_bge = [AnswerRelevancy(llm=llm, embeddings=embedding_bge)]
relevancy_qwen3 = [AnswerRelevancy(llm=llm, embeddings=embedding_qwen3)]

print("\n正在评估答案相关性...")
print("\n使用bge-m3:latest模型评估:")
# 使用模型进行评估
result_bge = evaluate(dataset, relevancy_bge)
scores = result_bge['answer_relevancy']
mean_bge = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {mean_bge:.4f}")

print("\n使用qwen3-embedding:0.6b模型评估:")
# 使用模型进行评估
result_qwen3 = evaluate(dataset, relevancy_qwen3)
scores = result_qwen3['answer_relevancy']
mean_qwen3 = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {mean_qwen3:.4f}")

# 比较两种embedding模型的结果
print("\n=== Embedding模型比较 ===")
diff = mean_qwen3 - mean_bge
print(f"bge-m3:latest模型评分: {mean_bge:.4f}")
print(f"qwen3-embedding:0.6b模型评分: {mean_qwen3:.4f}")
print(f"差异: {diff:.4f} ({'qwen3-embedding:0.6b更好' if diff > 0 else 'bge-m3:latest模型更好' if diff < 0 else '相当'})")


'''
能跑，但分数是nan，没意义

'''