from llmlingua import PromptCompressor

# =============================================================================
# 1. 初始化 PromptCompressor 压缩器 (同您的代码)
# =============================================================================
llm_lingua = PromptCompressor(
    use_llmlingua2=True,
    model_name="/mnt/d/modelscope_stuff/microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    device_map="cuda"  # 使用GPU加速压缩
)

# =============================================================================
# 2. 模拟 RECOMP 流程：检索步骤
# =============================================================================
# RECOMP的第一步是检索出长文档。这里我们模拟图7-11中的 Nissan Xterra 案例。
# 假设检索系统已经找出了非常冗长、包含诸多无关信息的原始文档。
user_query = "When did they stop making the Nissan Xterra?"

retrieved_document = """
The Nissan Xterra is a sport utility vehicle manufactured by Nissan. 
It was moved from Smyrna, Tennessee, to Nissan's facility in Canton, Mississippi. 
Early US models include X, S and PRO-4X, with a choice of 5-speed manual or 4-speed automatic transmission. 
The Xterra was introduced in 1999 as a compact SUV, designed to compete with the Jeep Cherokee. 
Over the years, it gained a reputation for its rugged off-road capabilities and distinctive styling. 
In 2015, after 16 years of production and over half a million units sold, Nissan officially discontinued the Xterra in North America, 
largely due to stricter fuel economy regulations and changing market preferences toward crossovers. 
However, Nissan continued to sell the Xterra in other markets for a few more years. 
The car had a very loyal fan base, especially in the US market.
"""

# =============================================================================
# 3. RECOMP 核心步骤：使用压缩器进行压缩
# =============================================================================
# 在 RECOMP 中，压缩器会考虑用户的 query，选择性保留或重新生成对回答问题最有用的信息。
# 我们调用 compress_prompt_llmlingua2，将 问题 和 上下文 一起传入。
compressed_result = llm_lingua.compress_prompt_llmlingua2(
    context=retrieved_document,    # 刚刚检索出的长文档 (原始Token: ~180个)
    rate=0.4,                      # 压缩至原始长度的40% (类似书中提到的 749->58)
    force_tokens=['\n'],
    chunk_end_tokens=['。', '\n'],
    return_word_label=True,
    drop_consecutive=True
)

# 获取压缩后的核心摘要 (对应图中 58 Token 的压缩结果)
compressed_prompt = compressed_result['compressed_prompt']

print("\n======== 用户问题 ========")
print(user_query)
print("\n======== 原始检索文档 (冗余较多) ========")
print(retrieved_document)
print(f"原始 Token 数: {len(retrieved_document.split())}")
print("\n======== RECOMP 压缩后的摘要 (精简准确) ========")
print(compressed_prompt)
print(f"压缩后 Token 数: {len(compressed_prompt.split())}")
print(f"压缩比例: {compressed_result.get('rate', 'N/A')}")

'''
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python /home/thbytwo/testGit/rag-in-action/07-检索后处理-PostRetrieval/02-压缩/RECOMP-local.py
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:09<00:00, 41.65it/s]

======== 用户问题 ========
When did they stop making the Nissan Xterra?

======== 原始检索文档 (冗余较多) ========

The Nissan Xterra is a sport utility vehicle manufactured by Nissan. 
It was moved from Smyrna, Tennessee, to Nissan's facility in Canton, Mississippi. 
Early US models include X, S and PRO-4X, with a choice of 5-speed manual or 4-speed automatic transmission. 
The Xterra was introduced in 1999 as a compact SUV, designed to compete with the Jeep Cherokee. 
Over the years, it gained a reputation for its rugged off-road capabilities and distinctive styling. 
In 2015, after 16 years of production and over half a million units sold, Nissan officially discontinued the Xterra in North America, 
largely due to stricter fuel economy regulations and changing market preferences toward crossovers. 
However, Nissan continued to sell the Xterra in other markets for a few more years. 
The car had a very loyal fan base, especially in the US market.

原始 Token 数: 136

======== RECOMP 压缩后的摘要 (精简准确) ========

 Nissan Xterra sport utility
 moved Smyrna Tennessee Canton Mississippi
 Early models X S PRO-4X 5-speed manual 4-speed transmission
 Xterra introduced 1999 compact SUV Jeep Cherokee
 rugged off-road capabilities distinctive styling
 2015, 16 years half million units sold discontinued Xterra North America
 fuel economy
 Xterra markets
 loyal fan base US

压缩后 Token 数: 49
压缩比例: 41.7%
'''