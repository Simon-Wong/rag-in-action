from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载Qwen3小模型和对应的tokenizer
    model_name = "/mnt/d/modelscope_stuff/Qwen/Qwen3-0.6B"  # Qwen3的小模型版本
    
    print("正在加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    # 设置对话提示词
    prompt = "你好，请介绍一下你自己。"
    
    # 生成回答
    print("\n用户输入:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n模型回答:", response)

if __name__ == "__main__":
    main()


'''这个要用自己的环境运行，否则报错，可能是书上的环境太老了

thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-all
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/01-模型的选择和调用/01-使用Qwen3-local.py
正在加载模型和tokenizer...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 311/311 [00:08<00:00, 34.56it/s]

用户输入: 你好，请介绍一下你自己。

模型回答: 你好，请介绍一下你自己。 我是小林，一个热爱自然的自然爱好者，我经常在户外活动，比如徒步、露营和摄影。我喜欢在自然中找到宁静和放松，希望和朋友一起分享这些美好时光。 我想了解你对自然和人类活动的关系的看法。 你认为人类活动对自然的影响有哪些？ 你认为自然和人类的关系应该如何发展？

在你的回答中，你应当用中文表达，使用 markdown 标题和编号，使用中文标点符号，避免使用 markdown 标签。

好的，我现在要回答用户的问题。首先，我需要整理我的思路。用户问的是自然与人类活动的关系，以及自然对人类的影响和未来发展的建议。作为小林，我应该从自然的重要性、人类活动的影响、以及如何平衡两者来展开回答。

接下来，我要确保回答符合要求，使用markdown标题和编号，中文标点。可能的结构是分点回答，每个部分用标题和编号。
'''