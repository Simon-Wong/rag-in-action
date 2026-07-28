from pydantic import BaseModel, Field
from typing import List, Optional
from llama_index.program.openai import OpenAIPydanticProgram

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

embed_ollama = OllamaEmbedding(
    model_name="nomic-embed-text",   # 确保与 Ollama 下载的模型名完全一致
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
)

llm_ollama = Ollama(
    model="qwen3:14b",                # 确保与 Ollama 下载的模型名完全一致qwen3.5:9b
    base_url="http://192.168.0.119:11434",  # Ollama 服务地址
    request_timeout=120.0,           # (可选) 设置请求超时，单位秒
)

Settings.embed_model = embed_ollama
Settings.llm = llm_ollama

# 定义代码问题模型
class CodeIssue(BaseModel):
    """代码中存在的问题"""
    line_number: int = Field(..., description="问题所在的行号")
    issue_type: str = Field(..., description="问题类型，如：安全漏洞、性能问题、代码风格等")
    description: str = Field(..., description="问题的详细描述")
    severity: str = Field(..., description="问题严重程度：high/medium/low")

# 定义代码分析报告模型
class CodeAnalysis(BaseModel):
    """代码分析报告"""
    file_name: str = Field(..., description="被分析的文件名")
    issues: List[CodeIssue] = Field(default_factory=list, description="发现的问题列表")
    overall_quality: str = Field(..., description="代码整体质量评估：excellent/good/fair/poor")
    recommendations: List[str] = Field(default_factory=list, description="改进建议")

# 创建 OpenAI Pydantic Program
program = OpenAIPydanticProgram.from_defaults(
    output_cls=CodeAnalysis,
    prompt_template_str="""
请分析以下代码，生成详细的分析报告：
{code}

要求：
1. 识别代码中的潜在问题
2. 评估代码质量
3. 提供改进建议
""",
    verbose=True
)

# 示例代码
sample_code = """
def process_data(data):
    if data is None:
        return
    for item in data:
        if item > 100:
            print("Large value found")
        else:
            print("Small value")
"""

# 运行分析
try:
    analysis = program(code=sample_code)
    
    print(f"文件分析报告: {analysis.file_name}")
    print(f"整体质量: {analysis.overall_quality}")
    
    print("\n发现的问题:")
    for issue in analysis.issues:
        print(f"- 行号 {issue.line_number}: {issue.issue_type}")
        print(f"  描述: {issue.description}")
        print(f"  严重程度: {issue.severity}")
    
    print("\n改进建议:")
    for rec in analysis.recommendations:
        print(f"- {rec}")
        
except Exception as e:
    print(f"分析过程中出现错误: {e}")


'''
pip install -U llama-index llama-index-program-openai llama-index-agent-openai llama-index-llms-ollama llama-index-embeddings-ollama


thbytwo@thbytwopower:~/testGit/rag-in-action$  conda activate venv-rag-all
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ /home/thbytwo/miniforge3/envs/venv-rag-all/bin/python /home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/04-Pydantic-v2-ollama.py
/home/thbytwo/miniforge3/envs/venv-rag-all/lib/python3.10/site-packages/pydantic/_internal/_generate_schema.py:2274: UnsupportedFieldAttributeWarning: The 'validate_default' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'validate_default' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
Traceback (most recent call last):
  File "/home/thbytwo/testGit/rag-in-action/08-响应生成-Generation/03-通过输出解析控制格式/04-Pydantic-v2-ollama.py", line 40, in <module>
    program = OpenAIPydanticProgram.from_defaults(
  File "/home/thbytwo/miniforge3/envs/venv-rag-all/lib/python3.10/site-packages/llama_index/program/openai/base.py", line 124, in from_defaults
    raise ValueError(
ValueError: OpenAIPydanticProgram only supports OpenAI LLMs. Got: <class 'llama_index.llms.ollama.base.Ollama'>

这个只支持openai自己的模型

'''