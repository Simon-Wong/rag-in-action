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

# Settings.embed_model = embed_ollama
# Settings.llm = llm_ollama

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

# 使用 LLMTextCompletionProgram
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser

# 1. 创建 Pydantic 输出解析器，它会自动生成 JSON schema
parser = PydanticOutputParser(output_cls=CodeAnalysis)
json_schema_str = parser.get_format_string()  # 获取格式说明，用于提示词

# 2. 构建 Program，明确指定 llm 和 output_parser
program = LLMTextCompletionProgram.from_defaults(
    output_parser=parser,
    llm=llm_ollama,   # 指定你的 Ollama 模型
    prompt_template_str=f"""
请分析以下代码，生成详细的分析报告：
{{code}}

你必须严格按照下面的 JSON 格式输出，**不要包含任何其他文字、解释或代码块标记**，直接输出合法的 JSON 对象。

{json_schema_str}
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
文件分析报告: example.py
整体质量: good

发现的问题:
- 行号 4: security
  描述: 未验证data参数的可迭代性，若传入非迭代对象（如整数）会导致运行时错误
  严重程度: high
- 行号 5: functionality
  描述: 未处理item可能为非数值类型的情况，比较运算可能引发异常
  严重程度: medium
- 行号 3: code_style
  描述: 缺少对data参数为空列表的显式处理，虽然逻辑上不会进入循环，但可增加注释说明
  严重程度: low

改进建议:
- 添加类型检查：if not isinstance(data, Iterable): return
- 将print语句替换为logging模块实现更可控的日志输出
- 增加异常处理机制捕获潜在的比较异常
- 为函数添加类型注解（如def process_data(data: Optional[Iterable])）
(venv-rag-all) thbytwo@thbytwopower:~/testGit/rag-in-action$ 

'''