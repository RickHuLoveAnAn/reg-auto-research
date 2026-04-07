# scoring_logic.py
# 可进化评分逻辑 —— Agent 可以修改此文件的 dimensions 和 aggregation
# 每次迭代后，如果 τ 提升则 commit，退步则 git checkout 丢弃改动

dimensions = [
    {
        "name": "解读准确性",
        "weight": 0.42857142857142855,
        "prompt": "你是一位资深金融合规官。请评估以下文章对监管规定的核心条款解读是否准确。",
        "rubric": "完全准确=10分，部分准确=6分，存在误解=2分，有严重错误=0分"
    },
    {
        "name": "实操建议清晰度",
        "weight": 0.23809523809523808,
        "prompt": "你是一位资深金融合规官。请评估以下文章提供的落地执行建议是否具体、可操作。",
        "rubric": "建议具体可落地=10分，建议较具体=7分，建议模糊=4分，无实操建议=0分"
    },
    {
        "name": "风险提示完整性",
        "weight": 0.19047619047619047,
        "prompt": "你是一位资深金融合规官。请评估以下文章是否完整提示了新规的主要风险点和潜在合规隐患。",
        "rubric": "风险提示全面=10分，部分覆盖=6分，仅提及=3分，未提及=0分"
    },
    {
        "name": "文章结构与可读性",
        "weight": 0.14285714285714285,
        "prompt": "你是一位资深金融合规官。请评估以下文章的逻辑结构是否清晰、层次是否分明、表述是否流畅。",
        "rubric": "结构清晰流畅=10分，结构较清晰=7分，结构混乱=3分"
    },
    {
        "name": "时效性与完整性",
        "weight": 0.0,
        "prompt": "你是一位资深金融合规官。请评估以下文章是否覆盖了新规的核心要点，是否有时效性相关的说明。",
        "rubric": "覆盖全面且时效性强=10分，较全面=7分，有遗漏=4分，严重遗漏=0分"
    },
]

aggregation = "weighted_sum"  # 可选: "weighted_sum", "llm_judge"

def build_score_prompt(article_text: str, dimension: dict) -> str:
    """构建单个维度的评分 Prompt"""
    return f"""
{article_text}

---
{dimension['prompt']}

请根据以下标准评分：
{dimension['rubric']}

请直接输出一个 0-10 的数字分数，不要输出其他内容。
"""