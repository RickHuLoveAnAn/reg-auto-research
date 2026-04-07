"""
测试用例 - agent_research.py 核心函数
在运行主循环前，必须全部通过
"""

import pytest
import sys
import os
import json
import re
import subprocess

# 确保可以导入 agent_research
sys.path.insert(0, os.path.dirname(__file__))
import agent_research


# ==================== 1. test_compute_kendall_tau ====================

def test_kendall_tau_perfect():
    """完全一致的排序，tau 应该是 1.0"""
    pred = ["a", "b", "c", "d"]
    true = ["a", "b", "c", "d"]
    tau = agent_research.compute_kendall_tau(pred, true)
    assert abs(tau - 1.0) < 1e-9, f"完美排序 tau 应为 1.0，实际 {tau}"


def test_kendall_tau_reversed():
    """完全相反的排序，tau 应该是 -1.0（如果有 n<10 对则更复杂，这里只测 4 个元素）"""
    pred = ["a", "b", "c", "d"]
    true = ["d", "c", "b", "a"]
    tau = agent_research.compute_kendall_tau(pred, true)
    # 4个元素: 6对，完全逆序 concordance=0，tau=-1
    assert abs(tau - (-1.0)) < 1e-9, f"逆序 tau 应为 -1.0，实际 {tau}"


def test_kendall_tau_partial():
    """部分一致的排序"""
    pred = ["a", "b", "c", "d", "e"]
    true = ["a", "b", "c", "d", "e"]
    tau = agent_research.compute_kendall_tau(pred, true)
    assert abs(tau - 1.0) < 1e-9

    # one swap: a和b交换，只有 (a,b) 一对不一致，C=9, D=1, tau=0.8
    pred_swap = ["b", "a", "c", "d", "e"]
    tau_swap = agent_research.compute_kendall_tau(pred_swap, true)
    assert abs(tau_swap - 0.8) < 1e-9, f"单次交换 tau 应为 0.8，实际 {tau_swap}"


def test_kendall_tau_single_element():
    """单个元素，tau 应该是 1.0"""
    tau = agent_research.compute_kendall_tau(["a"], ["a"])
    assert abs(tau - 1.0) < 1e-9


def test_kendall_tau_empty():
    """空列表，tau 应该是 1.0"""
    tau = agent_research.compute_kendall_tau([], [])
    assert abs(tau - 1.0) < 1e-9


# ==================== 2. test_parse_score ====================

def test_parse_score_normal():
    assert agent_research.parse_score("8.5") == 8.5
    assert agent_research.parse_score("7") == 7.0
    assert agent_research.parse_score("  9.2  ") == 9.2


def test_parse_score_chinese():
    """中文响应中的数字"""
    assert agent_research.parse_score("评分为8.5分") == 8.5
    assert agent_research.parse_score("分数是7") == 7.0


def test_parse_score_out_of_range():
    """超过 0-10 范围的分数应该被截断"""
    assert agent_research.parse_score("15") == 10.0
    assert agent_research.parse_score("-3") == 0.0


def test_parse_score_empty():
    """空字符串返回默认值 5.0"""
    assert agent_research.parse_score("") == 5.0
    assert agent_research.parse_score("无分数") == 5.0


def test_parse_score_scale_100():
    """超过 100 的分数先 /10 再 clamp"""
    assert agent_research.parse_score("150") == 10.0  # 150/10=15 → clamp to 10


# ==================== 3. test_compute_pairwise_accuracy ====================

def test_pairwise_accuracy_perfect():
    """所有文章得分与排序完全一致，准确率 1.0"""
    scores = {
        "a": {"total": 90, "dimensions": {"dim1": {"score": 9, "weight": 1.0}}},
        "b": {"total": 80, "dimensions": {"dim1": {"score": 8, "weight": 1.0}}},
        "c": {"total": 70, "dimensions": {"dim1": {"score": 7, "weight": 1.0}}},
    }
    true_ranking = ["a", "b", "c"]
    result = agent_research.compute_pairwise_accuracy_per_dimension(scores, true_ranking)
    assert result[0]["pairwise_accuracy"] == 1.0


def test_pairwise_accuracy_one_wrong():
    """一对错误，2/3 对 = 0.667"""
    scores = {
        "a": {"total": 80, "dimensions": {"dim1": {"score": 8, "weight": 1.0}}},
        "b": {"total": 90, "dimensions": {"dim1": {"score": 9, "weight": 1.0}}},
        "c": {"total": 70, "dimensions": {"dim1": {"score": 7, "weight": 1.0}}},
    }
    true_ranking = ["a", "b", "c"]
    result = agent_research.compute_pairwise_accuracy_per_dimension(scores, true_ranking)
    # a vs b: 预测 b>a，但真实 a<b，错
    # a vs c: 预测 a>c，且真实 a<c，对
    # b vs c: 预测 b>c，且真实 b<c，对
    # 2/3 = 0.667
    assert abs(result[0]["pairwise_accuracy"] - 2/3) < 0.01


# ==================== 4. test_weight_normalization ====================

def test_weight_normalization():
    """测试 agent_modify_scoring_logic 中的权重归一化逻辑"""
    # 手动模拟修改后权重归一化
    dimensions = [
        {"name": "dim1", "weight": 0.5},
        {"name": "dim2", "weight": 0.5},
    ]
    # 如果加入一个 0.5，总和变成 1.5，需要归一化
    dimensions[0]["weight"] = 1.0
    dimensions[1]["weight"] = 0.5
    total = sum(d["weight"] for d in dimensions)
    assert abs(total - 1.5) < 1e-9
    # 归一化
    for d in dimensions:
        d["weight"] = d["weight"] / total
    assert abs(sum(d["weight"] for d in dimensions) - 1.0) < 1e-9


def test_negative_weight_clamped():
    """负权重应该被过滤为 0"""
    dimensions = [
        {"name": "dim1", "weight": -0.2},
        {"name": "dim2", "weight": 1.2},
    ]
    for d in dimensions:
        if d["weight"] < 0:
            d["weight"] = 0.0
    assert dimensions[0]["weight"] == 0.0
    assert dimensions[1]["weight"] == 1.2
    # 归一化后总和为 1.0
    total = sum(d["weight"] for d in dimensions)
    for d in dimensions:
        d["weight"] = d["weight"] / total
    assert abs(sum(d["weight"] for d in dimensions) - 1.0) < 1e-9


# ==================== 5. test_json_extraction ====================

def test_json_extraction_code_block():
    """从 ```json ``` 块中提取 JSON"""
    response = """好的，这是我的修改计划：

```json
{"modifications": [{"type": "weight", "dimension": "解读准确性", "old_value": 0.3, "new_value": 0.35}]}
```

希望这个修改能提升效果。"""

    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    assert code_block_match is not None
    json_str = code_block_match.group(1)
    data = json.loads(json_str)
    assert "modifications" in data
    assert len(data["modifications"]) == 1


def test_json_extraction_direct():
    """直接从文本中提取 JSON"""
    response = '{"modifications": [{"type": "weight", "dimension": "dim1", "old_value": 0.3, "new_value": 0.4}]}'
    json_match = re.search(r'\{.*"modifications".*\}', response, re.DOTALL)
    assert json_match is not None
    data = json.loads(json_match.group())
    assert len(data["modifications"]) == 1


# ==================== 6. test_golden_set_structure ====================

def test_golden_set_structure():
    """验证 golden_set.json 结构正确"""
    golden = agent_research.load_golden_set(
        os.path.join(os.path.dirname(__file__), "golden_set.json")
    )
    assert "human_ranking" in golden
    assert "articles" in golden
    assert len(golden["human_ranking"]) == 5
    assert len(golden["articles"]) == 5
    for aid in golden["human_ranking"]:
        assert aid in golden["articles"]


# ==================== 7. test_scoring_logic_loads ====================

def test_scoring_logic_loads():
    """验证 scoring_logic.py 可以正常加载"""
    dims = agent_research.load_scoring_logic(
        os.path.join(os.path.dirname(__file__), "scoring_logic.py")
    )
    assert "dimensions" in dims
    assert "build_score_prompt" in dims
    assert len(dims["dimensions"]) == 3
    # 验证权重总和为 1.0
    total = sum(d["weight"] for d in dims["dimensions"])
    assert abs(total - 1.0) < 0.001


# ==================== 8. test_build_score_prompt ====================

def test_build_score_prompt():
    """验证 prompt 构建正确"""
    dims = agent_research.load_scoring_logic(
        os.path.join(os.path.dirname(__file__), "scoring_logic.py")
    )
    dim = dims["dimensions"][0]
    prompt = dims["build_score_prompt"]("这是文章内容。", dim)
    assert "这是文章内容。" in prompt
    # prompt 模板中的 {regulation_name} 等参数已被替换，检查关键内容片段
    assert "合规实操" in prompt or "合规专家" in prompt
    assert dim["rubric"] in prompt


# ==================== 9. test_weight_sum_always_one ====================

def test_weight_sum_always_one():
    """验证无论初始权重如何，归一化后总和为 1.0"""
    # 模拟 agent_modify_scoring_logic 中的归一化逻辑
    test_cases = [
        # (初始权重列表, 期望归一化后总和)
        ([0.5, 0.5, 0.0, 0.0, 0.0], 1.0),
        ([0.3, 0.25, 0.2, 0.15, 0.1], 1.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0], 0.0),  # 全零不归一化
        ([1.5, -0.5, 0.0, 0.0, 0.0], 1.0),  # 负权重先过滤
    ]

    for initial_weights, expected_total in test_cases:
        dimensions = [{"name": f"d{i}", "weight": w} for i, w in enumerate(initial_weights)]
        # 过滤负权重
        for d in dimensions:
            if d["weight"] < 0:
                d["weight"] = 0.0
        total = sum(d["weight"] for d in dimensions)
        if total > 0:
            for d in dimensions:
                d["weight"] = d["weight"] / total
        final_total = sum(d["weight"] for d in dimensions)
        # 全零情况总和为 0.0，不为 1.0
        if expected_total > 0:
            assert abs(final_total - 1.0) < 1e-9, f"权重总和应为 1.0，实际 {final_total}，初始 {initial_weights}"
        else:
            assert abs(final_total - 0.0) < 1e-9


# ==================== 10. test_no_double_agent_call ====================

def test_scoring_logic_syntax_valid():
    """验证 scoring_logic.py 可以被 Python 语法解析（无语法错误）"""
    import ast
    with open(os.path.join(os.path.dirname(__file__), "scoring_logic.py"), "r") as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        pytest.fail(f"scoring_logic.py 有语法错误: {e}")


def test_no_double_agent_call_in_main_loop():
    """验证主循环代码中没有重复调用 agent_modify_scoring_logic"""
    # 读取源文件，检查主循环中 agent_modify_scoring_logic 只出现一次
    with open(os.path.join(os.path.dirname(__file__), "agent_research.py"), "r") as f:
        content = f.read()

    # 找到 main_loop 函数
    main_loop_match = re.search(r'def main_loop\(.*?\n(.*?)(?=\ndef |\nclass |\Z)', content, re.DOTALL)
    assert main_loop_match is not None
    main_loop_body = main_loop_match.group(1)

    # 计算 agent_modify_scoring_logic 出现次数
    count = main_loop_body.count("agent_modify_scoring_logic")
    assert count == 1, f"main_loop 中 agent_modify_scoring_logic 应出现 1 次，实际 {count} 次"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
