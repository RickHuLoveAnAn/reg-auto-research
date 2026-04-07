"""
agent_research.py - 核心循环引擎
自动评估、进化和优化金融合规文章评分系统
"""

import json
import os
import re
import requests
import subprocess
import datetime
import importlib.util
from itertools import combinations
from typing import Dict, List, Any, Optional

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    for line in open(_env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ==================== 配置 ====================

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== 1. 加载数据 ====================

def load_golden_set(path: str) -> dict:
    """加载 golden_set.json"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scoring_logic(path: str) -> dict:
    """动态加载 scoring_logic.py 作为 Python 模块"""
    spec = importlib.util.spec_from_file_location("scoring_logic_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "dimensions": module.dimensions,
        "aggregation": getattr(module, "aggregation", "weighted_sum"),
        "build_score_prompt": module.build_score_prompt
    }


# ==================== 2. API 调用 ====================

def call_qwen_api(prompt: str, model: str = "qwen-plus") -> str:
    """调用 DashScope API 获取评分"""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "parameters": {"temperature": 0.1, "max_tokens": 50}
    }
    response = requests.post(DASHSCOPE_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    resp_data = response.json()
    # 支持两种响应格式
    if "output" in resp_data and "text" in resp_data["output"]:
        return resp_data["output"]["text"].strip()
    elif "output" in resp_data and "choices" in resp_data["output"]:
        return resp_data["output"]["choices"][0]["message"]["content"].strip()
    elif "choices" in resp_data:
        return resp_data["choices"][0]["message"]["content"].strip()
    return str(resp_data)


# ==================== 3. 分数解析 ====================

def parse_score(raw_text: str) -> float:
    """从模型响应中提取第一个 0-10 的数字分数"""
    if not raw_text:
        return 5.0
    # 匹配第一个非负浮点数
    match = re.search(r"(\d+\.?\d*)", raw_text.strip())
    if match:
        score = float(match.group(1))
        # 如果分数超出范围，进行缩放
        if score > 10:
            score = score / 10  # 假设可能是 0-100 的分数
        return min(max(score, 0.0), 10.0)
    return 5.0  # 默认分数


# ==================== 4. 文章评分 ====================

def score_article(article_text: str, scoring_info: dict) -> dict:
    """对单篇文章进行多维度评分"""
    dimensions = scoring_info["dimensions"]
    build_score_prompt = scoring_info["build_score_prompt"]

    dimension_scores = {}

    for dim in dimensions:
        prompt = build_score_prompt(article_text, dim)
        try:
            raw_response = call_qwen_api(prompt)
            score = parse_score(raw_response)
        except Exception as e:
            print(f"    API 调用失败 ({dim['name']}): {e}")
            score = 5.0
            raw_response = str(e)

        dimension_scores[dim["name"]] = {
            "score": score,
            "weight": dim["weight"]
        }

        # 记录 API 调用
        log_api_call(
            iteration=getattr(log_iteration, '_current_iteration', 0),
            article_id=getattr(log_iteration, '_current_article_id', 'unknown'),
            dimension_name=dim["name"],
            prompt=prompt,
            raw_response=str(raw_response),
            parsed_score=score
        )

    # 计算加权总分 (0-100)
    total = sum(
        dim_scores["score"] * dim_scores["weight"]
        for dim_scores in dimension_scores.values()
    ) * 10  # 缩放到 0-100

    return {
        "total": total,
        "dimensions": dimension_scores
    }


# ==================== 5. Kendall's Tau ====================

def compute_kendall_tau(pred_ranking: List[str], true_ranking: List[str]) -> float:
    """计算 Kendall's Tau 排名相关系数"""
    pred_ranks = {aid: i for i, aid in enumerate(pred_ranking)}
    true_ranks = {aid: i for i, aid in enumerate(true_ranking)}

    pairs = list(combinations(pred_ranking, 2))
    if not pairs:
        return 1.0

    concordant_count = sum(
        1
        for a, b in pairs
        if (pred_ranks[a] - pred_ranks[b]) * (true_ranks[a] - true_ranks[b]) > 0
    )

    return concordant_count / len(pairs)


# ==================== 6. 成对准确率 ====================

def compute_pairwise_accuracy_per_dimension(
    scores: Dict[str, dict],
    true_ranking: List[str]
) -> List[dict]:
    """计算每个维度的成对准确率"""
    true_ranks = {aid: i for i, aid in enumerate(true_ranking)}
    article_ids = list(scores.keys())

    dim_names = list(scores[article_ids[0]]["dimensions"].keys())
    results = []

    for dim_name in dim_names:
        pairs = list(combinations(article_ids, 2))
        if not pairs:
            continue

        correct = 0
        for a, b in pairs:
            # 获取维度分数
            score_a = scores[a]["dimensions"][dim_name]["score"]
            score_b = scores[b]["dimensions"][dim_name]["score"]

            # 判断排名方向
            pred_a_before_b = score_a > score_b
            true_a_before_b = true_ranks[a] < true_ranks[b]  # 排名靠前 = 分数高

            if pred_a_before_b == true_a_before_b:
                correct += 1

        pairwise_acc = correct / len(pairs) if pairs else 0.0
        results.append({
            "name": dim_name,
            "pairwise_accuracy": pairwise_acc
        })

    return results


# ==================== 7. 偏差报告 ====================

def generate_deviation_report(
    iteration: int,
    tau: float,
    tau_delta: float,
    dimension_stats: List[dict]
) -> str:
    """生成偏差报告"""
    lines = [
        f"\n{'='*60}",
        f"迭代 {iteration} 偏差报告",
        f"{'='*60}",
        f"Kendall's Tau: {tau:.4f} (delta: {tau_delta:+.4f})",
        f"\n各维度成对准确率:"
    ]

    for dim_stat in dimension_stats:
        lines.append(f"  - {dim_stat['name']}: {dim_stat['pairwise_accuracy']:.4f}")

    # 找出最弱的维度
    if dimension_stats:
        weakest = min(dimension_stats, key=lambda x: x["pairwise_accuracy"])
        lines.append(f"\n最弱维度: {weakest['name']} ({weakest['pairwise_accuracy']:.4f})")

    lines.append("=" * 60)

    return "\n".join(lines)


# ==================== 8. 日志记录 ====================

def log_iteration(
    iteration: int,
    tau: float,
    tau_delta: float,
    dimension_stats: List[dict],
    changed_fields: List[str],
    commit_hash: str
) -> None:
    """追加迭代记录到 evolution_log.jsonl"""
    log_path = os.path.join(WORKING_DIR, "evolution_log.jsonl")

    record = {
        "iteration": iteration,
        "timestamp": datetime.datetime.now().isoformat(),
        "tau": round(tau, 4),
        "tau_delta": round(tau_delta, 4),
        "pairwise_accuracy": sum(d["pairwise_accuracy"] for d in dimension_stats) / len(dimension_stats) if dimension_stats else 0,
        "dimensions": [
            {
                "name": d["name"],
                "pairwise_acc": round(d["pairwise_accuracy"], 4)
            }
            for d in dimension_stats
        ],
        "changed_fields": changed_fields,
        "commit_hash": commit_hash
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_api_call(
    iteration: int,
    article_id: str,
    dimension_name: str,
    prompt: str,
    raw_response: str,
    parsed_score: float
) -> None:
    """追加 API 调用记录到 api_calls.log"""
    log_path = os.path.join(WORKING_DIR, "api_calls.log")

    log_entry = [
        f"=== Iteration {iteration} | Article: {article_id} | Dimension: {dimension_name} ===",
        "[REQUEST]",
        f"Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}",
        "[RESPONSE]",
        f"Raw: {raw_response[:500]}" if len(raw_response) > 500 else f"Raw: {raw_response}",
        f"Parsed: {parsed_score}",
        "==="
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(log_entry) + "\n\n")


# ==================== 9. Git 操作 ====================

def git_commit_if_improved(
    tau: float,
    tau_prev: float,
    iteration: int,
    changed_fields: List[str]
) -> Optional[str]:
    """如果 tau 提升则 commit scoring_logic.py，否则 checkout"""
    scoring_path = os.path.join(WORKING_DIR, "scoring_logic.py")

    if tau > tau_prev and changed_fields:
        # Commit
        try:
            subprocess.run(["git", "add", "scoring_logic.py"], cwd=WORKING_DIR, check=True)
            commit_msg = f"Iteration {iteration}: tau {tau_prev:.4f} -> {tau:.4f}. Changes: {', '.join(changed_fields)}"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=WORKING_DIR,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=WORKING_DIR,
                    capture_output=True,
                    text=True
                )
                commit_hash = hash_result.stdout.strip()
                print(f"  [Git] Committed: {commit_hash}")
                return commit_hash
        except Exception as e:
            print(f"  [Git] Commit failed: {e}")
    elif tau <= tau_prev:
        # Checkout to discard changes
        try:
            subprocess.run(["git", "checkout", "scoring_logic.py"], cwd=WORKING_DIR, check=True)
            print(f"  [Git] Tau did not improve, discarded changes")
        except Exception as e:
            print(f"  [Git] Checkout failed: {e}")

    return None


# ==================== 10. Agent 修改评分逻辑 ====================

def agent_modify_scoring_logic(deviation_report: str, iteration: int) -> List[str]:
    """调用 Qwen API 决定并应用评分逻辑的修改"""
    scoring_info = load_scoring_logic(os.path.join(WORKING_DIR, "scoring_logic.py"))
    dimensions = scoring_info["dimensions"]

    # 构建维度列表
    dimensions_list = "\n".join([
        f"- {d['name']}: weight={d['weight']}, prompt={d['prompt'][:50]}..."
        for d in dimensions
    ])

    # Agent prompt
    system_prompt = f"""你是一个金融合规评分专家。请根据以下 deviation 报告决定如何修改 scoring_logic.py。

当前 scoring_logic.py 有以下维度：
{dimensions_list}

deviation 报告：
{deviation_report}

请以 JSON 格式输出修改计划：
{{"modifications": [{{"type": "weight|prompt|rubric|add|remove", "dimension": "维度名", "old_value": ..., "new_value": ..., "reason": "..."}}]}}

注意：权重总和必须保持为 1.0。每次最多修改 2 个维度。"""

    try:
        response = call_qwen_api(system_prompt)

        # 解析 JSON 响应
        json_match = re.search(r'\{.*"modifications".*\}', response, re.DOTALL)
        if not json_match:
            print(f"  [Agent] 无法解析修改计划，使用默认计划")
            return []

        modifications = json.loads(json_match.group())

        if "modifications" not in modifications or not modifications["modifications"]:
            print(f"  [Agent] 无需修改")
            return []

        changed_fields = []

        # 应用修改
        for mod in modifications["modifications"]:
            mod_type = mod.get("type")
            dim_name = mod.get("dimension")
            new_value = mod.get("new_value")

            for dim in dimensions:
                if dim["name"] == dim_name:
                    if mod_type == "weight":
                        old_weight = dim["weight"]
                        new_weight = float(new_value)
                        # 过滤无效权重：非负数
                        if new_weight < 0:
                            print(f"  [Agent] 跳过无效权重 {dim_name}: {new_weight} < 0")
                            continue
                        dim["weight"] = new_weight
                        changed_fields.append(f"{dim_name} weight: {old_weight:.2f}->{new_weight:.2f}")
                    elif mod_type == "prompt":
                        dim["prompt"] = new_value
                        changed_fields.append(f"{dim_name} prompt updated")
                    elif mod_type == "rubric":
                        dim["rubric"] = new_value
                        changed_fields.append(f"{dim_name} rubric updated")

        # 重新计算权重确保总和为 1.0，并过滤负权重
        for d in dimensions:
            if d["weight"] < 0:
                d["weight"] = 0.0
        total_weight = sum(d["weight"] for d in dimensions)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
            print(f"  [Agent] 权重归一化: sum={total_weight:.4f} -> 1.0")
            for d in dimensions:
                d["weight"] = d["weight"] / total_weight

        # 写入修改后的 scoring_logic.py
        _write_scoring_logic(os.path.join(WORKING_DIR, "scoring_logic.py"), scoring_info)

        print(f"  [Agent] 应用了 {len(changed_fields)} 项修改")
        return changed_fields

    except Exception as e:
        print(f"  [Agent] 修改失败: {e}")
        return []


def _write_scoring_logic(path: str, scoring_info: dict) -> None:
    """将评分逻辑写回 scoring_logic.py"""
    dimensions = scoring_info["dimensions"]
    aggregation = scoring_info.get("aggregation", "weighted_sum")

    lines = [
        "# scoring_logic.py",
        "# 可进化评分逻辑 —— Agent 可以修改此文件的 dimensions 和 aggregation",
        "# 每次迭代后，如果 τ 提升则 commit，退步则 git checkout 丢弃改动",
        "",
        "dimensions = ["
    ]

    for dim in dimensions:
        lines.append("    {")
        lines.append(f'        "name": "{dim["name"]}",')
        lines.append(f'        "weight": {dim["weight"]},')
        lines.append(f'        "prompt": "{dim["prompt"]}",')
        lines.append(f'        "rubric": "{dim["rubric"]}"')
        lines.append("    },")

    lines.append("]")
    lines.append("")
    lines.append(f'aggregation = "{aggregation}"  # 可选: "weighted_sum", "llm_judge"')
    lines.append("")
    lines.append("def build_score_prompt(article_text: str, dimension: dict) -> str:")
    lines.append('    """构建单个维度的评分 Prompt"""')
    lines.append('    return f"""')
    lines.append("{article_text}")
    lines.append("")
    lines.append("---")
    lines.append("{dimension['prompt']}")
    lines.append("")
    lines.append("请根据以下标准评分：")
    lines.append("{dimension['rubric']}")
    lines.append("")
    lines.append("请直接输出一个 0-10 的数字分数，不要输出其他内容。")
    lines.append('"""')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==================== 11. 保存最佳模型 ====================

def save_best_model(output_path: str, dims: dict, tau: float, iteration: int) -> None:
    """复制 scoring_logic.py 到 best_scoring_model.py 并添加元数据"""
    import shutil

    source = os.path.join(WORKING_DIR, "scoring_logic.py")
    dest = os.path.join(WORKING_DIR, output_path)

    # 读取源文件内容
    with open(source, "r", encoding="utf-8") as f:
        content = f.read()

    # 添加元数据
    metadata = f'''
# ============================================================
# best_scoring_model.py
# 元数据:
#   - tau: {tau:.4f}
#   - iteration: {iteration}
#   - saved_at: {datetime.datetime.now().isoformat()}
# ============================================================

'''

    with open(dest, "w", encoding="utf-8") as f:
        f.write(metadata + content)

    print(f"  [Model] 保存最佳模型到 {output_path}")


# ==================== 12. 主循环 ====================

def main_loop(max_iterations: int = 100) -> None:
    """主循环"""
    print("=" * 60)
    print("金融合规评分系统 - 自动进化引擎")
    print("=" * 60)

    golden_path = os.path.join(WORKING_DIR, "golden_set.json")
    scoring_path = os.path.join(WORKING_DIR, "scoring_logic.py")

    tau_prev = 0.0

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # 加载数据
        golden = load_golden_set(golden_path)
        dims = load_scoring_logic(scoring_path)

        # 评分所有文章
        scores = {}
        for aid, article in golden["articles"].items():
            print(f"  Scoring article: {aid}")
            # 设置当前上下文供 log_api_call 使用
            log_iteration._current_iteration = iteration
            log_iteration._current_article_id = aid

            article_scores = score_article(article["content"], dims)
            scores[aid] = article_scores

        # 计算排名
        pred_ranking = sorted(
            scores.keys(),
            key=lambda aid: scores[aid]["total"],
            reverse=True
        )

        # 计算 Kendall's Tau
        tau = compute_kendall_tau(pred_ranking, golden["human_ranking"])
        tau_delta = tau - tau_prev

        print(f"  Predicted ranking: {pred_ranking}")
        print(f"  True ranking: {golden['human_ranking']}")
        print(f"  Kendall's Tau: {tau:.4f} (delta: {tau_delta:+.4f})")

        # 计算各维度成对准确率
        dim_stats = compute_pairwise_accuracy_per_dimension(scores, golden["human_ranking"])

        # 打印偏差报告
        print(generate_deviation_report(iteration, tau, tau_delta, dim_stats))

        # 生成偏差报告并让 Agent 修改
        deviation_report = generate_deviation_report(iteration, tau, tau_delta, dim_stats)
        changed_fields = agent_modify_scoring_logic(deviation_report, iteration)

        # Git 操作（commit 当前状态，即产生 tau 的那个 scoring_logic）
        commit_hash = git_commit_if_improved(tau, tau_prev, iteration, changed_fields)

        # 记录迭代
        log_iteration(iteration, tau, tau_delta, dim_stats, changed_fields, commit_hash or "")

        # 保存最佳模型
        if commit_hash:
            save_best_model("best_scoring_model.py", dims, tau, iteration)

        tau_prev = tau

        # 生成偏差报告并让 Agent 修改
        deviation_report = generate_deviation_report(iteration, tau, tau_delta, dim_stats)
        changed_fields = agent_modify_scoring_logic(deviation_report, iteration)

        # 如果没有改进，等待下一次迭代
        if not changed_fields:
            print(f"  [Loop] 无需修改，进入下一迭代")

    print("\n" + "=" * 60)
    print("主循环完成")
    print("=" * 60)


# ==================== 入口 ====================

if __name__ == "__main__":
    main_loop()
