# 自进化评分引擎 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：**构建一个自主进化的 AI 评分引擎，对金融监管新规解读文章评分，通过 golden dataset + 闭环迭代让模型评分接近资深合规官。

**Architecture：**以 `agent_research.py` 为固定主循环，`scoring_logic.py` 为可进化单元（维度+权重+Prompt），每次迭代用 Qwen API 打分，计算 Kendall's τ 排名相关性，进化策略采用带方向性的 Mutation（优先修改拖后腿的维度）。

**Tech Stack：**Python 3.x / DashScope API (Qwen3.6-Plus) / Git / 标准库（json, itertools, subprocess, datetime）

---

## 文件结构

```
auto_research/
├── program.md              # Task 4：人类编写研究方向指引
├── golden_set.json         # Task 2：5篇高质量文章 + 人工排序
├── scoring_logic.py        # Task 3：可进化评分逻辑（初始版本）
├── agent_research.py       # Task 5：主循环引擎
└── best_scoring_model.py   # Task 6：最终最优模型
```

日志文件（由 Task 5 在运行时创建）：
```
evolution_log.jsonl    # 每次迭代的结构化结果
api_calls.log          # Qwen API 调用记录
```

---

## Task 1: 创建目录结构

**Files:**
- Create: `auto_research/` 目录（空目录占位）

- [ ] **Step 1: 创建目录**

Run: `mkdir -p /Users/rick/Projects/01_work/05_auto_research/auto_research`

---

## Task 2: 创建 golden_set.json 模板

**Files:**
- Create: `auto_research/golden_set.json`

- [ ] **Step 1: 创建带 PLACEHOLDER 的 golden_set.json**

```json
{
  "regulation": "PLACEHOLDER_监管外规名称",
  "description": "PLACEHOLDER_外规简要描述（如：2024年某月发布的关于XXX的监管规定）",
  "human_ranking": [
    "article_id_1",
    "article_id_2",
    "article_id_3",
    "article_id_4",
    "article_id_5"
  ],
  "articles": {
    "article_id_1": {
      "title": "PLACEHOLDER_文章1标题",
      "content": "PLACEHOLDER_文章1完整正文（文本内容）",
      "human_score": 92
    },
    "article_id_2": {
      "title": "PLACEHOLDER_文章2标题",
      "content": "PLACEHOLDER_文章2完整正文",
      "human_score": 85
    },
    "article_id_3": {
      "title": "PLACEHOLDER_文章3标题",
      "content": "PLACEHOLDER_文章3完整正文",
      "human_score": 78
    },
    "article_id_4": {
      "title": "PLACEHOLDER_文章4标题",
      "content": "PLACEHOLDER_文章4完整正文",
      "human_score": 70
    },
    "article_id_5": {
      "title": "PLACEHOLDER_文章5标题",
      "content": "PLACEHOLDER_文章5完整正文",
      "human_score": 62
    }
  }
}
```

> **注意**：`human_ranking` 顺序为从高到低的排序（article_id_1 排第一 = 最高分），`human_score` 为 100 分制分值仅供参考。

---

## Task 3: 创建 scoring_logic.py（初始版本）

**Files:**
- Create: `auto_research/scoring_logic.py`

- [ ] **Step 1: 编写初始评分逻辑**

```python
# scoring_logic.py
# 可进化评分逻辑 —— Agent 可以修改此文件的 dimensions 和 aggregation
# 每次迭代后，如果 τ 提升则 commit，退步则 git checkout 丢弃改动

dimensions = [
    {
        "name": "解读准确性",
        "weight": 0.30,
        "prompt": "你是一位资深金融合规官。请评估以下文章对监管规定的核心条款解读是否准确。",
        "rubric": "完全准确=10分，部分准确=6分，存在误解=2分，有严重错误=0分"
    },
    {
        "name": "实操建议清晰度",
        "weight": 0.25,
        "prompt": "你是一位资深金融合规官。请评估以下文章提供的落地执行建议是否具体、可操作。",
        "rubric": "建议具体可落地=10分，建议较具体=7分，建议模糊=4分，无实操建议=0分"
    },
    {
        "name": "风险提示完整性",
        "weight": 0.20,
        "prompt": "你是一位资深金融合规官。请评估以下文章是否完整提示了新规的主要风险点和潜在合规隐患。",
        "rubric": "风险提示全面=10分，部分覆盖=6分，仅提及=3分，未提及=0分"
    },
    {
        "name": "文章结构与可读性",
        "weight": 0.15,
        "prompt": "你是一位资深金融合规官。请评估以下文章的逻辑结构是否清晰、层次是否分明、表述是否流畅。",
        "rubric": "结构清晰流畅=10分，结构较清晰=7分，结构混乱=3分"
    },
    {
        "name": "时效性与完整性",
        "weight": 0.10,
        "prompt": "你是一位资深金融合规官。请评估以下文章是否覆盖了新规的核心要点，是否有时效性相关的说明。",
        "rubric": "覆盖全面且时效性强=10分，较全面=7分，有遗漏=4分，严重遗漏=0分"
    }
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
```

---

## Task 4: 创建 program.md（Agent 指引）

**Files:**
- Create: `auto_research/program.md`

- [ ] **Step 1: 编写 Agent 指引文件**

```markdown
# 自进化评分引擎 — Agent 指引

## 你的任务
你是一个金融合规领域的 AI 评分专家。你需要不断优化 `scoring_logic.py` 中的评分逻辑，使 AI 评分结果尽可能接近资深合规官的人工判断。

## 目标
通过迭代优化，让模型对文章的排序（从高到低）与 `golden_set.json` 中的 `human_ranking` 顺序一致。

## 成功指标
- **Kendall's τ（排名相关性）**：衡量 AI 排序与人工排序的一致程度
  - τ = 1.0 表示完全一致
  - τ = 0.0 表示无关
  - τ = -1.0 表示完全相反
  - 目标：τ > 0.9

## 评分维度（当前初始版本）
`scoring_logic.py` 中定义了 5 个评分维度，每个维度有独立的 weight、prompt 和 rubric。

## 进化策略

### 修改优先级规则
每次迭代，你将收到上次运行的 deviation 报告。按以下优先级修改：

1. **pairwise_accuracy < 0.6 的维度**：高优先级，必须修改
2. **pairwise_accuracy 0.6~0.8 的维度**：中优先级，可以优化
3. **pairwise_accuracy > 0.8 的维度**：低优先级，保持稳定

### 可执行的修改类型
- **调整权重**：修改某维度的 `weight`（0.0~1.0，总和应为 1.0）
- **优化 Prompt**：改写某维度的 `prompt`，使其更清晰地描述评分标准
- **细化 Rubric**：修改 `rubric`，增加更细粒度的评分描述
- **增删维度**：增加新维度或删除效果差的维度
- **切换聚合策略**：`weighted_sum`（加权求和）或 `llm_judge`（让模型直接给总分）

### 修改建议生成规则
基于 deviation 报告，生成具体、可执行的修改建议：
- 不要说"优化这个维度"，而要说"将'实操建议清晰度'的 weight 从 0.25 调整到 0.35，因为其实 pairwise_accuracy 只有 0.55"
- 每次修改不要超过 2 个维度，避免剧烈变动

## 每次迭代后的判断
- 如果 τ 提升了 → 修改将被保留
- 如果 τ 下降了 → 修改将被丢弃，恢复到上一版

## 输出格式
修改完成后，输出你做了哪些修改以及为什么。
```

---

## Task 5: 创建 agent_research.py（核心循环引擎）

**Files:**
- Create: `auto_research/agent_research.py`

- [ ] **Step 1: 编写 agent_research.py 主循环**

核心功能：
1. `load_scenario()` — 加载 `golden_set.json` 和 `scoring_logic.py`
2. `call_qwen_api(prompt)` — 调用 DashScope Qwen API
3. `score_article(article_text)` — 对单篇文章按所有维度打分
4. `compute_tau(ranking_pred, ranking_true)` — 计算 Kendall's τ
5. `generate_deviation_report()` — 生成 deviation 报告
6. `git_commit_if_improved()` — τ 提升则 commit，退步则 checkout
7. `main_loop()` — 无限循环执行上述步骤

- [ ] **Step 2: 实现 Kendall's τ 计算**

```python
from itertools import combinations

def compute_kendall_tau(pred_ranking, true_ranking):
    """
    pred_ranking: list of article_ids in predicted order (high to low)
    true_ranking: list of article_ids in true order (high to low)
    Returns: float tau in [-1, 1]
    """
    pred_ranks = {aid: i for i, aid in enumerate(pred_ranking)}
    true_ranks = {aid: i for i, aid in enumerate(true_ranking)}

    pairs = list(combinations(pred_ranking, 2))
    if not pairs:
        return 1.0

    concordant = sum(
        (pred_ranks[a] - pred_ranks[b]) * (true_ranks[a] - true_ranks[b]) > 0
        for a, b in pairs
    )
    return len(concordant) / len(pairs)
```

- [ ] **Step 3: 实现 Qwen API 调用**

```python
import os
import requests

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

def call_qwen_api(prompt: str, model: str = "qwen3.6-plus") -> str:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {"temperature": 0.1, "max_tokens": 50}
    }
    response = requests.post(DASHSCOPE_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["output"]["text"].strip()
```

- [ ] **Step 4: 实现日志写入（evolution_log.jsonl + api_calls.log）**

```python
import json
from datetime import datetime

def log_iteration(iteration, tau, tau_delta, dimension_stats, changed_fields, commit_hash):
    log_entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "tau": round(tau, 4),
        "tau_delta": round(tau_delta, 4),
        "pairwise_accuracy": round(sum(d["pairwise_acc"] for d in dimension_stats) / len(dimension_stats), 4),
        "dimensions": dimension_stats,
        "changed_fields": changed_fields,
        "commit_hash": commit_hash
    }
    with open("evolution_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def log_api_call(iteration, article_id, dimension_name, prompt, raw_response, parsed_score):
    with open("api_calls.log", "a") as f:
        f.write(f"=== Iteration {iteration} | Article: {article_id} | Dimension: {dimension_name} ===\n")
        f.write(f"[REQUEST]\nPrompt: {prompt[:200]}...\n")
        f.write(f"[RESPONSE]\nRaw: {raw_response[:200]}\nParsed: {parsed_score}\n===\n")
```

- [ ] **Step 5: 实现 deviation 报告生成**

```python
def generate_deviation_report(iteration, tau, tau_delta, dimension_stats, pairwise_acc_by_dim):
    lines = [
        f"\n{'='*60}",
        f"Deviation Report | Iteration {iteration} | τ={tau:.4f} ({'+' if tau_delta >= 0 else ''}{tau_delta:.4f})",
        f"{'='*60}"
    ]
    for dim in dimension_stats:
        status = "正常" if dim["pairwise_acc"] >= 0.8 else "需优化" if dim["pairwise_acc"] >= 0.6 else "严重拖后腿"
        lines.append(f"[{dim['name']}] weight={dim['weight']:.2f} | pairwise_acc={dim['pairwise_acc']:.2f}")
        lines.append(f"  → 状态: {status}")
        if dim["pairwise_acc"] < 0.6:
            lines.append(f"  → 高优先级修改：需重新设计 prompt 和 rubric")
        elif dim["pairwise_acc"] < 0.8:
            lines.append(f"  → 可优化：调整 rubric 增加细粒度描述")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)
```

- [ ] **Step 6: 实现 Git 集成**

```python
import subprocess

def git_commit_if_improved(tau, tau_prev, changed_fields):
    if tau > tau_prev:
        commit_msg = f"iter {iteration}: τ={tau:.4f} ({'+' if tau > tau_prev else ''}{tau-tau_prev:.4f}) | changed: {', '.join(changed_fields)}"
        subprocess.run(["git", "add", "scoring_logic.py"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        commit_hash = result.stdout.strip()
        print(f"✅ Commit: {commit_hash} - τ improved to {tau:.4f}")
        return commit_hash
    else:
        subprocess.run(["git", "checkout", "scoring_logic.py"], check=True)
        print(f"⚠️ τ dropped to {tau:.4f} - changes discarded")
        return None
```

- [ ] **Step 7: 实现 Agent 修改 scoring_logic.py 的逻辑**

```python
def agent_modify_scoring_logic(deviation_report, iteration):
    """
    基于 deviation 报告，让 Agent 修改 scoring_logic.py
    返回 changed_fields 列表
    """
    # 读取当前 scoring_logic.py
    with open("scoring_logic.py", "r") as f:
        content = f.read()

    # 生成修改建议（基于 deviation 报告）
    # 这里需要调用 Qwen API 来决定如何修改
    modification_prompt = f"""
当前 scoring_logic.py 的 deviation 报告：

{deviation_report}

请决定如何修改 scoring_logic.py。
每轮最多修改 2 个维度。
每次修改可以是：
1. 调整权重（±0.05~±0.15）
2. 改写 prompt（让描述更精确）
3. 细化 rubric（增加评分档次）
4. 增删维度

请以 JSON 格式输出修改计划：
{{
  "modifications": [
    {{"type": "weight", "dimension": "维度名", "old_value": 0.30, "new_value": 0.35, "reason": "pairwise_acc 只有 0.55，需要提高权重"}},
    {{"type": "prompt", "dimension": "维度名", "new_prompt": "新的 prompt 内容"}}
  ]
}}
"""

    response = call_qwen_api(modification_prompt)
    # 解析 JSON 并应用修改...
    # 具体实现见 Task 5 完整代码
```

- [ ] **Step 8: 编写完整主循环**

```python
def main_loop(max_iterations=100):
    iteration = 0
    tau_prev = 0.0

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # 1. 加载数据
        golden = load_golden_set("golden_set.json")
        dims = load_scoring_logic("scoring_logic.py")

        # 2. 对每篇文章评分
        scores = {}
        for aid, article in golden["articles"].items():
            article_scores = score_article(article["content"], dims)
            scores[aid] = article_scores

        # 3. 生成排序
        pred_ranking = sorted(scores.keys(), key=lambda aid: scores[aid]["total"], reverse=True)

        # 4. 计算 τ
        tau = compute_kendall_tau(pred_ranking, golden["human_ranking"])
        tau_delta = tau - tau_prev

        # 5. 生成分维度统计
        dim_stats = compute_dimension_stats(scores, golden["human_ranking"])

        # 6. 写入日志
        log_iteration(iteration, tau, tau_delta, dim_stats, changed_fields, commit_hash)
        print(generate_deviation_report(iteration, tau, tau_delta, dim_stats, scores))

        # 7. Git 决策
        commit_hash = git_commit_if_improved(tau, tau_prev, changed_fields)
        if commit_hash and tau > tau_prev:
            save_best_model("best_scoring_model.py", dims, tau, iteration)

        tau_prev = tau

        # 8. Agent 修改 scoring_logic.py
        deviation_report = generate_deviation_report(...)
        changed_fields = agent_modify_scoring_logic(deviation_report, iteration)
```

> **完整代码**见 `auto_research/agent_research.py` 最终实现。

---

## Task 6: 创建 best_scoring_model.py（最终输出模板）

**Files:**
- Create: `auto_research/best_scoring_model.py`

- [ ] **Step 1: 编写 best_scoring_model.py 模板**

```python
"""
best_scoring_model.py
由 agent_research.py 自动生成
最终最优评分逻辑，固定输出格式
"""

def get_best_model():
    return {
        "source": "auto_evolved_from_scoring_logic",
        "final_iteration": None,  # 由 agent_research.py 运行时填入
        "final_tau": None,       # 由 agent_research.py 运行时填入
        "dimensions": [
            # 由 agent_research.py 从 scoring_logic.py 复制填入
        ],
        "aggregation": "weighted_sum"
    }


def score(article_text: str) -> dict:
    """
    对输入文章返回评分结果
    Returns: {
        "total": float,  # 0-100
        "dimensions": {
            "维度名": {"score": float, "weight": float}
        }
    }
    """
    # 实现：调用 Qwen API，按各维度评分，加权求和
    pass


if __name__ == "__main__":
    model = get_best_model()
    print(f"Best model from iteration {model['final_iteration']}, τ={model['final_tau']}")
    print(f"Dimensions: {[d['name'] for d in model['dimensions']]}")
```

---

## 依赖清单

仅使用 Python 标准库 + `requests`：
```toml
[project]
requires-python = ">=3.10"
dependencies = ["requests"]
```

---

## 运行方式

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
cd auto_research
python agent_research.py
```

---

## 验证计划

- **Task 2** 后：检查 `golden_set.json` 格式正确，PLACEHOLDER 清晰
- **Task 3** 后：运行 `python -c "import scoring_logic; print(len(scoring_logic.dimensions))"` 确认 5 个维度
- **Task 5** 后：运行一次迭代（设置 `max_iterations=1`），检查：
  - `evolution_log.jsonl` 写入一行
  - `api_calls.log` 有内容
  - deviation 报告打印到 stdout
  - 若 τ > 0 则 git commit 成功

---

## Type Consistency Check

- `golden_set.json` 的 `human_ranking` 顺序 = 从高到低（ground truth 排序）
- `compute_kendall_tau()` 输入两个 ranking，顺序一致
- `scoring_logic.py` 的 `weight` 总和 = 1.0（初始版本手动保证）
- `api_calls.log` 的 timestamp 从 `evolution_log.jsonl` 读取，保证一致
