# 自进化评分引擎 — 设计文档

## 1. 项目概述

**目标**：构建一个自主进化的 AI 评分引擎，对金融监管新规解读文章进行评分，通过 golden dataset + 闭环迭代让模型评分逐渐接近资深合规官的人工判断。

**类比 karpathy/autoresearch**：
- `train.py` → `scoring_logic.py`（Agent 修改）
- `val_bpb` → Kendall's τ（排名相关性）
- `git reset` on regression → `git checkout` on deviation

---

## 2. 文件结构

```
auto_research/
├── program.md              # 人类编写：研究方向指引
├── golden_set.json         # 5篇高质量文章 + 人工排序
├── agent_research.py       # 主循环引擎（固定不变）
├── scoring_logic.py        # Agent 修改：维度、权重、Prompt
├── best_scoring_model.py   # 最终输出：最优评分逻辑
├── evolution_log.jsonl     # 结构化结果日志（每次迭代）
└── api_calls.log          # API 请求/响应日志
```

---

## 3. golden_set.json 结构

```json
{
  "regulation": "某金融监管外规名称",
  "description": "外规简要描述",
  "human_ranking": ["article_id_1", "article_id_3", "article_id_2", "article_id_5", "article_id_4"],
  "articles": {
    "article_id_1": {
      "title": "文章标题",
      "content": "文章完整正文（文本）",
      "human_score": 92
    }
  }
}
```

- `human_ranking`：人工确认的排序（从高到低），作为 ground truth
- `human_score`：综合分值（100分制），供参考
- 排序用于计算 Kendall's τ，分值仅作参考

---

## 4. scoring_logic.py 可进化单元

```python
dimensions = [
    {
        "name": "解读准确性",
        "weight": 0.30,
        "prompt": "评估文章对新规核心条款的解读是否准确...",
        "rubric": "完全准确=10分，部分准确=6分，有误解=2分"
    },
    {
        "name": "实操建议清晰度",
        "weight": 0.25,
        "prompt": "评估文章提供的落地建议是否具体可执行...",
        "rubric": "..."
    }
]

aggregation = "weighted_sum"  # 或 "llm_judge"
```

**每个 iteration，Agent 可以：**
- 增删维度
- 修改 `weight`（0.0 ~ 1.0）
- 修改 `prompt` 和 `rubric`
- 切换 `aggregation` 策略

---

## 5. 核心循环（agent_research.py）

```
while True:
    1. 读取 program.md 获取方向指引
    2. 读取当前 scoring_logic.py
    3. 读取 golden_set.json
    4. 对每篇文章调用 Qwen API，按当前逻辑打分
    5. 计算模型排序 vs 人工排序 → Kendall's τ
    6. 生成 deviation 报告
    7. if τ 提升了:
           git add scoring_logic.py && git commit
           保存当前为 best_scoring_model.py
       else:
           git checkout scoring_logic.py
    8. 将 iteration 结果写入 evolution_log.jsonl
    9. 将 API 调用记录写入 api_calls.log
    10. Agent 基于 deviation 报告修改 scoring_logic.py
    11. 等待下一轮
```

---

## 6. 日志规格

### 6.1 evolution_log.jsonl（结构化结果日志）

每次迭代写入一行 JSON：
```json
{
  "iteration": 1,
  "timestamp": "2026-04-07T10:00:00",
  "tau": 0.82,
  "tau_delta": 0.05,
  "pairwise_accuracy": 0.88,
  "dimensions": [
    {
      "name": "解读准确性",
      "weight": 0.30,
      "pairwise_accuracy": 0.85
    },
    {
      "name": "实操建议清晰度",
      "weight": 0.25,
      "pairwise_accuracy": 0.72
    }
  ],
  "changed_fields": ["实操建议清晰度 weight: 0.20→0.25"],
  "commit_hash": "abc1234"
}
```

### 6.2 api_calls.log（API 调用日志）

每次 Qwen API 调用记录：
```
=== Iteration 3 | Article: article_id_1 | Dimension: 解读准确性 ===
[REQUEST]
Model: qwen3.6-plus
Prompt: <full prompt sent to API>
[RESPONSE]
Raw output: <raw model response>
Parsed score: 8.5
===
```

### 6.3 Deviation 报告（打印到 stdout + 写入日志文件）

每个 iteration 末尾输出：
```
========== Deviation Report | Iteration 3 | τ=0.82 (+0.05) ==========
[解读准确性] weight=0.30 | pairwise_acc=0.85
  → 状态: 正常

[实操建议清晰度] weight=0.25 | pairwise_acc=0.72
  → 短板：prompt 描述模糊，建议强化 rubric
  → 建议修改：增加"具体案例数"作为子指标

[风险提示完整性] weight=0.20 | pairwise_acc=0.55
  → 严重拖后腿！建议提高权重或优化 prompt

==================== Commit: abc1234 ====================
```

---

## 7. Git 安全策略

- **只 add `scoring_logic.py`**，其他文件受保护
- `golden_set.json`、`agent_research.py` 不可修改
- 每次 commit message 格式：`iter N: τ=0.83 (+0.04) | changed: 解读准确性 weight 0.30→0.25`
- 历史 commit 形成 Score Evolution History，human 可随时 review

---

## 8. 进化策略（带方向性探索）

基于上次 deviation 报告，Agent 优先修改**拖后腿的维度**：
- pairwise_accuracy < 0.6 → 高优先级修改
- pairwise_accuracy 0.6~0.8 → 中优先级优化
- pairwise_accuracy > 0.8 → 保持稳定

Mutation 类型：
- **Weight tuning**：±0.05 扰动
- **Prompt refinement**：改写维度 prompt/rubric
- **Dimension add/remove**：增删新维度
- **Aggregation switch**：切换聚合策略

---

## 9. 最终输出 best_scoring_model.py

包含：
1. 进化后的完整 `dimensions` 列表（所有维度的 name、weight、prompt、rubric）
2. 最终 `aggregation` 策略
3. `score(article_text) → (总分, 各维度分)` 函数
4. 元信息：训练 iteration 数、最终 τ 值

---

## 10. 成功标准

- τ（Kendall's τ）收敛到 > 0.9
- 或连续 5 次 iteration τ 无显著提升（< 0.01）
- human 可以随时介入，修改 `program.md` 调整方向
