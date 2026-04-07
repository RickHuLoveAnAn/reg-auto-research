# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-evolving scoring engine for financial regulation article quality analysis. Inspired by karpathy/autoresearch. The system iteratively optimizes `scoring_logic.py` (dimensions, weights, prompts) by scoring 5 golden articles via DashScope/Qwen API, computing Kendall's τ vs. human ranking, and committing improvements to git.

## Commands

```bash
cd auto_research

# Run all tests
python3 -m pytest test_agent_research.py -v

# Run a single test
python3 -m pytest test_agent_research.py::test_kendall_tau_perfect -v

# Run the evolution loop (100 iterations by default)
python3 agent_research.py

# Run with custom iteration count
python3 agent_research.py 50
```

## Architecture

### Core Files

- **`agent_research.py`** — Main engine (~590 lines). Contains: data loading, API calls, Kendall's τ computation, pairwise accuracy, git operations, agent modification logic, main loop.
- **`scoring_logic.py`** — The evolvable unit. Contains `dimensions` (list of dicts with name/weight/prompt/rubric), `aggregation`, and `build_score_prompt()`. **This file is loaded as a Python module via `importlib.util.spec_from_file_location` and can be rewritten by the agent.**
- **`golden_set.json`** — 5 golden articles about 《中华人民共和国金融法（草案）》with `human_ranking` field. Article content is injected into prompts via `{article_text}`. The `regulation` field provides `{regulation_name}` for prompt parameterization.
- **`test_agent_research.py`** — 22 unit tests. Must pass 100% before running iterations.
- **`evolution_log.jsonl`** — Append-only iteration history (τ, pairwise accuracy per dimension, changed fields, commit hash).
- **`api_calls.log`** — Full API call log (prompt + raw response per call).

### Evolution Loop (main_loop in agent_research.py)

1. Load `scoring_logic.py` and `golden_set.json`
2. For each article and each dimension: call DashScope API, parse score
3. Compute weighted total score per article → predicted ranking
4. Compute Kendall's τ vs. `golden_set["human_ranking"]`
5. If τ > τ_prev: `git commit` scoring_logic.py; else: `git checkout` to discard
6. Call Qwen agent with deviation report → agent modifies `scoring_logic.py` (weight/prompt/rubric)
7. Repeat

### Scoring Logic Loading (Critical: Newline Handling)

`scoring_logic.py` is loaded as a Python module. A known issue causes real newline characters to corrupt the file when the Agent rewrites it via `_write_scoring_logic`. Two safeguards exist:

- **`_sanitize_dimensions()`** (on load): converts real `\n` in prompt/rubric back to escaped `\\n`
- **`_write_scoring_logic()`** uses `repr()` to safely escape strings, preventing SyntaxError

### Prompt Parameterization

`scoring_logic.py` dimension prompts contain `{regulation_name}` and `{target_focus}` placeholders. At runtime, `score_article()` injects `golden["regulation"]` as `regulation_name` into each dimension dict before calling `build_score_prompt()`.

### API

- **URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- **Model**: `qwen-plus` (DashScope)
- **Format**: `{"messages": [{"role": "user", "content": prompt}]}`
- **Temperature**: 0 (set in `call_qwen_api()` — deterministic scoring)

### Kendall's τ Formula

`(C - D) / len(pairs)` where C=concordant pairs, D=discordant pairs. Implemented in `compute_kendall_tau()`.

## Known Issues / Gotchas

- **Temperature 0.1 causes τ oscillation**: Previously caused τ to swing 0.4–0.6 between iterations. Temperature is now hardcoded to 0 in `call_qwen_api()`.
- **Agent may rewrite `build_score_prompt`**: The function lives inside `scoring_logic.py` (the evolvable file), so the Agent can modify or delete it. The test `test_no_double_agent_call_in_main_loop` guards against double-calling `agent_modify_scoring_logic` but cannot prevent content rewrites. If the Agent removes `.format(regulation_name=...)` calls, `{regulation_name}` appears literally in prompts.
- **Real newlines corrupt scoring_logic.py**: The recurring `\n` vs `\\n` issue. Use `_sanitize_dimensions()` on load and `repr()` in `_write_scoring_logic()`.
- **evolution_log.jsonl and api_calls.log are untracked**: They are gitignored. First-run or fresh clone will have empty log files.
