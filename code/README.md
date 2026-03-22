# Code Repositories

This directory contains cloned repositories relevant to the research topic:
**"Do Multi-Turn Conversations Regress to the Prior?"**

---

## 1. `lost_in_conversation/` — Microsoft/lost_in_conversation

**Source:** https://github.com/microsoft/lost_in_conversation
**Paper:** "LLMs Get Lost in Multi-Turn Conversation" (arXiv:2505.06120)
**License:** MIT

### What it provides
The primary codebase for studying multi-turn conversation degradation in LLMs. Enables simulation of multi-turn conversations across seven analytical generation tasks: Python code, SQL queries, API function calling, math, data-to-text, summarization, and translation.

### Key components
- `run_simulations.py` — entry point for running experiments
- `simulator_*.py` — conversation simulators (full, recap, sharded, snowball modes)
- `tasks/` — task-specific evaluation logic for all seven domains
- `data/sharded_instructions_600.json` — 600 pre-sharded instructions used in experiments
- `prompts/` — all prompts used during simulation and sharding
- `app_conv_viewer.py` — Streamlit-based web viewer for inspecting simulated conversations
- `user_agent.py` / `system_agent.py` — user and assistant agent implementations

### Relevance to research question
Directly measures how LLM performance degrades across conversation turns. Tested on 15 models; results show systematic performance drops in multi-turn vs. single-turn settings — the empirical core of the "regression to prior" question.

---

## 2. `URIAL/` — Re-Align/URIAL

**Source:** https://github.com/Re-Align/URIAL
**Paper:** "The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning" (ICLR 2024, arXiv:2312.01552)
**Organization:** AI2 Mosaic (AllenAI)
**License:** Apache-2.0

### What it provides
URIAL (Untuned LLMs with Restyled In-context Alignment) is a tuning-free alignment method. It achieves alignment of base (untuned) LLMs using only a small number of constant stylistic in-context examples and a system prompt — no fine-tuning required.

### Key components
- `src/unified_infer.py` — main inference script supporting multiple models and datasets
- `urial_prompts/` — in-context alignment prompt templates (e.g., `inst_1k_v4.txt`)
- `evaluate/` — evaluation scripts
- `run_scripts/` — shell scripts for reproducing paper results
- Supports benchmarks: AlpacaEval, MT-Bench, just-eval-instruct

### Relevance to research question
Provides a controlled baseline for studying what alignment via fine-tuning actually adds vs. what can be achieved purely through in-context priming. Useful for isolating whether multi-turn regression is driven by the fine-tuning process itself or by inherent properties of base model priors.

---

## 3. `FastChat/` — lm-sys/FastChat

**Source:** https://github.com/lm-sys/FastChat
**Papers:** Vicuna technical report; Chatbot Arena report (arXiv:2403.04132)
**License:** Apache-2.0

### What it provides
FastChat is the open platform behind Chatbot Arena (lmarena.ai). It includes training, serving, and evaluation infrastructure for LLM-based chatbots, with particular focus on multi-turn conversation quality.

### Key components
- `fastchat/llm_judge/` — **MT-Bench** implementation: a challenging 80-question multi-turn benchmark with GPT-4-based evaluation
- `fastchat/train/` — fine-tuning code (used to train Vicuna)
- `fastchat/serve/` — distributed multi-model serving with OpenAI-compatible API
- `data/` — conversation datasets including Chatbot Arena collections (1.5M human preference votes)
- LMSYS-Chat-1M dataset (1M real-world conversations, separate download)

### Relevance to research question
MT-Bench is the standard multi-turn evaluation benchmark. The LMSYS-Chat-1M dataset provides large-scale real-world multi-turn conversation data. Together these enable studying turn-by-turn quality trends and verifying whether regression patterns observed in simulation hold in naturalistic data.

---

## Summary

| Repo | Role in Research |
|---|---|
| `lost_in_conversation` | Primary: simulate and measure multi-turn performance degradation |
| `URIAL` | Baseline: study alignment via in-context priors vs. fine-tuning |
| `FastChat` | Evaluation + data: MT-Bench multi-turn benchmark and real conversation data |
