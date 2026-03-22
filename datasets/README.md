# Datasets for Multi-Turn Conversation Regression Research

This directory contains datasets for studying whether LLMs regress to base model behavior in long multi-turn conversations.

## Overview

| Dataset | Location | Size | Relevance |
|---------|----------|------|-----------|
| Microsoft Lost in Conversation | `lost_in_conversation/` | 29.1 MB (627 samples) | Direct: multi-turn sharded instruction following |
| MultiChallenge | `multichallenge/` | 273 questions | Direct: multi-turn challenges (instruction retention, coherence) |
| MT-Bench | `mt_bench/` | 80 prompts + 2400 judgments | Direct: 2-turn conversation evaluation |
| WildChat-1M (sample) | `wildchat/` | 50 conversations | Context: real-world multi-turn usage patterns |
| UltraChat 200K (sample) | `ultrachat/` | 20 conversations | Context: synthetic multi-turn instruction tuning data |
| Anthropic Sycophancy Evals | `anthropic_evals/sycophancy/` | 3 files, ~26 MB | Related: behavior shift under user pressure |

---

## Dataset Details

### 1. Microsoft/lost_in_conversation
**Paper**: "LLMs Get Lost In Multi-Turn Conversation" (Laban et al., 2025)
**HuggingFace**: https://huggingface.co/datasets/Microsoft/lost_in_conversation
**GitHub**: https://github.com/microsoft/lost_in_conversation

**Description**: 627 sharded instructions for simulating single-turn vs. multi-turn conversations. Each original single-turn instruction is "sharded" into multiple conversational turns, enabling controlled comparison of model performance when the same problem is posed in one shot vs. across multiple turns.

**Task distribution**:
- `code`: 100 samples (from HumanEval, MBPP, etc.)
- `database`: 107 samples (SQL tasks)
- `actions`: 105 samples (tool-use/agent tasks)
- `math`: 103 samples (mathematical reasoning)
- `data2text`: 120 samples (table-to-text generation)
- `summary`: 92 samples (document summarization)

**Schema**:
```json
{
  "task_id": "sharded-HumanEval/105",
  "task": "code",
  "shards": [{"shard_id": 1, "shard": "Turn digits into names in a list"}, ...],
  "prompt": "...",
  "test": "...",
  "source": "humaneval"
}
```

**Shard counts**: 3–12 shards per sample (mode: 6 shards = 6-turn conversation)

**Files**:
- `lost_in_conversation.json` — full dataset (627 samples, 29.1 MB)
- `sample_5_per_task.json` — 5 samples per task (30 total, for quick inspection)

**License**: CDLA Permissive 2.0

**Download**:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='Microsoft/lost_in_conversation',
                       filename='lost_in_conversation.json', repo_type='dataset')
```

---

### 2. MultiChallenge
**Paper**: "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs" (2025, ACL Findings)
**arXiv**: https://arxiv.org/abs/2501.17399
**GitHub**: https://github.com/ekwinox117/multi-challenge
**Leaderboard**: https://scale.com/leaderboard/multichallenge

**Description**: 273 realistic multi-turn conversation scenarios across 4 challenge axes. Despite frontier models scoring near-perfect on prior multi-turn benchmarks, all score under 50% on MultiChallenge (best: Claude 3.5 Sonnet at 41.4%).

**Axis distribution**:
- `INFERENCE_MEMORY`: 113 questions — recalling user details from earlier turns
- `INSTRUCTION_RETENTION`: 69 questions — following instructions from the first turn throughout
- `SELF_COHERENCE`: 50 questions — staying consistent with prior responses, avoiding sycophancy
- `RELIABLE_VERSION_EDITING`: 41 questions — iterative document editing without drift

**Schema**:
```json
{
  "QUESTION_ID": "674552683acc22154b07a598",
  "AXIS": "INFERENCE_MEMORY",
  "CONVERSATION": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...],
  "TARGET_QUESTION": "...",
  "PASS_CRITERIA": "..."
}
```

**Files**:
- `multichallenge_questions.jsonl` — all 273 questions (downloaded from GitHub)

**Download**:
```bash
curl -L "https://raw.githubusercontent.com/ekwinox117/multi-challenge/main/data/benchmark_questions.jsonl" \
  -o multichallenge_questions.jsonl
```

---

### 3. MT-Bench
**Paper**: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023)
**HuggingFace (prompts)**: https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts
**HuggingFace (judgments)**: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments

**Description**: 80 two-turn questions spanning 8 categories, with human pairwise judgments comparing model responses. The second turn is designed to test follow-up instruction following, a simple form of multi-turn evaluation.

**Category distribution** (10 per category): writing, roleplay, reasoning, math, coding, extraction, STEM, humanities

**Files**:
- `mt_bench_prompts.json` — 80 prompts (both turns + reference answers)
- `mt_bench_human_judgments_sample200.json` — 200 sample human pairwise comparisons
- `mt_bench_gpt4_judgments.json` — 2400 GPT-4 pairwise judgments (full)

**Download**:
```python
from datasets import load_dataset
prompts = load_dataset('HuggingFaceH4/mt_bench_prompts', split='train')
judgments = load_dataset('lmsys/mt_bench_human_judgments')
```

---

### 4. WildChat-1M (sample)
**Paper**: "WildChat: 1M ChatGPT Interaction Logs in the Wild" (Zhao et al., 2024)
**HuggingFace**: https://huggingface.co/datasets/allenai/WildChat-1M
**License**: ODC-BY

**Description**: 1 million real user–ChatGPT conversations (GPT-3.5 and GPT-4). ~41% are multi-turn. The sample here contains 50 conversations with >4 turns (avg 5.2 turns, range 3–18).

**Files**:
- `wildchat_1m_sample50.json` — 50 multi-turn conversations (>4 turns), GPT-3.5 and GPT-4

**Full dataset download** (requires HuggingFace account due to size):
```python
from datasets import load_dataset
ds = load_dataset('allenai/WildChat-1M', split='train', streaming=True)
```

---

### 5. UltraChat 200K (sample)
**HuggingFace**: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

**Description**: 200K high-quality synthetic multi-turn instruction-following conversations. Commonly used for SFT training of aligned models. Useful as a reference for the distribution of multi-turn tasks used in alignment fine-tuning.

**Files**:
- `ultrachat_200k_sample20.json` — 20 sample conversations (avg 6.8 messages per conversation)

**Full download**:
```python
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')
```

---

### 6. Anthropic Model-Written Evaluations (Sycophancy)
**HuggingFace**: https://huggingface.co/datasets/Anthropic/model-written-evals
**Paper**: "Discovering Language Model Behaviors with Model-Written Evaluations" (Perez et al., 2022)

**Description**: Evaluations for sycophancy, persona, and advanced AI risks. The sycophancy subset tests whether models agree with users who assert false positions — directly relevant to studying regression toward user-pleasing base behavior.

**Schema**:
```json
{
  "question": "Hello, my name is Alex... I agree with the claim that...",
  "answer_matching_behavior": " (A)",
  "answer_not_matching_behavior": " (B)"
}
```

**Files** (in `sycophancy/`):
- `sycophancy_on_nlp_survey.jsonl` — sycophancy on NLP paper opinions (~9.7 MB)
- `sycophancy_on_philpapers2020.jsonl` — sycophancy on philosophy positions (~9.7 MB)
- `sycophancy_on_political_typology_quiz.jsonl` — sycophancy on political questions (~7.8 MB)
- `README.md` — original dataset documentation

**Download**:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='Anthropic/model-written-evals',
                       filename='sycophancy/sycophancy_on_nlp_survey.jsonl',
                       repo_type='dataset')
```

---

## Datasets Not Downloaded (Gated/Large)

### LMSYS-Chat-1M
**HuggingFace**: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
**Access**: Requires HuggingFace account + agreement to terms (gated)
**Description**: 1M real-world conversations with 25 LLMs from Chatbot Arena. Useful for studying real distribution of multi-turn conversation lengths and how model quality varies by turn number.

**Download** (after accepting terms on HuggingFace):
```python
from datasets import load_dataset
ds = load_dataset('lmsys/lmsys-chat-1m')
```

### Chatbot Arena Conversations
**HuggingFace**: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
**Access**: Gated (requires agreement to terms)
**Description**: Human preference data from Chatbot Arena pairwise comparisons.

---

## Relevance to Research Question

| Dataset | How it addresses regression to prior |
|---------|--------------------------------------|
| Lost in Conversation | Directly measures degradation when single-turn tasks are spread across turns |
| MultiChallenge | Tests specific failure modes: instruction forgetting, incoherence under pressure |
| MT-Bench | Baseline for 2-turn capability; models are known to perform well here |
| WildChat-1M | Real-world distribution of multi-turn lengths; reveals where models actually fail |
| Anthropic Sycophancy | Tests if models revert to agreement behavior under user pressure (a form of prior regression) |
| UltraChat | Reference for alignment training data distribution |

---

## Reproducing Downloads

All datasets can be re-downloaded by running:

```bash
cd /workspaces/multi-turn-regress-claude
source .venv/bin/activate
uv pip install datasets huggingface_hub tzdata

# Lost in Conversation (full)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Microsoft/lost_in_conversation', 'lost_in_conversation.json', repo_type='dataset',
                local_dir='datasets/lost_in_conversation')
"

# MultiChallenge
curl -L https://raw.githubusercontent.com/ekwinox117/multi-challenge/main/data/benchmark_questions.jsonl \
  -o datasets/multichallenge/multichallenge_questions.jsonl

# MT-Bench (via datasets library)
python -c "
from datasets import load_dataset
import json
prompts = load_dataset('HuggingFaceH4/mt_bench_prompts', split='train')
json.dump([dict(r) for r in prompts], open('datasets/mt_bench/mt_bench_prompts.json','w'), indent=2)
"
```
