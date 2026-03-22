# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project investigating whether multi-turn conversations cause LLMs to regress toward their base pre-training prior, with alignment effects diminishing over turns.

---

## Papers

Total papers downloaded: 15

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | LLMs Get Lost In Multi-Turn Conversation | Laban et al. | 2025 | papers/2505.06120_*.pdf | 39% avg degradation, 200K+ conversations |
| 2 | The Unlocking Spell on Base LLMs (URIAL) | Lin et al. | 2023 | papers/2312.01552_*.pdf | Only 5-8% of tokens shifted by alignment |
| 3 | Superficial Safety Alignment Hypothesis | Li & Kim | 2024 | papers/2410.10862_*.pdf | Safety in 1.4% of neurons |
| 4 | Quantifying Conversational Reliability | Myung | 2026 | papers/2603.01423_*.pdf | Multi-turn reliability measurement |
| 5 | MultiChallenge Benchmark | Sirdeshmukh et al. | 2025 | papers/2501.17399_*.pdf | Frontier models <50% on multi-turn |
| 6 | Superficial Alignment via Task Complexity | Vergara-Browne et al. | 2026 | papers/2602.15829_*.pdf | Formal complexity framework |
| 7 | Behavior Shift After Instruction Tuning | — | 2023 | papers/2310.00492_*.pdf | Base→aligned behavior analysis |
| 8 | Multi-Turn RLHF Policy Optimization | — | 2024 | papers/2410.04612_*.pdf | Covariate shift in multi-turn RLHF |
| 9 | Aligning with Individual Preferences | — | 2024 | papers/2410.03642_*.pdf | Multi-turn personalization |
| 10 | Evaluating Multi-Turn Agents Survey | — | 2025 | papers/2503.22458_*.pdf | Evaluation methods survey |
| 11 | Prior Context Effects on LLM Performance | — | 2025 | papers/2506.00069_*.pdf | Up to 73% accuracy drops from context |
| 12 | Survey on Multi-Turn Interactions | — | 2025 | papers/2504.04717_*.pdf | Comprehensive multi-turn survey |
| 13 | Multi-Turn Jailbreak Tree Search | — | 2026 | papers/2601.02670_*.pdf | 97% attack success rate |
| 14 | Instruction Tuning Behavioral Shifts | — | 2025 | papers/2506.15480_*.pdf | Context vs. parametric knowledge |
| 15 | Limitations of Instruction Tuning | — | 2024 | papers/2402.05119_*.pdf | Instruction-following degradation |

See papers/README.md for detailed descriptions.

---

## Datasets

Total datasets downloaded: 6 (+ 2 documented but gated)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Microsoft/lost_in_conversation | HuggingFace | 627 instructions, 29MB | Multi-turn sharded tasks | datasets/lost_in_conversation/ | Primary dataset |
| MultiChallenge | GitHub | 273 examples | Multi-turn challenge | datasets/multichallenge/ | Frontier models <50% |
| MT-Bench | HuggingFace | 80 prompts + 2400 judgments | Multi-turn evaluation | datasets/mt_bench/ | Standard benchmark |
| WildChat-1M (sample) | HuggingFace | 50 conversations (sample) | Real multi-turn convos | datasets/wildchat/ | Full dataset via streaming |
| UltraChat 200K (sample) | HuggingFace | 20 conversations (sample) | Multi-turn SFT data | datasets/ultrachat/ | Training data for alignment |
| Anthropic Sycophancy Evals | HuggingFace | ~26MB, 3 eval files | Sycophancy testing | datasets/anthropic_evals/ | Alignment regression testing |
| LMSYS-Chat-1M | HuggingFace (gated) | 1M conversations | Real conversations | Not downloaded | Requires account approval |
| Chatbot Arena | HuggingFace (gated) | — | Pairwise comparisons | Not downloaded | Requires account approval |

See datasets/README.md for download instructions and detailed descriptions.

---

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| lost_in_conversation | github.com/microsoft/lost_in_conversation | Multi-turn simulation & evaluation | code/lost_in_conversation/ | Primary experimental framework |
| URIAL | github.com/Re-Align/URIAL | Tuning-free base model alignment | code/URIAL/ | Base model ICL alignment |
| FastChat | github.com/lm-sys/FastChat | MT-Bench evaluation & conversation data | code/FastChat/ | Standard evaluation toolkit |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Web search across arXiv, Semantic Scholar, and Google Scholar for multi-turn LLM degradation, superficial alignment, base model prior regression
2. Citation tracking from key papers (URIAL → SSAH → Lost in Conversation)
3. HuggingFace and GitHub search for datasets and implementations
4. Paper-finder tool (service unavailable; manual search used)

### Selection Criteria
- Papers directly studying multi-turn performance degradation
- Papers providing mechanistic evidence for why alignment is superficial
- Papers with available code and datasets for reproduction
- Recency (2023-2026) with established benchmarks preferred

### Challenges Encountered
- Paper-finder service was unavailable; relied on web search
- LMSYS-Chat-1M and Chatbot Arena datasets are gated (require HuggingFace approval)
- No existing paper directly tests our exact hypothesis (regression to base prior across turns)

### Gaps and Workarounds
- No dataset directly measures base-model similarity across turns → can be constructed from Microsoft/lost_in_conversation by running base models on same tasks
- No token-level analysis across multi-turn conversations → extend URIAL's methodology to multi-turn settings

---

## Recommendations for Experiment Design

Based on gathered resources, the recommended experimental approach:

### 1. Primary Dataset: Microsoft/lost_in_conversation
- Already has single-turn vs. multi-turn comparisons
- 627 instructions across 6 tasks
- Can extend by comparing aligned model outputs against base model outputs at each turn

### 2. Baseline Methods
- **Base model** (unaligned): Llama-2-7b, Mistral-7b — establish the "prior" behavior
- **Aligned model** (single-turn): Same task in one turn — establishes alignment ceiling
- **URIAL (ICL alignment)**: Tests whether in-context alignment also degrades across turns
- **CONCAT setting**: All information at once — controls for information content

### 3. Evaluation Metrics
- **Token distribution divergence from base model** at each turn (KL divergence, base rank)
- **Task accuracy per turn number** (following sharding framework)
- **Safety compliance rate per turn** (using safety classifiers)
- **Aptitude vs. Unreliability decomposition** per turn
- **Instruction retention rate** across turns

### 4. Code to Adapt/Reuse
- `code/lost_in_conversation/` — simulation framework, evaluation pipeline
- `code/URIAL/` — token distribution analysis, base model alignment
- `code/FastChat/` — MT-Bench evaluation, conversation infrastructure

### 5. Proposed Experimental Flow
1. Select 3-4 model pairs (base + aligned): Llama-2-7b/chat, Mistral-7b/Instruct, Llama-3-8B/Instruct
2. Run both base and aligned models on lost_in_conversation sharded tasks
3. At each turn, measure: (a) task accuracy, (b) output similarity to base model, (c) safety/alignment markers
4. Plot alignment divergence as a function of turn number
5. Test whether multi-turn aligned model outputs become statistically closer to base model outputs over turns
6. Use URIAL as a control: does ICL-based alignment degrade similarly to fine-tuned alignment?
