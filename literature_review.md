# Literature Review: Do Multi-Turn Conversations Regress to the Prior?

## Research Area Overview

This review examines whether large language models (LLMs) regress toward their base pre-training distribution during extended multi-turn conversations, with alignment training effects (safety, instruction-following, formatting) diminishing after the first few turns. The hypothesis sits at the intersection of three active research areas: (1) multi-turn conversation evaluation, (2) the superficial alignment hypothesis, and (3) alignment robustness/safety.

Recent work provides converging evidence from multiple angles: alignment modifies only a thin layer of model behavior (affecting ~5-8% of output tokens and ~1.4% of neurons), multi-turn settings dramatically increase unreliability (39% average performance drop), and alignment effects concentrate at early token positions within each response, fading monotonically.

---

## Key Papers

### Paper 1: LLMs Get Lost In Multi-Turn Conversation
- **Authors**: Laban, Hayashi, Zhou, Neville (Microsoft Research & Salesforce)
- **Year**: 2025
- **Source**: arXiv 2505.06120
- **Key Contribution**: First large-scale study (200,000+ conversations, 15 LLMs, 6 tasks) demonstrating universal multi-turn degradation via "sharding" framework
- **Methodology**: Transforms single-turn tasks into multi-turn by revealing requirements incrementally ("shards"). Compares FULL (single-turn), CONCAT (all shards at once), and SHARDED (multi-turn reveal) settings. Uses automated simulation with GPT-4o-mini as user.
- **Datasets Used**: HumanEval, LiveCodeBench, Spider 1.0, BFCL V3, GSM8K, ToTTo, Summary of a Haystack
- **Results**:
  - Average 39% performance drop from FULL to SHARDED across all models
  - CONCAT recovers 95.1% of FULL performance (degradation is format-specific, not content-specific)
  - Aptitude drops only 16%; unreliability increases 112% (the problem is inconsistency, not inability)
  - Even 2-turn conversations trigger the effect
  - Reasoning models (o3, Deepseek-R1) show no advantage
  - Early answer attempts score 30.9 avg; late attempts score 64.4
  - Loss-of-middle-turns: 150% citation bias toward first/last turns vs. middle
- **Code Available**: GitHub `Microsoft/lost_in_conversation`, HuggingFace `Microsoft/lost_in_conversation`
- **Relevance**: Most directly relevant paper. Demonstrates that multi-turn format induces catastrophic unreliability. The early-turn commitment mechanism (models anchor to first-turn "prior" and fail to update) directly supports our hypothesis. CONCAT control proves degradation is about sequential disclosure, not information content.

### Paper 2: The Unlocking Spell on Base LLMs (URIAL)
- **Authors**: Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula, Choi (Allen AI)
- **Year**: 2023 (published ICLR 2024)
- **Source**: arXiv 2312.01552
- **Key Contribution**: Quantitative evidence that alignment is superficial — only 5-8% of tokens are affected by alignment tuning, and these are predominantly stylistic
- **Methodology**: Token distribution shift analysis comparing base vs. aligned model outputs at each token position. Introduces URIAL: tuning-free alignment via 3 in-context examples.
- **Datasets Used**: just-eval-instruct (1,000 examples from 9 sources), AlpacaEval, MT-Bench, HH-RLHF-redteam, MaliciousInstruct
- **Results**:
  - 77.7% of aligned model tokens are identical to base model's top choice (Llama-2)
  - 92.2% within top-3 of base distribution; only 7.8% genuinely shifted
  - Shifted tokens are stylistic: "Thank", "Hello", "sorry", "cannot", discourse markers
  - KL divergence between base and aligned decreases monotonically over token position
  - URIAL on Mistral-7b (4.63) surpasses Mistral-Instruct SFT (4.44)
  - URIAL on Llama-2-70b (4.74) surpasses Llama-2-70b-chat RLHF (4.67)
- **Code Available**: https://allenai.github.io/re-align, GitHub `Re-Align/URIAL`
- **Relevance**: Provides the mechanistic foundation for our hypothesis. Alignment effects are strongest at early tokens and fade monotonically. In multi-turn conversations, the alignment signal (concentrated in first few tokens of each response) is diluted by accumulated context from the base distribution. "The conversation ability of aligned LLMs might be largely from base models themselves."

### Paper 3: Superficial Safety Alignment Hypothesis (SSAH)
- **Authors**: Li, Kim (NC State)
- **Year**: 2024 (published ICLR 2026)
- **Source**: arXiv 2410.10862
- **Key Contribution**: Safety alignment localizes to ~1.4% of neurons (Safety Critical Units), and these are fragile — easily reassigned during fine-tuning
- **Methodology**: Neuron-level probing, attribute-based pruning, freezing interventions on aligned models
- **Datasets Used**: AdvBench, HEx-PHI, Alpaca, Dolly, HarmBench
- **Results**:
  - SCUs: 1.3% (Llama-2), 1.4% (Llama-3), 2% (Mistral)
  - Pruning SCUs: ASR jumps from 10% → 66% (Llama-2), 15.5% → 86.5% (Llama-3)
  - During fine-tuning, 65.7% of SCUs get reassigned
  - "Current alignment can only hold the correct reasoning direction in a limited generated tokens"
- **Code Available**: https://ssa-h.github.io/
- **Relevance**: Explains WHY alignment is fragile. If safety depends on 1.4% of neurons whose influence extends only over initial tokens, multi-turn context accumulation could easily overwhelm this thin safety layer.

### Paper 4: Quantifying Conversational Reliability under Multi-Turn Interaction
- **Authors**: Myung (Samsung SDS)
- **Year**: 2026 (AAAI 2026)
- **Source**: arXiv 2603.01423
- **Key Contribution**: Controlled measurement of multi-turn reliability degradation across commercial and open-source models
- **Methodology**: Three synthetic tasks (instruction following, tool selection, entity extraction) with matched single/multi-turn variants
- **Results**:
  - Instruction Following: GPT-4o 96% → 63%, GPT-4o-mini 93% → 24%
  - Tool Selection: Qwen3-32B 100% → 47%, Ministral-8B 99% → 37%
  - Degradation driven by instruction drift and contextual overwriting, not raw length
- **Relevance**: Confirms systematic degradation. Key nuance: specific distractors matter more than turn count alone.

### Paper 5: MultiChallenge Benchmark
- **Authors**: Sirdeshmukh et al. (Scale AI)
- **Year**: 2025
- **Source**: arXiv 2501.17399
- **Key Contribution**: Multi-turn benchmark where all frontier models score below 50%
- **Results**:
  - Claude 3.5 Sonnet: 41.4% avg (best); Instruction Retention: 58.6%
  - GPT-4o: 12.5% avg
  - o1-preview: 37.2% avg
- **Code Available**: GitHub `ekwinox117/multi-challenge`
- **Relevance**: Instruction Retention category directly tests whether turn-1 instructions survive through conversation. Results show severe degradation even for frontier models.

### Paper 6: Operationalising the Superficial Alignment Hypothesis via Task Complexity
- **Authors**: Vergara-Browne et al. (Mila/McGill/ETH Zurich/Edinburgh/Amsterdam)
- **Year**: 2026
- **Source**: arXiv 2602.15829
- **Key Contribution**: Formal definition of alignment superficiality via Kolmogorov complexity — post-training adds kilobytes of information atop terabytes of pre-training
- **Results**:
  - Post-training collapses complexity curves: strong performance accessible with <1 KB additional program length
  - Pre-trained models already contain the knowledge; post-training merely surfaces it
- **Relevance**: Theoretical grounding for why alignment is fragile — informationally shallow signals are easily overwhelmed.

### Paper 7: Multi-Turn RLHF Policy Optimization
- **Authors**: arXiv 2410.04612
- **Year**: 2024
- **Key Contribution**: Shows applying single-turn RLHF to multi-turn settings introduces covariate shift
- **Relevance**: Alignment training itself is mismatched to multi-turn deployment.

### Paper 8: Multi-Turn Jailbreaking via Lexical Anchor Tree Search
- **Authors**: arXiv 2601.02670
- **Year**: 2026
- **Key Contribution**: Multi-turn attacks achieve 97% success rate by gradually shifting context
- **Relevance**: Demonstrates that safety alignment can be systematically eroded through multi-turn interaction, consistent with regression-to-prior mechanism.

---

## Common Methodologies

- **Single-turn vs. Multi-turn comparison**: Used in Papers 1, 4, 5 — the gold standard for measuring degradation
- **Token distribution shift analysis**: Paper 2 — directly compares base and aligned model outputs
- **Neuron-level probing/pruning**: Paper 3 — identifies which components encode alignment
- **Kolmogorov complexity estimation**: Paper 6 — formal framework for measuring alignment depth
- **Automated conversation simulation**: Papers 1, 4 — GPT-4o/GPT-5 as user simulators
- **LLM-as-judge evaluation**: Papers 1, 2, 5 — GPT-4 for quality assessment

## Standard Baselines

- **Single-turn performance**: The universal baseline — same task in one turn
- **CONCAT/aggregated input**: All information provided at once (Paper 1's key control)
- **Base model + ICL**: URIAL-style in-context alignment (Paper 2)
- **Various model sizes**: Small (8B), medium (13B-70B), frontier (GPT-4o, Claude, Gemini)

## Evaluation Metrics

- **Task accuracy** (exact match, functional correctness): Papers 1, 4, 5
- **Aptitude vs. Unreliability decomposition**: Paper 1 (90th percentile vs. interpercentile range)
- **Attack Success Rate (ASR)**: Paper 3 (safety evaluation)
- **Token distribution shift metrics**: Paper 2 (KL divergence, base rank, base probability)
- **Multi-aspect scoring** (helpfulness, safety, factuality): Paper 2 (just-eval)

## Datasets in the Literature

| Dataset | Used In | Task | Availability |
|---------|---------|------|-------------|
| Microsoft/lost_in_conversation | Paper 1 | Multi-turn sharded tasks | HuggingFace |
| MultiChallenge | Paper 5 | Multi-turn challenge benchmark | GitHub |
| MT-Bench | Papers 2, 3 | Multi-turn evaluation | HuggingFace |
| AdvBench / HarmBench | Paper 3 | Safety evaluation | Available |
| just-eval-instruct | Paper 2 | Alignment evaluation | Available |
| WildChat-1M | — | Real multi-turn conversations | HuggingFace (gated) |
| LMSYS-Chat-1M | — | Real multi-turn conversations | HuggingFace (gated) |
| Anthropic Sycophancy Evals | — | Sycophancy/alignment testing | HuggingFace |

---

## Gaps and Opportunities

1. **No direct measurement of base-model similarity across turns**: No paper explicitly computes how similar multi-turn responses become to base model outputs as turn count increases. This is the core experiment our hypothesis requires.

2. **Safety-specific multi-turn degradation is understudied**: Papers focus on task performance (accuracy) or safety attacks (jailbreaks) but not on gradual, natural safety erosion without adversarial intent.

3. **Token-level analysis extended to multi-turn**: Paper 2's distribution shift analysis is single-turn only. Extending it across turns would directly test our hypothesis.

4. **Role of conversation content vs. length**: Papers 1 and 4 suggest content matters more than raw turn count, but this isn't fully disentangled.

5. **Comparing aligned vs. base model multi-turn trajectories**: No study directly compares how base and aligned models diverge (or converge) over the course of a conversation.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Microsoft/lost_in_conversation** — Primary dataset. Already has single-turn vs. multi-turn comparisons across 6 tasks and 15 models. We can extend by comparing against base model outputs.
2. **MultiChallenge** — Tests instruction retention across turns. Good for measuring alignment-specific degradation.
3. **MT-Bench** — Standard 2-turn evaluation with human judgments for calibration.
4. **Anthropic Sycophancy Evals** — Tests whether models abandon aligned behavior under social pressure (a form of regression to prior).

### Recommended Baselines
1. **Base model performance** (e.g., Llama-2-7b, Mistral-7b) on same tasks as aligned counterparts
2. **URIAL (in-context alignment)** as a control for alignment without fine-tuning
3. **Single-turn aligned model** as the reference point for full alignment effect
4. **CONCAT setting** (all information at once) to control for information content

### Recommended Metrics
1. **Token distribution divergence from base model** across turns (extending Paper 2's methodology)
2. **Task accuracy per turn** (following Paper 1's sharding framework)
3. **Aptitude vs. Unreliability decomposition** (Paper 1's framework)
4. **Safety compliance rate per turn** (new metric for our hypothesis)
5. **Instruction retention rate** (from MultiChallenge framework)

### Methodological Considerations
- Use the same model in both base and aligned variants (e.g., Llama-2-7b and Llama-2-7b-chat)
- Control for information content using CONCAT-like settings
- Distinguish between capability degradation and alignment-specific degradation
- Consider that distractors and content type may matter more than raw turn count
- Temperature control: Paper 1 shows T=0 doesn't fix multi-turn unreliability
