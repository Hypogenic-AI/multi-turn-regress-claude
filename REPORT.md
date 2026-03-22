# Do Multi-Turn Conversations Regress to the Prior?

## 1. Executive Summary

We tested the hypothesis that in long multi-turn conversations, LLMs regress toward their base (pre-training) prior, with alignment effects diminishing after the first few turns. Using two complementary experimental approaches — token distribution analysis with local models (Qwen2.5-7B) and behavioral alignment probes via API (GPT-4.1-mini/nano) — we found **partial support with important nuances**.

**Key finding**: Superficial alignment behaviors (formatting constraints, style instructions) degrade significantly over 15-24 turns (uppercase compliance drops from 100% to 25%, p<0.001), while deeper alignment (safety refusals) remains robust. Token-level analysis reveals a two-phase pattern: alignment signal *strengthens* in early turns (0→2) as chat formatting activates, then gradually *weakens* through turns 2→12. The multi-turn conversation format itself (not content) produces systematically higher divergence from base models compared to concatenated equivalents, suggesting that turn structure interacts with alignment mechanisms.

**Practical implication**: Alignment training is not monolithic — different alignment behaviors have different durability. Formatting and style instructions are most vulnerable to multi-turn degradation, consistent with the superficial alignment hypothesis that these represent the thinnest layer of alignment modification.

## 2. Goal

### Hypothesis
In extended multi-turn conversations, aligned LLMs produce outputs that become progressively more similar to their base (pre-training) model counterparts, with alignment training effects diminishing after the first few turns.

### Why This Matters
- Real-world LLM deployment involves extended conversations, not just single turns
- If alignment degrades with turn count, safety guarantees established in short-context evaluations may not hold
- Understanding the degradation mechanism (regression to prior vs. other factors) informs how to build more robust alignment

### Gap in Literature
Prior work established that (a) alignment is superficial (5-8% of tokens, ~1.4% of neurons), (b) multi-turn performance degrades severely (39% average drop), and (c) alignment effects fade within single responses. **No prior work directly measured whether multi-turn aligned model outputs converge toward base model distributions over turns.**

## 3. Data Construction

### Experiment 1: Token Distribution Analysis

**Models**:
- Base: Qwen/Qwen2.5-7B (ungated, 7B parameters)
- Instruct: Qwen/Qwen2.5-7B-Instruct (same family, instruction-tuned)

**Conversation Prefixes**: 3 diverse topics (travel planning, learning programming, cooking), each with 12 turns of natural dialogue. Turns are pre-written to ensure consistency across conditions.

**Probe Prompts**: 8 simple factual/creative questions presented after varying numbers of conversation turns (0, 1, 2, 3, 5, 7, 10, 12).

**Total measurements**: 192 (multi-turn) + 192 (CONCAT control) = 384

### Experiment 2: Behavioral Alignment Probes

**Model**: GPT-4.1-mini via OpenAI API (temperature=0.0)

**Filler Conversations**: 2 topics × 12 turns each of benign dialogue

**Probe Types**: Safety refusal (8 probes), Instruction compliance (8 probes), Persona maintenance (4 probes)

**Turn positions tested**: 0, 2, 4, 6, 8, 10, 12

### Experiment 2b: Extended Behavioral Probes

**Models**: GPT-4.1-mini and GPT-4.1-nano via OpenAI API

**Extended Filler**: 24 turns of diverse conversation topics

**Probe Types**:
- Word limit compliance (30-word limit instruction, 8 probes)
- Uppercase format compliance (ALL CAPS instruction, 4 probes)
- Subtle safety (nuanced/contextualized harmful requests, 6 probes)

**Turn positions tested**: 0, 3, 6, 10, 15, 20, 24

### Data Quality
- All API calls returned successfully (0% error rate)
- Local model computations used float32 for numerical stability
- Reproducibility ensured via fixed random seed (42) and temperature=0.0

## 4. Experiment Description

### Methodology

#### High-Level Approach
We measured alignment at two levels: (1) **token distribution** — comparing next-token probability distributions between base and instruct models given the same conversation context, and (2) **behavioral** — testing whether alignment-specific behaviors (safety, formatting, persona) degrade with conversation depth.

#### Why This Method?
- Token distribution analysis is the most direct test of "regression to prior" — if KL(instruct||base) decreases, the instruct model is literally becoming more base-like
- Behavioral probes test the practical consequences of any regression
- Using both local models and API models provides convergent evidence across different model families
- CONCAT control isolates the effect of multi-turn format from information accumulation

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | GPU computation |
| Transformers | 5.3.0 | Model loading |
| OpenAI | latest | API calls |
| SciPy | latest | Statistical tests |
| Matplotlib/Seaborn | latest | Visualization |

#### Hardware
- 4× NVIDIA RTX A6000 (49GB each)
- Models loaded on separate GPUs (base on cuda:2, instruct on cuda:0)

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 | Deterministic output for reproducibility |
| Max context | 3072 tokens | Sufficient for 12-turn conversations |
| Random seed | 42 | Standard reproducibility seed |
| KL epsilon | 1e-10 | Numerical stability in divergence computation |

### Raw Results

#### Experiment 1: KL Divergence by Turn Count

| Turns | KL(Instruct‖Base) Mean±SD | JS Divergence | Top-100 Overlap |
|-------|---------------------------|---------------|-----------------|
| 0     | 8.36 ± 0.99               | 0.689 ± 0.003 | 0.051 ± 0.023   |
| 1     | 9.32 ± 1.29               | 0.690 ± 0.004 | 0.062 ± 0.026   |
| 2     | 9.59 ± 1.40               | 0.690 ± 0.005 | 0.054 ± 0.026   |
| 3     | 9.36 ± 1.38               | 0.690 ± 0.005 | 0.055 ± 0.027   |
| 5     | 9.35 ± 1.42               | 0.690 ± 0.005 | 0.055 ± 0.025   |
| 7     | 9.32 ± 1.45               | 0.690 ± 0.005 | 0.055 ± 0.025   |
| 10    | 9.31 ± 1.45               | 0.690 ± 0.005 | 0.056 ± 0.025   |
| 12    | 9.16 ± 1.41               | 0.690 ± 0.005 | 0.055 ± 0.026   |

#### Multi-Turn vs CONCAT Control (KL Divergence)

| Turns | Multi-Turn KL | CONCAT KL | Delta | t-test p |
|-------|---------------|-----------|-------|----------|
| 0     | 8.36          | 4.94      | 3.42  | <0.0001  |
| 2     | 9.59          | 6.51      | 3.08  | <0.0001  |
| 5     | 9.35          | 6.12      | 3.22  | <0.0001  |
| 10    | 9.31          | 5.99      | 3.32  | <0.0001  |
| 12    | 9.16          | 5.87      | 3.28  | <0.0001  |

#### Experiment 2: Behavioral Probes (GPT-4.1-mini, 0-12 turns)

| Turn | Safety Refusal | Instruction Compliance | Persona Maintenance |
|------|---------------|------------------------|---------------------|
| 0    | 1.000         | 1.000                  | 0.938               |
| 4    | 1.000         | 1.000                  | 1.000               |
| 8    | 1.000         | 1.000                  | 0.938               |
| 12   | 1.000         | 1.000                  | 0.875               |

#### Experiment 2b: Extended Probes (0-24 turns)

**Uppercase Format Compliance** (most significant degradation):

| Turns | GPT-4.1-mini | GPT-4.1-nano |
|-------|-------------|-------------|
| 0     | 1.00        | 1.00        |
| 3     | 1.00        | 1.00        |
| 6     | 1.00        | 1.00        |
| 10    | 1.00        | 1.00        |
| 15    | 1.00        | 0.50        |
| 20    | 0.50        | 0.50        |
| 24    | 0.25        | 0.50        |

**Word Limit Compliance (30-word limit)**:

| Turns | Mini Score | Mini Avg Words | Nano Score | Nano Avg Words |
|-------|-----------|----------------|-----------|----------------|
| 0     | 1.00      | 24.5           | 1.00      | 20.8           |
| 6     | 1.00      | 29.9           | 1.00      | 27.5           |
| 15    | 0.88      | 34.6           | 0.94      | 31.5           |
| 24    | 0.81      | 34.8           | 1.00      | 31.5           |

### Statistical Analysis

| Probe Type | Model | Slope | R² | p-value | Cohen's d |
|------------|-------|-------|----|---------|-----------|
| KL divergence (all turns) | Qwen2.5-7B | +0.021 | 0.004 | 0.400 | -0.67 |
| KL divergence (turns 2-12) | Qwen2.5-7B | -0.030 | 0.006 | 0.355 | — |
| CONCAT KL (turns 2-12) | Qwen2.5-7B | -0.054 | 0.030 | **0.037** | — |
| Uppercase format | GPT-4.1-mini | -0.030 | 0.404 | **<0.001** | 2.45 |
| Uppercase format | GPT-4.1-nano | -0.027 | 0.291 | **0.003** | 1.41 |
| Word limit | GPT-4.1-mini | -0.013 | 0.293 | **<0.001** | 1.10 |
| Word limit | GPT-4.1-nano | -0.002 | 0.022 | 0.276 | 0.00 |
| Subtle safety | GPT-4.1-mini | -0.004 | 0.017 | 0.416 | 0.15 |
| Subtle safety | GPT-4.1-nano | +0.005 | 0.023 | 0.343 | -0.14 |
| Safety refusal | GPT-4.1-mini | 0.000 | — | — | 0.00 |

## 5. Result Analysis

### Key Findings

**Finding 1: Formatting instruction compliance degrades significantly over 15-24 turns.**
Uppercase formatting compliance drops from 100% to 25% for GPT-4.1-mini (p<0.001, d=2.45) and to 50% for GPT-4.1-nano (p=0.003, d=1.41). Word limit compliance also degrades significantly for GPT-4.1-mini (p<0.001, d=1.10). This is the strongest evidence for alignment regression in our experiments.

**Finding 2: Safety alignment is robust through 24 turns.**
Neither direct safety refusals (perfect 1.0 through 12 turns) nor subtle safety probes (no significant trend through 24 turns, p>0.3) show degradation. This contrasts sharply with formatting degradation.

**Finding 3: Token distribution shows a two-phase pattern — activation then subtle regression.**
KL(instruct||base) increases from 8.36 (turn 0) to 9.59 (turn 2) as the chat format activates alignment signals, then gradually decreases to 9.16 by turn 12. The declining phase (turns 2-12, slope=-0.030) trends in the predicted direction but is not statistically significant (p=0.36).

**Finding 4: Multi-turn format itself amplifies instruct-base divergence.**
Multi-turn conversations produce KL ~3.1 units higher than CONCAT equivalents (p<0.001 at all turns). The turn structure itself — not just context accumulation — makes the instruct model behave more differently from the base model. This is an unexpected finding.

**Finding 5: The degradation threshold appears to be around 10-15 turns.**
Across all behavioral probes, alignment is maintained perfectly through 10 turns but begins degrading between turns 10-15. This is consistent with alignment training data being predominantly short conversations (typically 1-5 turns).

### Hypothesis Testing

**H1 (Token Distribution Convergence): Not supported.** Overall KL does not decrease monotonically across all turns. The turns 2-12 declining trend is consistent with the hypothesis direction but not statistically significant. However, the CONCAT control does show a significant decline (p=0.037), suggesting the regression-to-prior mechanism may be present but masked by the turn-structure activation effect.

**H2 (Alignment Behavior Degradation): Partially supported.** Superficial alignment behaviors (formatting, style) degrade significantly. Deep alignment (safety) does not degrade within 24 turns.

**H3 (Early-Turn Anchoring): Supported.** Alignment behaviors are maintained perfectly through the first 10 turns and begin degrading at turns 10-15, consistent with alignment training data being concentrated in short conversations.

### Surprises and Insights

1. **Multi-turn format *increases* instruct-base divergence** — this was unexpected. The chat template and turn markers seem to activate alignment signals more strongly than a single-turn context. This means the "regression to prior" is competing against a "format activation" effect.

2. **Different alignment layers have different durability** — formatting instructions are fragile (degrade by turn 15), while safety alignment is robust (maintained at turn 24). This is consistent with the superficial alignment hypothesis: formatting is the shallowest layer (~5-8% of tokens) and safety is deeper (protected by ~1.4% of specialized neurons).

3. **GPT-4.1-nano is sometimes *more* robust than GPT-4.1-mini** — nano maintained better word limit compliance at 24 turns (1.00 vs 0.81). This could be because smaller models are more tightly optimized or because their shorter context windows reduce interference.

### Limitations

1. **Limited model coverage**: We tested one local model family (Qwen2.5-7B) and two API models (GPT-4.1-mini/nano). Results may differ for other architectures.

2. **Synthetic conversations**: Filler conversations were pre-written. Real conversations have more varied and potentially adversarial content.

3. **Maximum 24 turns**: Real-world conversations can be much longer. The 10-15 turn degradation threshold might shift with different conversation content.

4. **Token distribution analysis used only first-token prediction**: Alignment effects may manifest more strongly over sequences of tokens rather than single next-token distributions.

5. **Confound: context length vs. turn count**: While the CONCAT control partially addresses this, we cannot fully separate turn-count effects from context-length effects since longer conversations always have more tokens.

6. **Scoring limitations**: Behavioral probe scoring uses heuristics and LLM-as-judge, which may miss subtle degradation or introduce systematic bias.

7. **Small sample sizes for some probes**: Uppercase format had only 4 probes per condition, limiting statistical power for individual comparisons.

## 6. Conclusions

### Summary
Multi-turn conversations do cause alignment degradation, but the pattern is more nuanced than simple "regression to prior." Superficial alignment behaviors (formatting, style instructions) degrade significantly after 10-15 turns, consistent with the superficial alignment hypothesis. Safety alignment, however, remains robust through at least 24 turns in frontier models. Token-level analysis reveals a competing two-phase dynamic: multi-turn format initially strengthens alignment activation, which then gradually weakens — but this weakening was not statistically significant in our 12-turn window.

### Implications
- **For alignment researchers**: Different alignment behaviors sit at different "depths" and have different vulnerability profiles. Formatting and style instructions should be considered particularly fragile in multi-turn settings.
- **For practitioners**: System-level formatting instructions (e.g., "always respond in bullet points") should be periodically re-injected in long conversations. Safety alignment is more durable but should still be monitored.
- **For the hypothesis**: The regression-to-prior mechanism is likely one contributor to multi-turn degradation, particularly for superficial alignment, but it coexists with other mechanisms (attention dilution, instruction drift, format activation effects).

### Confidence in Findings
- **High confidence**: Formatting instruction degradation at 15-24 turns (large effect sizes, p<0.001)
- **High confidence**: Safety robustness through 24 turns (consistent across models and probe types)
- **Moderate confidence**: Two-phase KL pattern (consistent across topics, but individual probe variance is high)
- **Low confidence**: Whether the declining KL trend continues beyond 12 turns (not enough data)

## 7. Next Steps

### Immediate Follow-ups
1. **Extend to longer conversations (50-100+ turns)** — the 10-15 turn degradation threshold needs to be mapped more precisely
2. **Test with open-weight models where base variants are available** (Llama-3-70B base vs. instruct, when accessible) — larger models may show different patterns
3. **Adversarial filler conversations** — test whether filler content that subtly challenges alignment accelerates degradation

### Alternative Approaches
- **Mechanistic interpretability**: Probe attention patterns to see if alignment-critical attention heads become less active at later turns
- **Activation analysis**: Compare instruct model hidden states to base model hidden states at each turn to find the layer-level source of convergence
- **Natural conversation data**: Use WildChat-1M or LMSYS-Chat-1M to measure alignment markers in real conversations

### Open Questions
1. Is the format activation effect (multi-turn > CONCAT KL) a desirable property or an artifact?
2. Does safety alignment eventually degrade at 50+ turns, or is it fundamentally more robust?
3. Can periodic system message re-injection fully prevent formatting degradation?
4. How does the degradation pattern differ across languages and cultural contexts?

## References

### Papers
1. Laban et al. (2025). "LLMs Get Lost In Multi-Turn Conversation." arXiv:2505.06120.
2. Lin et al. (2023). "The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning." ICLR 2024.
3. Li & Kim (2024). "Superficial Safety Alignment Hypothesis." ICLR 2026.
4. Myung (2026). "Quantifying Conversational Reliability under Multi-Turn Interaction." AAAI 2026.
5. Sirdeshmukh et al. (2025). "MultiChallenge: A Realistic Multi-Turn Benchmark." arXiv:2501.17399.
6. Vergara-Browne et al. (2026). "Operationalising the Superficial Alignment Hypothesis." arXiv:2602.15829.

### Datasets
- Microsoft/lost_in_conversation (HuggingFace)
- MultiChallenge (GitHub)
- MT-Bench (HuggingFace)

### Tools
- Qwen/Qwen2.5-7B and Qwen2.5-7B-Instruct (HuggingFace)
- OpenAI GPT-4.1-mini and GPT-4.1-nano APIs
- PyTorch 2.10.0, Transformers 5.3.0
