# Research Plan: Do Multi-Turn Conversations Regress to the Prior?

## Motivation & Novelty Assessment

### Why This Research Matters
Current alignment techniques (SFT, RLHF, DPO) are validated primarily in single-turn or short multi-turn settings, yet real-world deployment involves extended conversations. If alignment effects systematically diminish over turns—regressing toward the base model's pre-training distribution—this has critical implications for safety, reliability, and the design of alignment training pipelines. Understanding this degradation mechanism is essential for building trustworthy AI systems.

### Gap in Existing Work
The literature establishes three key facts in isolation:
1. **Alignment is superficial**: Only 5-8% of tokens are affected by alignment tuning (URIAL, Lin et al. 2023), and safety localizes to ~1.4% of neurons (SSAH, Li & Kim 2024).
2. **Multi-turn performance degrades severely**: 39% average performance drop in multi-turn settings (Lost in Conversation, Laban et al. 2025), with instruction retention plummeting.
3. **Alignment effects fade within responses**: KL divergence between base and aligned models decreases monotonically over token positions within a single response.

**No paper has directly measured whether aligned model outputs become more similar to base model outputs as conversation turn count increases.** This is the critical missing link: connecting the superficial alignment hypothesis to multi-turn degradation by showing that the degradation mechanism is specifically regression toward the base prior.

### Our Novel Contribution
We directly test whether multi-turn degradation is explained by regression to the base model prior, using two complementary approaches:
1. **Token-level distribution analysis** across turns using local models with both base and instruct variants
2. **Behavioral alignment probes** measuring alignment-specific behaviors at different conversation depths via API

### Experiment Justification
- **Experiment 1 (Token Distribution Analysis)**: Directly measures KL divergence between instruct and base model output distributions at each turn. This is the most direct possible test of "regression to prior" — if KL(instruct||base) decreases over turns, the instruct model is literally becoming more like the base model.
- **Experiment 2 (Behavioral Alignment Probes)**: Tests whether alignment-specific behaviors (safety refusal, instruction compliance, response formatting) degrade at later turn positions in frontier models. This captures the practical consequences of regression to prior.

## Research Question
In extended multi-turn conversations, do aligned LLMs produce outputs that become progressively more similar to their base (pre-training) model counterparts, indicating that alignment training effects diminish with conversation depth?

## Hypothesis Decomposition

### H1: Token Distribution Convergence
As turn count increases, the KL divergence between the instruct model's next-token distribution and the base model's next-token distribution will decrease, indicating the instruct model's outputs are converging toward the base prior.

### H2: Alignment Behavior Degradation
Alignment-specific behaviors (safety refusals, instruction following, formatting compliance) will show systematic degradation at later turn positions compared to earlier turn positions.

### H3: Early-Turn Anchoring
The strongest alignment effects will be concentrated in turns 1-3, with diminishing alignment signal thereafter — consistent with alignment training data being predominantly short conversations.

## Proposed Methodology

### Approach
Two complementary experiments targeting different levels of analysis:

**Experiment 1: Token Distribution Analysis (Local Models)**
- Models: Llama-3-8B (base) vs Llama-3-8B-Instruct
- Method: Construct multi-turn conversation prefixes of varying lengths (1, 3, 5, 8, 12 turns). At each depth, present standardized probe prompts and measure the next-token probability distributions from both base and instruct models.
- Metric: KL divergence, Jensen-Shannon divergence, top-k token overlap between instruct and base distributions.
- Controls: (a) Single-turn baseline, (b) Same content concatenated (no turn structure), (c) Random conversation content vs. coherent conversation.

**Experiment 2: Behavioral Alignment Probes (API)**
- Models: GPT-4.1-mini via OpenAI API
- Method: Embed standardized probe questions at different turn positions within synthetic multi-turn conversations. Measure alignment-specific responses.
- Probe types:
  - Safety: Requests that aligned models should refuse
  - Instruction compliance: Following formatting instructions from turn 1
  - Helpfulness markers: Quality of assistance
- Controls: Same probes in single-turn (turn 1) as baseline.

### Experimental Steps
1. Download and load Llama-3-8B and Llama-3-8B-Instruct models
2. Generate multi-turn conversation prefixes using the instruct model
3. At each turn depth, compute next-token distributions from both models given the same conversation prefix
4. Calculate divergence metrics across turn positions
5. Design behavioral probe battery for API experiment
6. Run probes at positions 1, 3, 5, 8, 12 in synthetic conversations via GPT-4.1-mini
7. Score alignment behaviors using automated classifiers
8. Statistical analysis: regression of divergence/behavior scores on turn position

### Baselines
1. **Turn 1 (single-turn)**: Maximum alignment effect — the ceiling
2. **Concatenated input**: All conversation content in one message (controls for information accumulation)
3. **Shuffled turns**: Same content but shuffled order (controls for specific conversation dynamics)

### Evaluation Metrics
- **KL Divergence** (D_KL(P_instruct || P_base)): Primary metric for Experiment 1. Lower = more similar to base.
- **Jensen-Shannon Divergence**: Symmetric version for robustness.
- **Top-k Token Overlap**: Fraction of top-k tokens shared between instruct and base distributions.
- **Safety Refusal Rate**: Fraction of harmful requests properly refused (Experiment 2).
- **Instruction Compliance Score**: Whether formatting/style instructions from turn 1 are maintained.

### Statistical Analysis Plan
- **Primary test**: Linear regression of divergence metrics on turn number, with random effects for probe type.
- **Significance level**: α = 0.05, with Bonferroni correction for multiple comparisons.
- **Effect size**: Report Cohen's d and R² for regression models.
- **Bootstrap confidence intervals** (1000 iterations) for all point estimates.

## Expected Outcomes
- **If hypothesis supported**: KL divergence decreases monotonically (or at least significantly) with turn number. Safety refusal rates and instruction compliance decrease at later turns. Effect is stronger for alignment-specific behaviors than for general capability.
- **If hypothesis refuted**: No systematic relationship between turn number and base-model similarity. Degradation explained by other factors (attention dilution, context window effects, etc.).

## Timeline and Milestones
1. Environment setup + model loading: 15 min
2. Experiment 1 implementation + execution: 90 min
3. Experiment 2 implementation + execution: 60 min
4. Analysis + visualization: 45 min
5. Documentation: 30 min

## Potential Challenges
1. **Memory constraints**: Llama-3-8B needs ~16GB. With 4x A6000 (49GB each), this is fine.
2. **Conversation generation quality**: Synthetic conversations may not reflect real patterns. Mitigation: use varied conversation templates and real conversation snippets from WildChat.
3. **API costs**: GPT-4.1-mini is affordable. Budget ~$10-20 for sufficient sample size.
4. **Confound: context length vs turn count**: Longer conversations have more tokens regardless of turns. Mitigation: CONCAT control condition.

## Success Criteria
1. Statistically significant (p < 0.05) negative slope in KL divergence vs. turn number
2. At least 2 of 3 behavioral probe types showing degradation with turn number
3. Effect is robust to conversation content variation (at least 2 different conversation types)
