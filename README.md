# Do Multi-Turn Conversations Regress to the Prior?

Testing whether alignment effects diminish as conversation depth increases, with LLMs regressing toward their base pre-training distribution.

## Key Findings

- **Formatting instructions degrade significantly at 15-24 turns**: Uppercase format compliance drops from 100% to 25% (GPT-4.1-mini, p<0.001, Cohen's d=2.45). Word limit compliance also degrades (p<0.001).
- **Safety alignment remains robust through 24 turns**: No significant degradation in safety refusal rates for either GPT-4.1-mini or GPT-4.1-nano.
- **Token distributions show a two-phase pattern**: KL(instruct||base) rises at turns 0-2 (format activation), then gradually declines at turns 2-12 (partial regression), though the decline is not statistically significant.
- **Multi-turn format amplifies alignment divergence**: Turn-structured conversations produce ~3.1 KL units more divergence from base than concatenated equivalents (p<0.001).
- **Degradation threshold at ~10-15 turns**: Consistent across probe types — alignment holds perfectly through 10 turns, then begins weakening.

## Reproducing

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformers accelerate openai numpy scipy matplotlib seaborn pandas tqdm

# Run experiments
python src/experiment1_token_distribution.py  # Requires 2x GPU (Qwen2.5-7B)
python src/experiment2_behavioral_probes.py   # Requires OPENAI_API_KEY
python src/experiment2b_extended.py           # Extended probes (24 turns)

# Analyze
python src/analyze_results.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and experimental design
├── src/
│   ├── experiment1_token_distribution.py  # Token-level KL divergence analysis
│   ├── experiment2_behavioral_probes.py   # Behavioral probes (12 turns)
│   ├── experiment2b_extended.py           # Extended probes (24 turns)
│   └── analyze_results.py                 # Statistical analysis and visualization
├── results/
│   ├── experiment1_results.json           # Raw token distribution data
│   ├── experiment2_results.json           # Behavioral probe data (12 turns)
│   ├── experiment2b_results.json          # Extended probe data (24 turns)
│   ├── analysis_summary.json              # Combined statistical summary
│   └── plots/                             # All generated figures
├── literature_review.md         # Prior work synthesis
├── resources.md                 # Available datasets and code
├── datasets/                    # Downloaded datasets
├── code/                        # Cloned baseline repositories
└── papers/                      # Downloaded research papers
```

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.
