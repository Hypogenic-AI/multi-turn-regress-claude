"""
Analysis script: Statistical tests and visualizations for both experiments.
Produces publication-quality figures and statistical results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})
sns.set_style("whitegrid")


def analyze_experiment1():
    """Analyze token distribution results."""
    print("=" * 60)
    print("EXPERIMENT 1 ANALYSIS: Token Distribution Convergence")
    print("=" * 60)

    path = os.path.join(RESULTS_DIR, "experiment1_results.json")
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["multi_turn_results"])
    df_concat = pd.DataFrame(data["concat_results"])

    config = data["config"]
    print(f"Model pair: {config['base_model']} vs {config['instruct_model']}")
    print(f"Topics: {config['num_topics']}, Probes: {config['num_probes']}")
    print(f"Turn counts: {config['turn_counts']}")
    print(f"Total measurements: {len(df)}")

    # --- Summary statistics by turn count ---
    print("\n--- KL(Instruct || Base) by Turn Count ---")
    summary = df.groupby("num_turns").agg({
        "kl_instruct_base": ["mean", "std", "median", "count"],
        "js_divergence": ["mean", "std"],
        "top100_overlap": ["mean", "std"],
    }).round(4)
    print(summary)

    # --- Statistical test: linear regression of KL on turn number ---
    print("\n--- Regression Analysis ---")
    means_by_turn = df.groupby("num_turns")["kl_instruct_base"].mean()
    stds_by_turn = df.groupby("num_turns")["kl_instruct_base"].std()

    slope, intercept, r_value, p_value, std_err = linregress(
        df["num_turns"], df["kl_instruct_base"]
    )
    print(f"KL Divergence ~ Turn Number:")
    print(f"  Slope: {slope:.6f} (SE: {std_err:.6f})")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6e}")

    # Spearman correlation (non-parametric)
    rho, p_spearman = spearmanr(df["num_turns"], df["kl_instruct_base"])
    print(f"  Spearman ρ: {rho:.4f}, p={p_spearman:.6e}")

    # Same for JS divergence
    slope_js, _, r_js, p_js, _ = linregress(df["num_turns"], df["js_divergence"])
    print(f"\nJS Divergence ~ Turn Number:")
    print(f"  Slope: {slope_js:.6f}, R²: {r_js**2:.4f}, p={p_js:.6e}")

    # Top-100 overlap
    slope_overlap, _, r_overlap, p_overlap, _ = linregress(df["num_turns"], df["top100_overlap"])
    print(f"\nTop-100 Overlap ~ Turn Number:")
    print(f"  Slope: {slope_overlap:.6f}, R²: {r_overlap**2:.4f}, p={p_overlap:.6e}")

    # --- Compare multi-turn vs CONCAT ---
    if len(df_concat) > 0:
        print("\n--- Multi-Turn vs CONCAT Control ---")
        for turns in sorted(df["num_turns"].unique()):
            mt_kl = df[df["num_turns"] == turns]["kl_instruct_base"].values
            concat_kl = df_concat[df_concat["num_turns"] == turns]["kl_instruct_base"].values
            if len(mt_kl) > 0 and len(concat_kl) > 0:
                t_stat, p_val = stats.ttest_ind(mt_kl, concat_kl)
                print(f"  Turn {turns:2d}: MT KL={mt_kl.mean():.4f}±{mt_kl.std():.4f}, "
                      f"CONCAT KL={concat_kl.mean():.4f}±{concat_kl.std():.4f}, "
                      f"t={t_stat:.2f}, p={p_val:.4f}")

    # --- Bootstrap confidence intervals ---
    print("\n--- Bootstrap 95% CI for KL slope ---")
    n_bootstrap = 1000
    slopes = []
    for _ in range(n_bootstrap):
        sample = df.sample(frac=1, replace=True)
        s, _, _, _, _ = linregress(sample["num_turns"], sample["kl_instruct_base"])
        slopes.append(s)
    ci_lower = np.percentile(slopes, 2.5)
    ci_upper = np.percentile(slopes, 97.5)
    print(f"  Slope: {slope:.6f} [{ci_lower:.6f}, {ci_upper:.6f}]")

    # --- Effect size ---
    turn0_kl = df[df["num_turns"] == 0]["kl_instruct_base"].values
    max_turn = df["num_turns"].max()
    turnN_kl = df[df["num_turns"] == max_turn]["kl_instruct_base"].values
    if len(turn0_kl) > 0 and len(turnN_kl) > 0:
        cohens_d = (turn0_kl.mean() - turnN_kl.mean()) / np.sqrt(
            (turn0_kl.std()**2 + turnN_kl.std()**2) / 2
        )
        print(f"\n  Cohen's d (Turn 0 vs Turn {max_turn}): {cohens_d:.4f}")
        print(f"  Turn 0 KL: {turn0_kl.mean():.4f}±{turn0_kl.std():.4f}")
        print(f"  Turn {max_turn} KL: {turnN_kl.mean():.4f}±{turnN_kl.std():.4f}")

    # ==================== PLOTS ====================

    # Plot 1: KL divergence vs turn number (main result)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a: KL divergence
    ax = axes[0]
    means = df.groupby("num_turns")["kl_instruct_base"].mean()
    sems = df.groupby("num_turns")["kl_instruct_base"].sem()
    ax.errorbar(means.index, means.values, yerr=1.96*sems.values,
                marker='o', linewidth=2, capsize=4, color='#2196F3', label='Multi-turn')

    if len(df_concat) > 0:
        c_means = df_concat.groupby("num_turns")["kl_instruct_base"].mean()
        c_sems = df_concat.groupby("num_turns")["kl_instruct_base"].sem()
        ax.errorbar(c_means.index, c_means.values, yerr=1.96*c_sems.values,
                    marker='s', linewidth=2, capsize=4, color='#FF9800', linestyle='--', label='CONCAT')

    # Add regression line
    x_line = np.linspace(df["num_turns"].min(), df["num_turns"].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, '--', color='red', alpha=0.5,
            label=f'Fit: slope={slope:.4f}, p={p_value:.3e}')

    ax.set_xlabel("Number of Prior Conversation Turns")
    ax.set_ylabel("KL(Instruct || Base)")
    ax.set_title("Token Distribution Divergence\nfrom Base Model")
    ax.legend()

    # 1b: JS divergence
    ax = axes[1]
    means_js = df.groupby("num_turns")["js_divergence"].mean()
    sems_js = df.groupby("num_turns")["js_divergence"].sem()
    ax.errorbar(means_js.index, means_js.values, yerr=1.96*sems_js.values,
                marker='o', linewidth=2, capsize=4, color='#4CAF50')
    ax.set_xlabel("Number of Prior Conversation Turns")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("JS Divergence Between\nInstruct and Base")

    # 1c: Top-k overlap
    ax = axes[2]
    for k, col, color in [(10, "top10_overlap", "#E91E63"),
                           (50, "top50_overlap", "#9C27B0"),
                           (100, "top100_overlap", "#00BCD4")]:
        means_k = df.groupby("num_turns")[col].mean()
        sems_k = df.groupby("num_turns")[col].sem()
        ax.errorbar(means_k.index, means_k.values, yerr=1.96*sems_k.values,
                    marker='o', linewidth=2, capsize=4, color=color, label=f'Top-{k}')
    ax.set_xlabel("Number of Prior Conversation Turns")
    ax.set_ylabel("Token Overlap Fraction")
    ax.set_title("Top-k Token Overlap Between\nInstruct and Base")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_divergence_vs_turns.png"))
    plt.close()
    print(f"\nPlot saved: {PLOTS_DIR}/exp1_divergence_vs_turns.png")

    # Plot 2: Per-topic breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    for topic_idx in df["topic_idx"].unique():
        topic_df = df[df["topic_idx"] == topic_idx]
        means = topic_df.groupby("num_turns")["kl_instruct_base"].mean()
        ax.plot(means.index, means.values, marker='o', linewidth=1.5,
                label=f'Topic {topic_idx + 1}', alpha=0.7)

    overall_means = df.groupby("num_turns")["kl_instruct_base"].mean()
    ax.plot(overall_means.index, overall_means.values, marker='s', linewidth=3,
            color='black', label='Overall Mean', zorder=10)

    ax.set_xlabel("Number of Prior Conversation Turns")
    ax.set_ylabel("KL(Instruct || Base)")
    ax.set_title("Per-Topic KL Divergence Trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_per_topic.png"))
    plt.close()

    # Plot 3: Distribution of KL values at each turn
    fig, ax = plt.subplots(figsize=(10, 6))
    turn_vals = sorted(df["num_turns"].unique())
    data_for_box = [df[df["num_turns"] == t]["kl_instruct_base"].values for t in turn_vals]
    bp = ax.boxplot(data_for_box, positions=range(len(turn_vals)), widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.5)
    ax.set_xticks(range(len(turn_vals)))
    ax.set_xticklabels(turn_vals)
    ax.set_xlabel("Number of Prior Conversation Turns")
    ax.set_ylabel("KL(Instruct || Base)")
    ax.set_title("Distribution of KL Divergence at Each Turn Depth")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp1_boxplot.png"))
    plt.close()

    return {
        "kl_slope": slope,
        "kl_slope_ci": [ci_lower, ci_upper],
        "kl_pvalue": p_value,
        "kl_r_squared": r_value**2,
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "js_slope": slope_js,
        "js_pvalue": p_js,
        "overlap_slope": slope_overlap,
        "overlap_pvalue": p_overlap,
        "turn0_kl_mean": float(turn0_kl.mean()) if len(turn0_kl) > 0 else None,
        "turnN_kl_mean": float(turnN_kl.mean()) if len(turnN_kl) > 0 else None,
        "cohens_d": float(cohens_d) if 'cohens_d' in dir() else None,
        "summary_by_turn": {int(k): {"mean": float(v), "std": float(stds_by_turn[k])}
                           for k, v in means_by_turn.items()},
    }


def analyze_experiment2():
    """Analyze behavioral probe results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 ANALYSIS: Behavioral Alignment Probes")
    print("=" * 60)

    path = os.path.join(RESULTS_DIR, "experiment2_results.json")
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    config = data["config"]
    print(f"Model: {config['model']}")
    print(f"Turn positions: {config['turn_positions_tested']}")

    results_by_type = {}

    for probe_type, key in [("Safety Refusal", "safety_results"),
                             ("Instruction Compliance", "instruction_results"),
                             ("Persona Maintenance", "persona_results")]:
        if key not in data or not data[key]:
            print(f"\n{probe_type}: No data")
            continue

        df = pd.DataFrame(data[key])
        print(f"\n--- {probe_type} ---")
        print(f"Total measurements: {len(df)}")

        summary = df.groupby("num_filler_turns")["score"].agg(["mean", "std", "count"]).round(3)
        print(summary)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            df["num_filler_turns"], df["score"]
        )
        print(f"  Slope: {slope:.6f} (SE: {std_err:.6f})")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.6e}")

        rho, p_spearman = spearmanr(df["num_filler_turns"], df["score"])
        print(f"  Spearman ρ: {rho:.4f}, p={p_spearman:.6e}")

        # Effect size
        turn0 = df[df["num_filler_turns"] == 0]["score"].values
        max_turn = df["num_filler_turns"].max()
        turnN = df[df["num_filler_turns"] == max_turn]["score"].values
        if len(turn0) > 0 and len(turnN) > 0:
            pooled_std = np.sqrt((turn0.std()**2 + turnN.std()**2) / 2)
            d = (turn0.mean() - turnN.mean()) / pooled_std if pooled_std > 0 else 0
            print(f"  Cohen's d (Turn 0 vs Turn {max_turn}): {d:.4f}")
            print(f"  Turn 0: {turn0.mean():.3f}±{turn0.std():.3f}")
            print(f"  Turn {max_turn}: {turnN.mean():.3f}±{turnN.std():.3f}")

        results_by_type[probe_type] = {
            "slope": slope,
            "pvalue": p_value,
            "r_squared": r_value**2,
            "spearman_rho": rho,
            "spearman_p": p_spearman,
            "turn0_mean": float(turn0.mean()) if len(turn0) > 0 else None,
            "turnN_mean": float(turnN.mean()) if len(turnN) > 0 else None,
        }

    # ==================== PLOTS ====================

    # Combined plot of all probe types
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Safety Refusal": "#F44336", "Instruction Compliance": "#2196F3",
              "Persona Maintenance": "#4CAF50"}

    for idx, (probe_type, key) in enumerate([("Safety Refusal", "safety_results"),
                                               ("Instruction Compliance", "instruction_results"),
                                               ("Persona Maintenance", "persona_results")]):
        ax = axes[idx]
        if key not in data or not data[key]:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        df = pd.DataFrame(data[key])
        means = df.groupby("num_filler_turns")["score"].mean()
        sems = df.groupby("num_filler_turns")["score"].sem()

        ax.errorbar(means.index, means.values, yerr=1.96*sems.values,
                    marker='o', linewidth=2, capsize=4, color=colors[probe_type])

        # Regression line
        slope, intercept, _, p_value, _ = linregress(df["num_filler_turns"], df["score"])
        x_line = np.linspace(df["num_filler_turns"].min(), df["num_filler_turns"].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, '--', color='black', alpha=0.5,
                label=f'slope={slope:.4f}, p={p_value:.3e}')

        ax.set_xlabel("Filler Conversation Turns Before Probe")
        ax.set_ylabel("Alignment Score (0=failed, 1=maintained)")
        ax.set_title(probe_type)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='lower left')

    plt.suptitle("Behavioral Alignment Probes: Score vs. Conversation Depth\n(GPT-4.1-mini)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp2_behavioral_probes.png"))
    plt.close()
    print(f"\nPlot saved: {PLOTS_DIR}/exp2_behavioral_probes.png")

    # Summary heatmap
    all_data = []
    for probe_type, key in [("Safety", "safety_results"),
                             ("Instructions", "instruction_results"),
                             ("Persona", "persona_results")]:
        if key in data and data[key]:
            df = pd.DataFrame(data[key])
            for turn, group in df.groupby("num_filler_turns"):
                all_data.append({
                    "Probe Type": probe_type,
                    "Turn Position": int(turn),
                    "Mean Score": group["score"].mean(),
                })

    if all_data:
        heatmap_df = pd.DataFrame(all_data)
        pivot = heatmap_df.pivot(index="Probe Type", columns="Turn Position", values="Mean Score")

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                    ax=ax, cbar_kws={'label': 'Alignment Score'})
        ax.set_title("Alignment Score Heatmap: Probe Type × Conversation Depth")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "exp2_heatmap.png"))
        plt.close()
        print(f"Plot saved: {PLOTS_DIR}/exp2_heatmap.png")

    return results_by_type


def create_combined_summary(exp1_results, exp2_results):
    """Create a combined summary of both experiments."""
    summary = {
        "experiment1": exp1_results,
        "experiment2": exp2_results,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(os.path.join(RESULTS_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)

    if exp1_results:
        print("\nExperiment 1 (Token Distribution):")
        print(f"  KL slope: {exp1_results['kl_slope']:.6f} (p={exp1_results['kl_pvalue']:.4e})")
        if exp1_results['kl_slope'] < 0:
            print("  → KL DECREASING: Instruct model becoming MORE similar to base over turns")
            print("  → SUPPORTS hypothesis of regression to prior")
        else:
            print("  → KL INCREASING: Instruct model becoming LESS similar to base over turns")
            print("  → DOES NOT SUPPORT hypothesis")

    if exp2_results:
        print("\nExperiment 2 (Behavioral Probes):")
        for probe_type, res in exp2_results.items():
            direction = "degrading" if res["slope"] < 0 else "stable/improving"
            sig = "significant" if res["pvalue"] < 0.05 else "not significant"
            print(f"  {probe_type}: {direction} ({sig}, p={res['pvalue']:.4e})")


def main():
    exp1_results = analyze_experiment1()
    exp2_results = analyze_experiment2()
    create_combined_summary(exp1_results, exp2_results)


if __name__ == "__main__":
    main()
