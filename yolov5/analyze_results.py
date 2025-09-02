# Results Analysis and Comparison Tool
"""
Analyze and compare results from different optimization experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import argparse


def load_results(results_dir):
    """Load results from all experiments"""
    results_path = Path(results_dir)
    experiments = {}

    # Find all experiment directories
    for exp_dir in results_path.glob("**/"):
        if "tensorboard" in str(exp_dir):
            continue

        # Look for results.csv
        results_csv = exp_dir / "results.csv"
        if results_csv.exists():
            exp_name = exp_dir.name
            try:
                df = pd.read_csv(results_csv)
                experiments[exp_name] = df
                print(f"✅ Loaded {exp_name}: {len(df)} epochs")
            except Exception as e:
                print(f"❌ Error loading {exp_name}: {e}")

    return experiments


def plot_convergence_comparison(experiments, metric="metrics/mAP_0.5", save_path=None):
    """Plot convergence curves for comparison"""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'orange', 'green', 'red',
              'purple', 'brown', 'pink', 'gray']

    for i, (exp_name, df) in enumerate(experiments.items()):
        if metric in df.columns:
            epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
            values = df[metric]

            plt.plot(epochs, values,
                     label=exp_name,
                     color=colors[i % len(colors)],
                     linewidth=2,
                     marker='o',
                     markersize=4)

    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Convergence Comparison - {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_metrics(experiments, save_dir=None):
    """Plot multiple metrics for comparison"""
    metrics = ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
               'metrics/precision', 'metrics/recall']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    colors = ['blue', 'orange', 'green', 'red',
              'purple', 'brown', 'pink', 'gray']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for j, (exp_name, df) in enumerate(experiments.items()):
            if metric in df.columns:
                epochs = df['epoch'] if 'epoch' in df.columns else range(
                    len(df))
                values = df[metric]

                ax.plot(epochs, values,
                        label=exp_name,
                        color=colors[j % len(colors)],
                        linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').replace('metrics/', '').title())
        ax.set_title(metric.replace('_', ' ').replace('metrics/', '').title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / "metrics_comparison.png",
                    dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table(experiments):
    """Create summary table with best results"""
    summary_data = []

    for exp_name, df in experiments.items():
        # Get final epoch results
        final_results = df.iloc[-1] if len(df) > 0 else None

        if final_results is not None:
            summary_data.append({
                'Experiment': exp_name,
                'Best mAP@0.5': df['metrics/mAP_0.5'].max() if 'metrics/mAP_0.5' in df.columns else 0,
                'Best mAP@0.5:0.95': df['metrics/mAP_0.5:0.95'].max() if 'metrics/mAP_0.5:0.95' in df.columns else 0,
                'Best Precision': df['metrics/precision'].max() if 'metrics/precision' in df.columns else 0,
                'Best Recall': df['metrics/recall'].max() if 'metrics/recall' in df.columns else 0,
                'Final Train Loss': final_results.get('train/box_loss', 0) + final_results.get('train/obj_loss', 0) + final_results.get('train/cls_loss', 0),
                'Convergence Epoch': df['metrics/mAP_0.5'].idxmax() if 'metrics/mAP_0.5' in df.columns else 0,
                'Total Epochs': len(df)
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def analyze_convergence_stability(experiments):
    """Analyze convergence stability"""
    stability_analysis = {}

    for exp_name, df in experiments.items():
        if 'metrics/mAP_0.5' in df.columns:
            values = df['metrics/mAP_0.5'].values

            # Calculate convergence metrics
            max_val = np.max(values)
            final_val = values[-1]
            std_last_10 = np.std(
                values[-10:]) if len(values) >= 10 else np.std(values)

            # Find convergence point (when improvement becomes < 1% for 5 consecutive epochs)
            convergence_epoch = None
            for i in range(5, len(values)):
                recent_improvements = [(values[j] - values[j-1]) / values[j-1]
                                       for j in range(i-4, i+1) if values[j-1] > 0]
                if all(imp < 0.01 for imp in recent_improvements):
                    convergence_epoch = i
                    break

            stability_analysis[exp_name] = {
                'max_mAP': max_val,
                'final_mAP': final_val,
                'stability_last_10_epochs': std_last_10,
                'convergence_epoch': convergence_epoch or len(values),
                'convergence_ratio': (convergence_epoch or len(values)) / len(values)
            }

    return stability_analysis


def plot_training_efficiency(experiments, save_dir=None):
    """Plot training efficiency comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Convergence speed
    convergence_data = []
    stability_data = []

    for exp_name, df in experiments.items():
        if 'metrics/mAP_0.5' in df.columns:
            values = df['metrics/mAP_0.5'].values

            # Find epoch where mAP reaches 90% of final value
            final_val = values[-1]
            target_val = final_val * 0.9
            convergence_epoch = next((i for i, val in enumerate(
                values) if val >= target_val), len(values))

            convergence_data.append({
                'Experiment': exp_name,
                'Epochs to 90% Final mAP': convergence_epoch,
                'Final mAP': final_val
            })

            # Stability (std of last 25% of training)
            last_quarter = values[int(len(values)*0.75):]
            stability = np.std(last_quarter)

            stability_data.append({
                'Experiment': exp_name,
                'Stability (std)': stability,
                'Final mAP': final_val
            })

    # Plot convergence speed
    conv_df = pd.DataFrame(convergence_data)
    if not conv_df.empty:
        ax1.bar(conv_df['Experiment'], conv_df['Epochs to 90% Final mAP'])
        ax1.set_title('Convergence Speed')
        ax1.set_ylabel('Epochs to reach 90% of final mAP')
        ax1.tick_params(axis='x', rotation=45)

    # Plot stability
    stab_df = pd.DataFrame(stability_data)
    if not stab_df.empty:
        ax2.bar(stab_df['Experiment'], stab_df['Stability (std)'])
        ax2.set_title('Training Stability')
        ax2.set_ylabel('Standard deviation (last 25% epochs)')
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / "training_efficiency.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return conv_df, stab_df


def generate_report(experiments, save_dir=None):
    """Generate comprehensive analysis report"""
    report = []

    report.append("# YOLOv5 Optimization Techniques Comparison Report\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n")
    report.append(f"Total experiments analyzed: {len(experiments)}\n\n")

    # Summary table
    summary_df = create_summary_table(experiments)
    report.append("## Summary Table\n")
    report.append(summary_df.to_string(index=False))
    report.append("\n\n")

    # Best performing experiment
    if not summary_df.empty:
        best_exp = summary_df.loc[summary_df['Best mAP@0.5'].idxmax()]
        report.append(f"## Best Performing Experiment\n")
        report.append(f"**{best_exp['Experiment']}**\n")
        report.append(f"- Best mAP@0.5: {best_exp['Best mAP@0.5']:.4f}\n")
        report.append(
            f"- Best mAP@0.5:0.95: {best_exp['Best mAP@0.5:0.95']:.4f}\n")
        report.append(f"- Best Precision: {best_exp['Best Precision']:.4f}\n")
        report.append(f"- Best Recall: {best_exp['Best Recall']:.4f}\n")
        report.append(
            f"- Converged at epoch: {best_exp['Convergence Epoch']}\n\n")

    # Convergence analysis
    stability_analysis = analyze_convergence_stability(experiments)
    report.append("## Convergence Analysis\n")
    for exp_name, analysis in stability_analysis.items():
        report.append(f"### {exp_name}\n")
        report.append(f"- Max mAP@0.5: {analysis['max_mAP']:.4f}\n")
        report.append(f"- Final mAP@0.5: {analysis['final_mAP']:.4f}\n")
        report.append(
            f"- Convergence epoch: {analysis['convergence_epoch']}\n")
        report.append(
            f"- Stability (std last 10 epochs): {analysis['stability_last_10_epochs']:.6f}\n\n")

    # Recommendations
    report.append("## Recommendations\n")

    # Find best technique for each metric
    if not summary_df.empty:
        best_map05 = summary_df.loc[summary_df['Best mAP@0.5'].idxmax(),
                                    'Experiment']
        best_map0595 = summary_df.loc[summary_df['Best mAP@0.5:0.95'].idxmax(),
                                      'Experiment']
        fastest_convergence = min(
            stability_analysis.items(), key=lambda x: x[1]['convergence_epoch'])
        most_stable = min(stability_analysis.items(),
                          key=lambda x: x[1]['stability_last_10_epochs'])

        report.append(f"- **Best mAP@0.5**: {best_map05}\n")
        report.append(f"- **Best mAP@0.5:0.95**: {best_map0595}\n")
        report.append(
            f"- **Fastest convergence**: {fastest_convergence[0]} (epoch {fastest_convergence[1]['convergence_epoch']})\n")
        report.append(
            f"- **Most stable training**: {most_stable[0]} (std: {most_stable[1]['stability_last_10_epochs']:.6f})\n\n")

    report_text = "".join(report)

    if save_dir:
        with open(Path(save_dir) / "analysis_report.md", "w") as f:
            f.write(report_text)

    print(report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs/experiments",
                        help="Directory containing experiment results")
    parser.add_argument("--save-dir", default="analysis_results",
                        help="Directory to save analysis results")

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load results
    print("📊 Loading experiment results...")
    experiments = load_results(args.results_dir)

    if not experiments:
        print("❌ No experiment results found!")
        return

    print(f"✅ Loaded {len(experiments)} experiments")

    # Generate plots
    print("📈 Generating comparison plots...")
    plot_multiple_metrics(experiments, save_dir)
    plot_training_efficiency(experiments, save_dir)

    # Generate report
    print("📝 Generating analysis report...")
    generate_report(experiments, save_dir)

    print(f"✅ Analysis complete! Results saved to {save_dir}")


if __name__ == "__main__":
    main()
