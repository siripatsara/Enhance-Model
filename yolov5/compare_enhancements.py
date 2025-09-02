# Compare Enhancement Techniques for YOLOv5
# This script runs experiments to compare baseline vs enhanced models

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Setup paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset_allBB/data.yaml"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "runs/train"


def run_experiment(name, weights, hyp_file, additional_args="", epochs=50):
    """Run a single training experiment"""
    print(f"\n{'='*60}")
    print(f"🚀 Running experiment: {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "train_tensor.py",
        "--weights", str(weights),
        "--data", str(DATA_PATH),
        "--hyp", str(hyp_file),
        "--epochs", str(epochs),
        "--name", name,
        "--device", "0",
        "--batch-size", "16"
    ]

    if additional_args:
        cmd.extend(additional_args.split())

    print(f"📝 Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if result.returncode == 0:
            print(f"✅ {name} completed successfully!")
            return True
        else:
            print(f"❌ {name} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {name} failed with exception: {e}")
        return False


def run_pruning_experiment(name, weights, hyp_file, pruning_method="magnitude", pruning_amount=0.2, epochs=50):
    """Run a pruning experiment"""
    print(f"\n{'='*60}")
    print(f"🔪 Running pruning experiment: {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "train_pruning.py",
        "--weights", str(weights),
        "--data", str(DATA_PATH),
        "--hyp", str(hyp_file),
        "--epochs", str(epochs),
        "--name", name,
        "--device", "0",
        "--batch-size", "16",
        "--pruning",
        "--pruning-method", pruning_method,
        "--pruning-amount", str(pruning_amount)
    ]

    print(f"📝 Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if result.returncode == 0:
            print(f"✅ {name} completed successfully!")
            return True
        else:
            print(f"❌ {name} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {name} failed with exception: {e}")
        return False


def extract_tensorboard_metrics(log_dir):
    """Extract metrics from TensorBoard logs"""
    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()

        metrics = {}

        # Extract scalar metrics
        scalar_tags = ea.Tags()['scalars']
        for tag in scalar_tags:
            if any(metric in tag.lower() for metric in ['map', 'loss', 'precision', 'recall']):
                steps, values = [], []
                for scalar_event in ea.Scalars(tag):
                    steps.append(scalar_event.step)
                    values.append(scalar_event.value)
                metrics[tag] = {'steps': steps, 'values': values}

        return metrics
    except Exception as e:
        print(f"Warning: Could not extract TensorBoard metrics: {e}")
        return {}


def parse_results_csv(results_path):
    """Parse results.csv file"""
    try:
        df = pd.read_csv(results_path)
        return df
    except Exception as e:
        print(f"Warning: Could not parse results.csv: {e}")
        return None


def create_comparison_plots(experiments_data, save_dir):
    """Create comparison plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Plot mAP@0.5 comparison
    plt.figure(figsize=(12, 8))

    for i, (name, data) in enumerate(experiments_data.items()):
        if 'metrics/mAP_0.5' in data:
            steps = data['metrics/mAP_0.5']['steps']
            values = data['metrics/mAP_0.5']['values']
            plt.plot(steps, values, label=name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.title('Model Performance Comparison - mAP@0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'map_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot training loss comparison
    plt.figure(figsize=(12, 8))

    for name, data in experiments_data.items():
        if 'train/box_loss' in data:
            steps = data['train/box_loss']['steps']
            values = data['train/box_loss']['values']
            plt.plot(steps, values, label=f'{name} - Box Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Comparison plots saved to: {save_dir}")


def create_convergence_analysis(experiments_data, save_dir):
    """Analyze convergence speed and final performance"""
    save_dir = Path(save_dir)
    analysis_file = save_dir / 'convergence_analysis.txt'

    with open(analysis_file, 'w') as f:
        f.write("Convergence Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        for name, data in experiments_data.items():
            f.write(f"Experiment: {name}\n")
            f.write("-" * 30 + "\n")

            if 'metrics/mAP_0.5' in data:
                map_values = data['metrics/mAP_0.5']['values']
                if map_values:
                    final_map = map_values[-1]
                    max_map = max(map_values)

                    # Find epoch where 90% of max performance is reached
                    target = 0.9 * max_map
                    convergence_epoch = None
                    for i, val in enumerate(map_values):
                        if val >= target:
                            convergence_epoch = i + 1
                            break

                    f.write(f"Final mAP@0.5: {final_map:.4f}\n")
                    f.write(f"Best mAP@0.5: {max_map:.4f}\n")
                    if convergence_epoch:
                        f.write(
                            f"90% convergence at epoch: {convergence_epoch}\n")
                    else:
                        f.write("90% convergence: Not reached\n")

            f.write("\n")

    print(f"📈 Convergence analysis saved to: {analysis_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLOv5 enhancement techniques")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs")
    parser.add_argument("--yolov5s", action="store_true",
                        help="Run YOLOv5s experiments")
    parser.add_argument("--yolov5s-seg", action="store_true",
                        help="Run YOLOv5s-seg experiments")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    args = parser.parse_args()

    if args.all:
        args.yolov5s = True
        args.yolov5s_seg = True

    if not (args.yolov5s or args.yolov5s_seg):
        print("Please specify --yolov5s, --yolov5s-seg, or --all")
        return

    experiments_data = {}

    # YOLOv5s experiments
    if args.yolov5s:
        print("\n🔍 Running YOLOv5s experiments...")

        # Baseline
        if run_experiment("yolov5s_baseline", "yolov5s.pt", "data/hyps/hyp.scratch-low.yaml", epochs=args.epochs):
            log_dir = RESULTS_DIR / "yolov5s_baseline"
            experiments_data["YOLOv5s Baseline"] = extract_tensorboard_metrics(
                log_dir)

        # Enhanced data augmentation
        if run_experiment("yolov5s_enhanced_aug", "yolov5s.pt", "data/hyps/hyp.enhanced.yaml", epochs=args.epochs):
            log_dir = RESULTS_DIR / "yolov5s_enhanced_aug"
            experiments_data["YOLOv5s Enhanced Aug"] = extract_tensorboard_metrics(
                log_dir)

        # With pruning
        if run_pruning_experiment("yolov5s_pruned", "yolov5s.pt", "data/hyps/hyp.enhanced.yaml",
                                  "magnitude", 0.2, epochs=args.epochs):
            log_dir = RESULTS_DIR / "yolov5s_pruned"
            experiments_data["YOLOv5s Pruned"] = extract_tensorboard_metrics(
                log_dir)

        # With cosine LR decay
        if run_experiment("yolov5s_cos_lr", "yolov5s.pt", "data/hyps/hyp.enhanced.yaml",
                          "--cos-lr", epochs=args.epochs):
            log_dir = RESULTS_DIR / "yolov5s_cos_lr"
            experiments_data["YOLOv5s Cosine LR"] = extract_tensorboard_metrics(
                log_dir)

    # YOLOv5s-seg experiments
    if args.yolov5s_seg:
        print("\n🔍 Running YOLOv5s-seg experiments...")

        # Check if segmentation weights exist
        seg_weights = "yolov5s-seg.pt"
        if not Path(seg_weights).exists():
            print(f"⚠️ {seg_weights} not found, downloading...")
            import torch
            torch.hub.download_url_to_file(
                f'https://github.com/ultralytics/yolov5/releases/download/v7.0/{seg_weights}',
                seg_weights
            )

        # Baseline segmentation
        if run_experiment("yolov5s_seg_baseline", seg_weights, "data/hyps/hyp.scratch-low.yaml", epochs=args.epochs):
            log_dir = RESULTS_DIR / "yolov5s_seg_baseline"
            experiments_data["YOLOv5s-seg Baseline"] = extract_tensorboard_metrics(
                log_dir)

        # Enhanced segmentation
        if run_experiment("yolov5s_seg_enhanced", seg_weights, "data/hyps/hyp.enhanced.yaml", epochs=args.epochs):
            log_dir = RESULTS_DIR / "yolov5s_seg_enhanced"
            experiments_data["YOLOv5s-seg Enhanced"] = extract_tensorboard_metrics(
                log_dir)

    # Create comparison analysis
    if experiments_data:
        comparison_dir = ROOT / "comparison_results"
        comparison_dir.mkdir(exist_ok=True)

        create_comparison_plots(experiments_data, comparison_dir)
        create_convergence_analysis(experiments_data, comparison_dir)

        print(f"\n🎉 All experiments completed!")
        print(f"📊 Results saved to: {comparison_dir}")
        print(f"🌐 View TensorBoard: tensorboard --logdir runs/train")
    else:
        print("❌ No experiments completed successfully!")


if __name__ == "__main__":
    main()
