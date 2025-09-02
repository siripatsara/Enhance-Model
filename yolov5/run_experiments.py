# Enhanced YOLOv5 Training Experiment Runner
"""
Script to run comprehensive experiments comparing different optimization techniques:
1. Baseline (original model)
2. Enhanced Data Augmentation
3. Model Pruning
4. Advanced Learning Rate Decay
5. Combined Techniques

Usage:
    python run_experiments.py --model yolov5s
    python run_experiments.py --model yolov5s-seg
    python run_experiments.py --model both  # Run both models
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_experiment(model_type, experiment, epochs=30, batch_size=16):
    """Run a single experiment"""
    cmd = [
        sys.executable, "train_enhanced.py",
        "--experiment", experiment,
        "--model-type", model_type,
        "--data", "dataset_allBB/data.yaml",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", "0",
        "--project", "runs/experiments",
        "--name", f"{model_type}_{experiment}"
    ]

    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error in {experiment} experiment:")
        print(result.stderr)
        return False
    else:
        print(f"✅ {experiment} experiment completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolov5s", "yolov5s-seg", "both"],
                        default="yolov5s", help="Model type to experiment with")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs per experiment")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="Batch size")

    args = parser.parse_args()

    models = ["yolov5s", "yolov5s-seg"] if args.model == "both" else [args.model]
    experiments = ["baseline", "augmented", "pruned", "lr_decay", "combined"]

    total_experiments = len(models) * len(experiments)
    current_exp = 0

    print(f"🧪 Starting {total_experiments} experiments...")
    print(f"📝 Models: {models}")
    print(f"🔬 Experiments: {experiments}")
    print(f"📊 Each experiment: {args.epochs} epochs")

    start_time = time.time()
    results = {}

    for model in models:
        results[model] = {}
        print(f"\n{'='*60}")
        print(f"🤖 Starting experiments with {model}")
        print(f"{'='*60}")

        for experiment in experiments:
            current_exp += 1
            print(
                f"\n[{current_exp}/{total_experiments}] Running {experiment} with {model}")

            success = run_experiment(
                model, experiment, args.epochs, args.batch_size)
            results[model][experiment] = success

            if not success:
                print(
                    f"⚠️ Experiment {experiment} failed, continuing with next...")

    total_time = (time.time() - start_time) / 3600

    # Print summary
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"⏱️ Total time: {total_time:.2f} hours")

    for model in models:
        print(f"\n🤖 {model}:")
        for experiment in experiments:
            status = "✅" if results[model][experiment] else "❌"
            print(f"   {status} {experiment}")

    print(f"\n📈 View results in TensorBoard:")
    print(f"tensorboard --logdir runs/experiments")
    print(f"🌐 http://localhost:6006/")

    print(f"\n📁 Results saved in: runs/experiments/")


if __name__ == "__main__":
    main()
