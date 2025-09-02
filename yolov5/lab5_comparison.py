#!/usr/bin/env python3
"""
Lab-5 Training Comparison Script
Enhanced YOLOv5 Models vs Baseline

This script demonstrates the effectiveness of enhancement techniques:
1. Data Augmentation
2. Model Pruning  
3. Learning Rate Decay

Usage:
    python lab5_comparison.py --experiments baseline,augmented,pruning,combined
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add YOLOv5 root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run_experiment(name, command, description):
    """Run a training experiment"""
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT: {name}")
    print(f"📝 DESCRIPTION: {description}")
    print(f"🔧 COMMAND: {command}")
    print(f"{'='*60}")

    # Create experiment directory
    exp_dir = f"runs/train/{name}"
    os.makedirs(exp_dir, exist_ok=True)

    # Run the training command
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        print(f"✅ {name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Lab-5 Training Comparison')
    parser.add_argument('--experiments',
                        default='baseline,augmented',
                        help='Comma-separated list of experiments to run')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')

    args = parser.parse_args()
    experiments = args.experiments.split(',')

    # Define experiments
    experiment_configs = {
        'baseline': {
            'description': 'Baseline YOLOv5s with original dataset',
            'command': f'python train.py --data dataset_allBB/data.yaml --cfg yolov5s.yaml --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --name baseline --project runs/train'
        },

        'augmented': {
            'description': 'Enhanced with Data Augmentation (8x more data)',
            'command': f'python train.py --data dataset_allBB/data_augmented.yaml --cfg yolov5s.yaml --hyp hyp.augmented.yaml --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --name augmented --project runs/train'
        },

        'pruning': {
            'description': 'Enhanced with Model Pruning',
            'command': f'python train_pruning.py --data dataset_allBB/data.yaml --cfg yolov5s.yaml --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --name pruning --project runs/train'
        },

        'lr_decay': {
            'description': 'Enhanced with Learning Rate Decay',
            'command': f'python train.py --data dataset_allBB/data.yaml --cfg yolov5s.yaml --hyp hyp.enhanced.yaml --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --name lr_decay --project runs/train'
        },

        'combined': {
            'description': 'All enhancements: Augmented data + Pruning + LR Decay',
            'command': f'python train_pruning.py --data dataset_allBB/data_augmented.yaml --cfg yolov5s.yaml --hyp hyp.augmented.yaml --epochs {args.epochs} --batch-size {args.batch_size} --img {args.img_size} --name combined --project runs/train'
        }
    }

    print("🚀 Starting Lab-5 Enhancement Comparison")
    print(f"📊 Running experiments: {', '.join(experiments)}")

    results = {}
    for exp_name in experiments:
        if exp_name in experiment_configs:
            config = experiment_configs[exp_name]
            success = run_experiment(
                exp_name, config['command'], config['description'])
            results[exp_name] = success
        else:
            print(f"⚠️  Unknown experiment: {exp_name}")

    # Summary
    print(f"\n{'='*60}")
    print("📈 EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{exp_name:15}: {status}")

    print(f"\n🔍 To compare results:")
    print(f"tensorboard --logdir runs/train")
    print(f"\n📊 Results saved in: runs/train/")


if __name__ == '__main__':
    main()
