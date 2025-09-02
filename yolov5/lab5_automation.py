# Lab-5: Model Enhancement Automation Script
# สำหรับงาน: "ปรับปรุงประสิทธิภาพ YOLOv5 ด้วยเทคนิค Data Augmentation, Model Pruning, และ Decay Learning Rate Function"

import os
import sys
import subprocess
import time
from pathlib import Path


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "train_tensor.py",
        "train_pruning.py",
        "compare_enhancements.py",
        "data/hyps/hyp.enhanced.yaml",
        "dataset_allBB/data.yaml"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    print("✅ All required files found!")
    return True


def download_weights():
    """Download required weights if not present"""
    weights = ["yolov5s.pt", "yolov5s-seg.pt"]

    for weight in weights:
        if not Path(weight).exists():
            print(f"📥 Downloading {weight}...")
            try:
                import torch
                url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{weight}"
                torch.hub.download_url_to_file(url, weight)
                print(f"✅ {weight} downloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to download {weight}: {e}")
                return False

    return True


def run_lab5_experiments():
    """Run complete Lab-5 experiments"""

    print_banner("🚀 Lab-5: YOLOv5 Model Enhancement Experiments")

    # Check requirements
    if not check_requirements():
        print("Please ensure all required files are present.")
        return False

    # Download weights
    if not download_weights():
        print("Failed to download required weights.")
        return False

    print_banner("📋 Experiment Plan")
    print("""
    เทคนิคที่จะทดสอบ:
    1. 📊 Data Augmentation (Enhanced hyperparameters)
    2. ✂️  Model Pruning (Magnitude-based pruning)
    3. 📉 Decay Learning Rate Function (Cosine annealing)
    
    โมเดลที่จะทดสอบ:
    - YOLOv5s (Object Detection)
    - YOLOv5s-seg (Instance Segmentation)
    
    การเปรียบเทียบ:
    - Baseline models vs Enhanced models
    - Convergence analysis
    - Performance metrics (mAP@0.5)
    """)

    # Get user confirmation
    choice = input(
        "\n🤔 Ready to start experiments? This will take approximately 2-3 hours. (y/n): ")
    if choice.lower() != 'y':
        print("Experiments cancelled.")
        return False

    # Start experiments
    print_banner("🏃‍♂️ Starting Experiments")

    try:
        # Run comparison script
        cmd = [sys.executable, "compare_enhancements.py",
               "--all", "--epochs", "30"]
        print(f"📝 Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=Path.cwd())

        if result.returncode == 0:
            print_banner("🎉 Experiments Completed Successfully!")
            print("""
            📊 Results Location:
            - Comparison plots: comparison_results/
            - TensorBoard logs: runs/train/
            - Model weights: runs/train/*/weights/
            
            📈 To view results:
            1. Check comparison_results/ folder for plots
            2. Run: tensorboard --logdir runs/train
            3. Open browser: http://localhost:6006
            """)
            return True
        else:
            print("❌ Experiments failed!")
            return False

    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        return False


def create_lab_report():
    """Create lab report template"""
    report_content = """# Lab-5 Report: YOLOv5 Model Enhancement

## วัตถุประสงค์
ปรับปรุงประสิทธิภาพ YOLOv5 ด้วยเทคนิค Data Augmentation, Model Pruning, และ Decay Learning Rate Function

## เทคนิคที่ใช้

### 1. Data Augmentation
- Enhanced HSV augmentation: hsv_h=0.015, hsv_s=0.6, hsv_v=0.4
- Geometric transforms: rotation, scaling, translation
- Advanced techniques: Mosaic=1.0, Mixup=0.1
- Label smoothing: 0.1

### 2. Model Pruning
- Method: Magnitude-based unstructured pruning
- Pruning ratio: 20%
- Target: Reduce model size while maintaining accuracy

### 3. Decay Learning Rate Function
- Cosine annealing scheduler
- Smooth learning rate decay
- Better convergence properties

## ผลการทดลอง

### YOLOv5s Results
[เติมผลการทดลองที่นี่]

### YOLOv5s-seg Results
[เติมผลการทดลองที่นี่]

## การวิเคราะห์
[วิเคราะห์ผลการทดลองว่าเทคนิคไหนช่วยปรับปรุงประสิทธิภาพได้ดีที่สุด]

## สรุป
[สรุปผลการทดลองและข้อเสนอแนะ]
"""

    with open("Lab5_Report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("📝 Lab report template created: Lab5_Report.md")


def main():
    """Main function"""
    print_banner("🎓 Lab-5 Automation Script")
    print("YOLOv5 Model Enhancement with Multiple Techniques")

    while True:
        print("\n📋 Menu:")
        print("1. 🏃‍♂️ Run complete Lab-5 experiments")
        print("2. 📊 Check experiment status")
        print("3. 📝 Create lab report template")
        print("4. ❌ Exit")

        choice = input("\nSelect option (1-4): ")

        if choice == "1":
            run_lab5_experiments()
        elif choice == "2":
            # Check if results exist
            results_dir = Path("comparison_results")
            runs_dir = Path("runs/train")

            if results_dir.exists() and any(results_dir.iterdir()):
                print("✅ Experiment results found!")
                print(f"📊 Comparison plots: {results_dir}")
                if runs_dir.exists():
                    exp_dirs = list(runs_dir.glob("yolov5s*"))
                    print(f"🏃‍♂️ Completed experiments: {len(exp_dirs)}")
                    for exp_dir in exp_dirs:
                        print(f"   - {exp_dir.name}")
            else:
                print("❌ No experiment results found. Please run experiments first.")
        elif choice == "3":
            create_lab_report()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
