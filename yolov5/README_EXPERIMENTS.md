# YOLOv5 Performance Optimization Experiments

โปรเจกต์นี้เปรียบเทียบเทคนิคต่างๆ เพื่อปรับปรุงประสิทธิภาพของ YOLOv5 model:

## 🎯 เทคนิคที่ทดสอบ

### 1. **Data Augmentation**
- Enhanced HSV augmentation
- Improved mosaic and mixup
- Random erasing และ crop
- Copy-paste augmentation

### 2. **Model Pruning**
- Structured pruning (30% filter removal)
- Model size reduction
- Maintained accuracy with faster inference

### 3. **Advanced Learning Rate Decay**
- Cosine Annealing with Warm Restarts
- Exponential Decay
- Polynomial Decay
- Step Decay

### 4. **Combined Techniques**
- รวมเทคนิคทั้งหมดเข้าด้วยกัน
- Optimized for best convergence

## 🚀 การใช้งาน

### รันการทดลองแบบครั้งเดียว:
```bash
# Baseline experiment
python train_enhanced.py --experiment baseline --model-type yolov5s

# Enhanced data augmentation
python train_enhanced.py --experiment augmented --model-type yolov5s

# Model pruning
python train_enhanced.py --experiment pruned --model-type yolov5s

# Advanced learning rate decay
python train_enhanced.py --experiment lr_decay --model-type yolov5s

# Combined techniques
python train_enhanced.py --experiment combined --model-type yolov5s
```

### รันการทดลองสำหรับ Segmentation:
```bash
python train_enhanced.py --experiment baseline --model-type yolov5s-seg
python train_enhanced.py --experiment augmented --model-type yolov5s-seg
# ... etc
```

### รันการทดลองทั้งหมดพร้อมกัน:
```bash
# YOLOv5s only
python run_experiments.py --model yolov5s

# YOLOv5s-seg only
python run_experiments.py --model yolov5s-seg

# Both models
python run_experiments.py --model both
```

### วิเคราะห์ผลลัพธ์:
```bash
python analyze_results.py --results-dir runs/experiments
```

## 📊 การดูผลลัพธ์

### TensorBoard:
```bash
tensorboard --logdir runs/experiments
```
เปิดเบราว์เซอร์ไปที่: http://localhost:6006/

### ไฟล์ผลลัพธ์:
```
runs/experiments/
├── yolov5s_baseline/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── tensorboard/
│   ├── results.csv
│   └── confusion_matrix.png
├── yolov5s_augmented/
├── yolov5s_pruned/
├── yolov5s_lr_decay/
├── yolov5s_combined/
└── analysis_results/
    ├── metrics_comparison.png
    ├── training_efficiency.png
    └── analysis_report.md
```

## 📈 คาดหวังผลลัพธ์

### Expected Performance Improvements:

1. **Enhanced Data Augmentation**
   - ✅ Better generalization
   - ✅ Reduced overfitting
   - ✅ Higher mAP@0.5 (+2-5%)

2. **Model Pruning**
   - ✅ 30% model size reduction
   - ✅ Faster inference time
   - ⚠️ Slight accuracy trade-off (<1%)

3. **Advanced Learning Rate Decay**
   - ✅ Better convergence
   - ✅ More stable training
   - ✅ Higher final accuracy

4. **Combined Techniques**
   - ✅ Best overall performance
   - ✅ Optimal convergence
   - ✅ Production-ready model

## 🔍 การวิเคราะห์

### Convergence Metrics:
- **mAP@0.5**: Primary metric
- **mAP@0.5:0.95**: Comprehensive accuracy
- **Precision**: Detection precision
- **Recall**: Detection coverage
- **Convergence Speed**: Epochs to reach 90% final performance
- **Training Stability**: Standard deviation in final epochs

### Comparison Criteria:
1. **Performance**: Best mAP scores
2. **Convergence**: Fastest to reach optimal performance
3. **Stability**: Most consistent training
4. **Efficiency**: Best accuracy/speed trade-off

## 📋 Requirements

```bash
pip install torch torchvision
pip install tensorboard
pip install matplotlib seaborn pandas
pip install ultralytics
```

## 🎛️ Hyperparameters

### Baseline:
- Learning Rate: 0.01
- Batch Size: 16
- Epochs: 50
- Optimizer: SGD
- Weight Decay: 0.0005

### Enhanced Settings:
- Advanced augmentation parameters
- Pruning ratio: 30%
- LR schedulers with optimal parameters
- Early stopping: 100 patience

## 📊 Expected Results Table

| Experiment | mAP@0.5 | mAP@0.5:0.95 | Convergence | Stability | Model Size |
|------------|---------|--------------|-------------|-----------|------------|
| Baseline   | 0.650   | 0.450        | 40 epochs   | High      | 100%       |
| Augmented  | 0.675   | 0.465        | 35 epochs   | High      | 100%       |
| Pruned     | 0.645   | 0.445        | 38 epochs   | Medium    | 70%        |
| LR Decay   | 0.670   | 0.460        | 30 epochs   | Very High | 100%       |
| Combined   | 0.685   | 0.475        | 32 epochs   | High      | 70%        |

## 🚨 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   python train_enhanced.py --batch-size 8  # Reduce batch size
   ```

2. **Model Download Issues**:
   ```bash
   # Download manually
   wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
   wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt
   ```

3. **Dataset Path Issues**:
   - Ensure `dataset_allBB/data.yaml` exists
   - Check image and label paths

### Performance Tips:

1. **For faster training**:
   ```bash
   --batch-size 32 --workers 16  # If you have enough GPU memory
   ```

2. **For quick testing**:
   ```bash
   --epochs 10  # Reduce epochs for testing
   ```

3. **For production**:
   ```bash
   --epochs 100 --patience 50  # Longer training for best results
   ```

## 📝 Notes

- การทดลองแต่ละครั้งจะใช้เวลาประมาณ 1-2 ชั่วโมง (50 epochs)
- รันการทดลองทั้งหมดจะใช้เวลาประมาณ 8-10 ชั่วโมง
- ผลลัพธ์จะแสดงการปรับปรุงที่ชัดเจนในการ converge
- สามารถปรับแต่ง hyperparameters เพิ่มเติมได้ในไฟล์ `train_enhanced.py`
