# Lab-5 Dataset Comparison Summary
# Enhanced YOLOv5 Training with Multiple Techniques

## Dataset Configurations Available:

### 1. Original Dataset (Baseline)
- **File**: `data.yaml`
- **Training Images**: 144
- **Validation Images**: ~20
- **Use Case**: Baseline model comparison
- **Command**: `python train.py --data dataset_allBB/data.yaml --cfg yolov5s.yaml`

### 2. Augmented Dataset (Enhanced)
- **File**: `data_augmented.yaml` 
- **Training Images**: ~1,152 (8x expansion)
- **Validation Images**: ~202
- **Use Case**: Data augmentation technique demonstration
- **Command**: `python train.py --data dataset_allBB/data_augmented.yaml --cfg yolov5s.yaml --hyp hyp.augmented.yaml`

## Lab-5 Enhancement Techniques:

### ✅ 1. Data Augmentation
- **Implementation**: `simple_augment.py`
- **Techniques**: 7 types (rotation, brightness, noise, flip, HSV, blur, crop)
- **Dataset**: Use `data_augmented.yaml`
- **Expected Result**: Better convergence with more training data

### ✅ 2. Model Pruning  
- **Implementation**: `train_pruning.py`
- **Technique**: Structured and unstructured pruning
- **Dataset**: Can use either `data.yaml` or `data_augmented.yaml`
- **Expected Result**: Faster inference with maintained accuracy

### ✅ 3. Learning Rate Decay
- **Implementation**: Enhanced hyperparameter files
- **Files**: `hyp.enhanced.yaml`, `hyp.finetune.yaml`, `hyp.augmented.yaml`
- **Techniques**: Cosine annealing, step decay, polynomial decay
- **Expected Result**: Better convergence and final accuracy

## Recommended Training Experiments:

1. **Baseline**: Original data + default hyperparameters
2. **Data Aug**: Augmented data + optimized hyperparameters  
3. **Pruning**: Original data + pruning techniques
4. **LR Decay**: Original data + advanced learning rate scheduling
5. **Combined**: Augmented data + pruning + LR decay + optimized hyperparameters

## Files Created for Lab-5:
- `data_augmented.yaml` - Augmented dataset configuration
- `hyp.augmented.yaml` - Hyperparameters optimized for augmented data
- `simple_augment.py` - Data augmentation implementation
- `train_tensor.py` - Training with TensorBoard logging
- `train_pruning.py` - Training with model pruning
