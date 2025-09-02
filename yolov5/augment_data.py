# Data Augmentation Script for YOLOv5 Food Detection
# เพิ่มข้อมูลด้วยเทคนิคต่างๆ เพื่อปรับปรุงประสิทธิภาพโมเดล

import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class FoodDataAugmentation:
    """คลาสสำหรับเพิ่มข้อมูลสำหรับ food detection"""
    
    def __init__(self, source_dir, output_dir, labels_dir, output_labels_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.labels_dir = Path(labels_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # สร้างโฟลเดอร์ output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation pipeline
        self.augment_pipeline = A.Compose([
            # Geometric transformations
            A.OneOf([
                A.Rotate(limit=15, p=0.8),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8),
            ], p=0.7),
            
            # Color and lighting
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.8),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),
            ], p=0.8),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
            ], p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ], p=0.3),
            
            # Weather effects
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.2),
            ], p=0.2),
            
            # Perspective and distortion
            A.OneOf([
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.2),
            ], p=0.3),
            
            # Cutout and erasing (สำหรับทำให้โมเดลแข็งแกร่งขึ้น)
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.3),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=5, 
                               min_height=4, min_width=4, fill_value=0, p=0.3),
            ], p=0.2),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def load_yolo_labels(self, label_path):
        """อ่าน YOLO labels"""
        if not label_path.exists():
            return [], []
        
        bboxes = []
        class_labels = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def save_yolo_labels(self, bboxes, class_labels, output_path):
        """บันทึก YOLO labels"""
        with open(output_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_single_image(self, image_path, multiplier=3):
        """เพิ่มข้อมูลรูปเดียว"""
        print(f"🔄 Augmenting: {image_path.name}")
        
        # อ่านรูปและ labels
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        bboxes, class_labels = self.load_yolo_labels(label_path)
        
        # Copy original
        original_output = self.output_dir / image_path.name
        original_label_output = self.output_labels_dir / f"{image_path.stem}.txt"
        
        shutil.copy2(image_path, original_output)
        if label_path.exists():
            shutil.copy2(label_path, original_label_output)
        
        # สร้างรูปเพิ่ม
        for i in range(multiplier):
            try:
                # Apply augmentation
                if len(bboxes) > 0:
                    augmented = self.augment_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                else:
                    # ถ้าไม่มี bboxes ให้ augment แค่รูป
                    augmented = A.Compose([
                        A.Rotate(limit=15, p=0.8),
                        A.RandomBrightnessContrast(p=0.8),
                        A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                    ])(image=image)
                    aug_image = augmented['image']
                    aug_bboxes = []
                    aug_labels = []
                
                # บันทึกรูปใหม่
                aug_name = f"{image_path.stem}_aug_{i+1}{image_path.suffix}"
                aug_image_path = self.output_dir / aug_name
                aug_label_path = self.output_labels_dir / f"{image_path.stem}_aug_{i+1}.txt"
                
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_image_path), aug_image_bgr)
                
                # บันทึก labels
                if len(aug_bboxes) > 0:
                    self.save_yolo_labels(aug_bboxes, aug_labels, aug_label_path)
                elif label_path.exists():
                    # ถ้าไม่มี augmented bboxes แต่มี original labels ให้ copy
                    shutil.copy2(label_path, aug_label_path)
                    
            except Exception as e:
                print(f"❌ Error augmenting {image_path.name}: {e}")
                continue
    
    def augment_dataset(self, multiplier=3, file_extensions=['.jpg', '.jpeg', '.png']):
        """เพิ่มข้อมูลทั้ง dataset"""
        print(f"🚀 Starting data augmentation...")
        print(f"📁 Source: {self.source_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🔢 Multiplier: {multiplier}x")
        
        # หารูปทั้งหมด
        image_files = []
        for ext in file_extensions:
            image_files.extend(list(self.source_dir.glob(f"*{ext}")))
            image_files.extend(list(self.source_dir.glob(f"*{ext.upper()}")))
        
        print(f"📸 Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("❌ No images found!")
            return
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"📋 Progress: {i}/{len(image_files)}")
            self.augment_single_image(image_path, multiplier)
        
        # สร้าง train.txt และ val.txt ใหม่
        self.create_file_lists()
        
        print(f"✅ Data augmentation completed!")
        print(f"📈 Generated {len(image_files) * (multiplier + 1)} total images")
    
    def create_file_lists(self):
        """สร้าง train.txt และ val.txt สำหรับ augmented data"""
        output_images = list(self.output_dir.glob("*.jpg")) + list(self.output_dir.glob("*.png"))
        
        # สร้าง relative paths
        relative_paths = []
        for img_path in output_images:
            # สร้าง relative path จาก yolov5 root
            relative_path = f"dataset_allBB/images/train_augmented/{img_path.name}"
            relative_paths.append(relative_path)
        
        # แบ่ง train/val (90/10)
        random.shuffle(relative_paths)
        split_idx = int(len(relative_paths) * 0.9)
        
        train_paths = relative_paths[:split_idx]
        val_paths = relative_paths[split_idx:]
        
        # บันทึก train.txt
        train_file = self.output_dir.parent / "train_augmented.txt"
        with open(train_file, 'w') as f:
            for path in train_paths:
                f.write(f"{path}\n")
        
        # บันทึก val.txt  
        val_file = self.output_dir.parent / "val_augmented.txt"
        with open(val_file, 'w') as f:
            for path in val_paths:
                f.write(f"{path}\n")
        
        print(f"📝 Created train_augmented.txt ({len(train_paths)} images)")
        print(f"📝 Created val_augmented.txt ({len(val_paths)} images)")


def main():
    """ฟังก์ชันหลักสำหรับรัน data augmentation"""
    
    # กำหนด paths
    source_images = "dataset_allBB/images/train"
    output_images = "dataset_allBB/images/train_augmented" 
    source_labels = "dataset_allBB/labels/train"
    output_labels = "dataset_allBB/labels/train_augmented"
    
    print("🎯 YOLOv5 Food Detection Data Augmentation")
    print("=" * 50)
    
    # สร้าง augmentation object
    augmenter = FoodDataAugmentation(
        source_dir=source_images,
        output_dir=output_images,
        labels_dir=source_labels,
        output_labels_dir=output_labels
    )
    
    # เลือก multiplier
    multiplier = 3  # เพิ่มข้อมูล 3 เท่า (รวมของเดิม = 4 เท่า)
    
    print(f"📊 Augmentation Settings:")
    print(f"   - Rotation: ±15°")
    print(f"   - Brightness/Contrast: ±20%")
    print(f"   - HSV shifts: H±10, S±20, V±20")
    print(f"   - Gaussian noise: 10-50 variance")
    print(f"   - Motion blur: up to 3px")
    print(f"   - Perspective transform")
    print(f"   - Random shadows/sunflare")
    print(f"   - Cutout/dropout")
    print("")
    
    # รัน augmentation
    augmenter.augment_dataset(multiplier=multiplier)
    
    print("\n🎉 Data augmentation completed successfully!")
    print(f"📁 Check output folders:")
    print(f"   - Images: {output_images}")
    print(f"   - Labels: {output_labels}")
    print(f"   - File lists: dataset_allBB/train_augmented.txt, val_augmented.txt")


if __name__ == "__main__":
    main()
