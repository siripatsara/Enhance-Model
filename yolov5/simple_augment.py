# Simple Data Augmentation using OpenCV only
# ไม่ต้องใช้ external libraries เพิ่ม

import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
import math


class SimpleDataAugmentation:
    """คลาสสำหรับเพิ่มข้อมูลด้วย OpenCV และ NumPy เท่านั้น"""

    def __init__(self, source_dir, output_dir, labels_dir, output_labels_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.labels_dir = Path(labels_dir)
        self.output_labels_dir = Path(output_labels_dir)

        # สร้างโฟลเดอร์ output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

    def load_yolo_labels(self, label_path):
        """อ่าน YOLO labels"""
        if not label_path.exists():
            return []

        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0])] + [float(x)
                                  for x in parts[1:]])
        return labels

    def save_yolo_labels(self, labels, output_path):
        """บันทึก YOLO labels"""
        with open(output_path, 'w') as f:
            for label in labels:
                class_id = int(label[0])
                x_center, y_center, width, height = label[1:]
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def rotate_image_and_labels(self, image, labels, angle):
        """หมุนรูปและ bounding boxes"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # สร้าง rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # หมุนรูป
        rotated = cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # หมุน bounding boxes (simplified - อาจไม่แม่นยำ 100%)
        rotated_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label

            # Convert to absolute coordinates
            abs_x = x_center * w
            abs_y = y_center * h

            # Apply rotation (simplified)
            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))

            new_x = cos_a * (abs_x - center[0]) - \
                sin_a * (abs_y - center[1]) + center[0]
            new_y = sin_a * (abs_x - center[0]) + \
                cos_a * (abs_y - center[1]) + center[1]

            # Convert back to relative coordinates
            new_x_rel = max(0, min(1, new_x / w))
            new_y_rel = max(0, min(1, new_y / h))

            rotated_labels.append(
                [class_id, new_x_rel, new_y_rel, width, height])

        return rotated, rotated_labels

    def adjust_brightness_contrast(self, image, brightness=0, contrast=1):
        """ปรับความสว่างและ contrast"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def add_gaussian_noise(self, image, variance=25):
        """เพิ่ม Gaussian noise"""
        noise = np.random.normal(
            0, variance**0.5, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy

    def horizontal_flip(self, image, labels):
        """พลิกแนวนอนและปรับ bounding boxes"""
        flipped = cv2.flip(image, 1)

        flipped_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label
            # พลิก x coordinate
            new_x_center = 1.0 - x_center
            flipped_labels.append(
                [class_id, new_x_center, y_center, width, height])

        return flipped, flipped_labels

    def change_hue_saturation(self, image, hue_shift=0, sat_scale=1.0, val_scale=1.0):
        """เปลี่ยน HSV values"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # ปรับ Hue
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # ปรับ Saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)

        # ปรับ Value (brightness)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def motion_blur(self, image, size=5):
        """เพิ่ม motion blur"""
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(image, -1, kernel)

    def random_crop_and_resize(self, image, labels, crop_factor=0.8):
        """Crop แบบสุ่มแล้ว resize กลับ"""
        h, w = image.shape[:2]

        # สุ่มขนาด crop
        new_w = int(w * crop_factor)
        new_h = int(h * crop_factor)

        # สุ่มตำแหน่ง crop
        x_start = random.randint(0, w - new_w)
        y_start = random.randint(0, h - new_h)

        # Crop
        cropped = image[y_start:y_start+new_h, x_start:x_start+new_w]

        # Resize กลับ
        resized = cv2.resize(cropped, (w, h))

        # ปรับ bounding boxes (simplified)
        adjusted_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label

            # Convert to absolute coordinates
            abs_x = x_center * w
            abs_y = y_center * h
            abs_w = width * w
            abs_h = height * h

            # ตรวจสอบว่า bbox อยู่ใน crop area หรือไม่
            if (abs_x - abs_w/2 > x_start and abs_x + abs_w/2 < x_start + new_w and
                    abs_y - abs_h/2 > y_start and abs_y + abs_h/2 < y_start + new_h):

                # ปรับตำแหน่งตาม crop
                new_abs_x = (abs_x - x_start) * w / new_w
                new_abs_y = (abs_y - y_start) * h / new_h
                new_abs_w = abs_w * w / new_w
                new_abs_h = abs_h * h / new_h

                # Convert back to relative
                new_x_rel = new_abs_x / w
                new_y_rel = new_abs_y / h
                new_w_rel = new_abs_w / w
                new_h_rel = new_abs_h / h

                adjusted_labels.append(
                    [class_id, new_x_rel, new_y_rel, new_w_rel, new_h_rel])

        return resized, adjusted_labels

    def augment_single_image(self, image_path, multiplier=3):
        """เพิ่มข้อมูลรูปเดียว"""
        print(f"🔄 Augmenting: {image_path.name}")

        # อ่านรูปและ labels
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        labels = self.load_yolo_labels(label_path)

        # Copy original
        original_output = self.output_dir / image_path.name
        original_label_output = self.output_labels_dir / \
            f"{image_path.stem}.txt"

        shutil.copy2(image_path, original_output)
        if label_path.exists():
            shutil.copy2(label_path, original_label_output)

        # สร้างรูปเพิ่ม
        augmentations = [
            ("rotate", lambda img, lbl: self.rotate_image_and_labels(
                img, lbl, random.uniform(-15, 15))),
            ("bright", lambda img, lbl: (self.adjust_brightness_contrast(
                img, random.randint(-30, 30), random.uniform(0.8, 1.2)), lbl)),
            ("noise", lambda img, lbl: (self.add_gaussian_noise(
                img, random.randint(10, 30)), lbl)),
            ("flip", lambda img, lbl: self.horizontal_flip(img, lbl)),
            ("hsv", lambda img, lbl: (self.change_hue_saturation(
                img, random.randint(-10, 10), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)), lbl)),
            ("blur", lambda img, lbl: (self.motion_blur(
                img, random.choice([3, 5, 7])), lbl)),
            ("crop", lambda img, lbl: self.random_crop_and_resize(
                img, lbl, random.uniform(0.7, 0.9))),
        ]

        for i in range(multiplier):
            try:
                # เลือก augmentation แบบสุ่ม 2-3 อัน
                selected_augs = random.sample(
                    augmentations, random.randint(2, 3))

                aug_image = image.copy()
                aug_labels = labels.copy()

                aug_name_parts = []

                # Apply augmentations
                for aug_name, aug_func in selected_augs:
                    aug_image, aug_labels = aug_func(aug_image, aug_labels)
                    aug_name_parts.append(aug_name)

                # บันทึกรูปใหม่
                aug_suffix = "_".join(aug_name_parts)
                aug_name = f"{image_path.stem}_aug_{i+1}_{aug_suffix}{image_path.suffix}"
                aug_image_path = self.output_dir / aug_name
                aug_label_path = self.output_labels_dir / \
                    f"{image_path.stem}_aug_{i+1}_{aug_suffix}.txt"

                cv2.imwrite(str(aug_image_path), aug_image)

                # บันทึก labels
                if len(aug_labels) > 0:
                    self.save_yolo_labels(aug_labels, aug_label_path)
                elif label_path.exists():
                    shutil.copy2(label_path, aug_label_path)

            except Exception as e:
                print(f"❌ Error augmenting {image_path.name}: {e}")
                continue

    def augment_dataset(self, multiplier=3):
        """เพิ่มข้อมูลทั้ง dataset"""
        print(f"🚀 Starting simple data augmentation...")
        print(f"📁 Source: {self.source_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🔢 Multiplier: {multiplier}x")

        # หารูปทั้งหมด
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(list(self.source_dir.glob(f"*{ext}")))

        print(f"📸 Found {len(image_files)} images")

        if len(image_files) == 0:
            print("❌ No images found!")
            return

        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"📋 Progress: {i}/{len(image_files)}")
            self.augment_single_image(image_path, multiplier)

        # สร้าง file lists
        self.create_file_lists()

        print(f"✅ Data augmentation completed!")
        print(
            f"📈 Generated approximately {len(image_files) * (multiplier + 1)} total images")

    def create_file_lists(self):
        """สร้าง train.txt และ val.txt สำหรับ augmented data"""
        output_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            output_images.extend(list(self.output_dir.glob(f"*{ext}")))

        # สร้าง relative paths
        relative_paths = []
        for img_path in output_images:
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

    print("🎯 Simple YOLOv5 Food Detection Data Augmentation")
    print("=" * 55)
    print("🔧 Using OpenCV and NumPy only (no external dependencies)")
    print("")

    # ตรวจสอบว่ามี source directory หรือไม่
    if not Path(source_images).exists():
        print(f"❌ Source directory not found: {source_images}")
        print("Please make sure you're in the YOLOv5 directory and have the dataset ready.")
        return

    # สร้าง augmentation object
    augmenter = SimpleDataAugmentation(
        source_dir=source_images,
        output_dir=output_images,
        labels_dir=source_labels,
        output_labels_dir=output_labels
    )

    # เลือก multiplier
    multiplier = 3  # เพิ่มข้อมูล 3 เท่า (รวมของเดิม = 4 เท่า)

    print(f"📊 Augmentation Techniques:")
    print(f"   ✅ Rotation: ±15°")
    print(f"   ✅ Brightness/Contrast adjustment")
    print(f"   ✅ Gaussian noise")
    print(f"   ✅ Horizontal flip")
    print(f"   ✅ HSV color space shifts")
    print(f"   ✅ Motion blur")
    print(f"   ✅ Random crop and resize")
    print(f"   🔢 Multiplier: {multiplier}x")
    print("")

    # ถาม user ก่อนเริ่ม
    response = input("🤔 Ready to start augmentation? (y/n): ")
    if response.lower() != 'y':
        print("❌ Augmentation cancelled.")
        return

    # รัน augmentation
    augmenter.augment_dataset(multiplier=multiplier)

    print("\n🎉 Data augmentation completed successfully!")
    print(f"📁 Check output folders:")
    print(f"   - Images: {output_images}")
    print(f"   - Labels: {output_labels}")
    print(f"   - File lists: dataset_allBB/train_augmented.txt, val_augmented.txt")
    print("")
    print("🚀 Next steps:")
    print("   1. Use hyp.augmented.yaml for training")
    print("   2. Update data.yaml to point to augmented files")
    print("   3. Train with: python train_tensor.py --data data_augmented.yaml")


if __name__ == "__main__":
    main()
