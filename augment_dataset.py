import cv2
import numpy as np
from pathlib import Path
import shutil

def augment_image_and_labels(img_path, label_path, output_img_dir, output_label_dir, base_idx):
    """Create augmented versions of image and labels"""
    img = cv2.imread(str(img_path))
    
    # Read labels
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    # Original
    cv2.imwrite(str(output_img_dir / f"aug_{base_idx}_orig.jpg"), img)
    with open(output_label_dir / f"aug_{base_idx}_orig.txt", 'w') as f:
        f.writelines(labels)
    
    # Brightness variations
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3]):
        bright_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        cv2.imwrite(str(output_img_dir / f"aug_{base_idx}_bright{i}.jpg"), bright_img)
        with open(output_label_dir / f"aug_{base_idx}_bright{i}.txt", 'w') as f:
            f.writelines(labels)
    
    # Contrast variations
    for i, factor in enumerate([0.8, 1.2]):
        contrast_img = cv2.convertScaleAbs(img, alpha=factor, beta=30)
        cv2.imwrite(str(output_img_dir / f"aug_{base_idx}_contrast{i}.jpg"), contrast_img)
        with open(output_label_dir / f"aug_{base_idx}_contrast{i}.txt", 'w') as f:
            f.writelines(labels)

# Create augmented dataset
input_img_dir = Path("data/dataset/images")
input_label_dir = Path("data/dataset/labels")
output_img_dir = Path("data/augmented_dataset/images")
output_label_dir = Path("data/augmented_dataset/labels")

output_img_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

print("Creating augmented dataset...")
all_images = sorted(list(input_img_dir.glob("*.jpg")))

for idx, img_path in enumerate(all_images):
    label_path = input_label_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        augment_image_and_labels(img_path, label_path, output_img_dir, output_label_dir, idx)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(all_images)}")

# Copy data.yaml
shutil.copy("data/dataset/data.yaml", "data/augmented_dataset/data.yaml")

# Update yaml path
import yaml
with open("data/augmented_dataset/data.yaml", 'r') as f:
    config = yaml.safe_load(f)

config['path'] = str(Path("data/augmented_dataset").absolute())
config['train'] = 'images'
config['val'] = 'images'

with open("data/augmented_dataset/data.yaml", 'w') as f:
    yaml.dump(config, f)

print(f"\nAugmented dataset created!")
print(f"Original: {len(all_images)} images")
print(f"Augmented: {len(list(output_img_dir.glob('*.jpg')))} images")