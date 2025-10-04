import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# Check 5 different annotated images
annotated_folder = Path("google_drive_data/annotated_images")
all_images = list(annotated_folder.glob("*.png"))[:5]

for img_path in all_images:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Sample pixels
    sample_points = []
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            pixel = img[y, x]
            if np.mean(pixel) < 240 and np.mean(pixel) > 50:  # Not white or black
                sample_points.append(tuple(pixel))
    
    color_counts = Counter(sample_points)
    top_colors = color_counts.most_common(5)
    
    print(f"\n{img_path.name[:50]}...")
    for color, count in top_colors:
        print(f"  BGR: {color}")