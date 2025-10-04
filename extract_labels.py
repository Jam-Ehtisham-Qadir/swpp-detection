import cv2
import numpy as np
from pathlib import Path
import yaml
import re

def normalize_filename(filename):
    """
    Normalize filename for matching by removing markup indicators
    """
    # Remove file extension
    name = filename.rsplit('.', 1)[0]
    
    # Remove common markup phrases
    markup_patterns = [
        r' - markup md',
        r' MARKED UP CJA',
        r'MARKED UP',
        r'marked up cja',
        r'marked up',
        r' markup md',
        r'_markup_md',
    ]
    
    for pattern in markup_patterns:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name.lower().strip()

def find_matching_file(target_file, search_folder):
    """
    Find matching file in another folder by normalized name
    """
    target_normalized = normalize_filename(target_file.name)
    
    # Get all images in search folder
    all_files = list(search_folder.glob("*.png")) + \
                list(search_folder.glob("*.jpg")) + \
                list(search_folder.glob("*.jpeg"))
    
    for file in all_files:
        file_normalized = normalize_filename(file.name)
        if target_normalized == file_normalized:
            return file
    
    return None

def extract_labels_from_images(original_img_path, annotated_img_path):
    """
    Compare original and annotated images to extract bounding boxes
    """
    # Read images
    orig_img = cv2.imread(str(original_img_path))
    annot_img = cv2.imread(str(annotated_img_path))
    
    if orig_img is None or annot_img is None:
        raise Exception(f"Could not read images")
    
    # Resize if needed
    if orig_img.shape != annot_img.shape:
        annot_img = cv2.resize(annot_img, (orig_img.shape[1], orig_img.shape[0]))
    
    # Define color ranges (BGR format)
    color_ranges = {
        0: {
            'name': 'Inlet_Protection',
            'lower': np.array([100, 200, 200]),    # Cyan/Yellow range
            'upper': np.array([150, 255, 255])
        },
        1: {
            'name': 'Washout_Pit',
            'lower': np.array([0, 0, 200]),        # Red range
            'upper': np.array([50, 50, 255])
        },
        2: {
            'name': 'Skimmer',
            'lower': np.array([150, 0, 0]),        # Dark Blue range
            'upper': np.array([170, 20, 20])
        },
        3: {
            'name': 'Sediment_Post',
            'lower': np.array([100, 0, 200]),      # Magenta/Purple range
            'upper': np.array([150, 50, 255])
        }
    }
    
    labels = []
    img_height, img_width = annot_img.shape[:2]
    
    for class_id, color_info in color_ranges.items():
        mask = cv2.inRange(annot_img, color_info['lower'], color_info['upper'])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 5 or h < 5:
                continue
            
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
    
    return labels, orig_img

def process_google_drive_images():
    """
    Process all images with smart filename matching
    """
    base_path = Path("google_drive_data")
    original_img_folder = base_path / "original_images"
    annotated_img_folder = base_path / "annotated_images"
    
    output_base = Path("data/dataset")
    images_dir = output_base / "images"
    labels_dir = output_base / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting label extraction with smart filename matching...")
    print(f"Original images folder: {original_img_folder}")
    print(f"Annotated images folder: {annotated_img_folder}")
    
    original_images = sorted(list(original_img_folder.glob("*.png")) + 
                            list(original_img_folder.glob("*.jpg")) + 
                            list(original_img_folder.glob("*.jpeg")))
    
    if len(original_images) == 0:
        print(f"ERROR: No images found")
        return 0
    
    print(f"Found {len(original_images)} original images")
    
    processed = 0
    errors = []
    
    for idx, orig_img_path in enumerate(original_images):
        try:
            # Find matching annotated image
            annot_img_path = find_matching_file(orig_img_path, annotated_img_folder)
            
            if annot_img_path is None:
                errors.append(f"No match for: {orig_img_path.name}")
                continue
            
            # Extract labels
            labels, image = extract_labels_from_images(orig_img_path, annot_img_path)
            
            # Save
            image_filename = f"doc_{processed:03d}.jpg"
            cv2.imwrite(str(images_dir / image_filename), image)
            
            label_filename = f"doc_{processed:03d}.txt"
            with open(labels_dir / label_filename, 'w') as f:
                f.write('\n'.join(labels))
            
            processed += 1
            print(f"[{processed}] {orig_img_path.name[:50]}... -> {len(labels)} objects")
            
        except Exception as e:
            errors.append(f"{orig_img_path.name}: {str(e)}")
            print(f"ERROR: {str(e)}")
    
    # Create config
    dataset_config = {
        'path': str(output_base.absolute()),
        'train': 'images',
        'val': 'images',
        'names': {
            0: 'Inlet_Protection',
            1: 'Washout_Pit',
            2: 'Skimmer',
            3: 'Sediment_Post'
        }
    }
    
    with open(output_base / 'data.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed: {processed} documents")
    print(f"📁 Images: {images_dir}")
    print(f"📁 Labels: {labels_dir}")
    print(f"📄 Config: {output_base / 'data.yaml'}")
    
    if errors:
        print(f"\n⚠️ Errors: {len(errors)}")
        for err in errors[:10]:
            print(f"  - {err}")
    
    return processed

if __name__ == "__main__":
    num_processed = process_google_drive_images()
    
    if num_processed > 0:
        print(f"\n🎉 Dataset ready for training!")
    else:
        print(f"\n❌ Failed. Check errors above.")