from ultralytics import YOLO
from pathlib import Path

def train_model():
    """
    Train YOLOv8 model on extracted dataset
    """
    print("Starting YOLO training...")
    print("This will take 2-4 hours on CPU, or 30-60 mins on GPU")
    
    # Load pretrained YOLOv8 nano model (smallest, fastest)
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data='data/dataset/data.yaml',
        epochs=50,                    # Number of training iterations
        imgsz=640,                    # Image size (smaller = faster on CPU)
        batch=8,                      # Batch size (smaller = less memory)
        patience=10,                  # Early stopping patience
        device='cpu',                 # Use CPU (change to 0 for GPU)
        project='models',
        name='swpp_run',
        exist_ok=True,
        
        # Optimizations for CPU training
        workers=2,                    # Number of data loading workers
        cache=False,                  # Don't cache images (saves RAM)
        
        # Data augmentation (helps with small dataset)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved at: models/swpp_run/weights/best.pt")
    print("="*60)
    
    return results

if __name__ == "__main__":
    train_model()