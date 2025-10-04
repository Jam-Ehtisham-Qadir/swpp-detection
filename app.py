import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="SWPP Detection System",
    page_icon="🏗️",
    layout="wide"
)

# Title
st.title("🏗️ SWPP Element Detection System")
st.markdown("Upload a construction drawing PDF to detect SWPP elements")

# Class colors (BGR format for display)
CLASS_COLORS = {
    'Inlet_Protection': (255, 0, 255),      # Magenta
    'Washout_Pit': (0, 255, 0),             # Green
    'Skimmer': (255, 0, 0),                 # Blue
    'Sediment_Post': (255, 255, 0)          # Cyan
}

@st.cache_resource
def load_model():
    """Load YOLO model (cached)"""
    model_path = Path('models/best.pt')
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        st.stop()
    
    return YOLO(str(model_path))

def pdf_to_image(pdf_bytes):
    """Convert PDF first page to image"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    
    # Render at high resolution
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to numpy array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    doc.close()
    return img

def detect_and_visualize(model, image):
    """Run detection and draw bounding boxes"""
    # Run inference
    results = model.predict(image, conf=0.25, iou=0.5, verbose=False)
    
    # Count detections
    counts = {name: 0 for name in CLASS_COLORS.keys()}
    
    # Draw on image
    annotated_img = image.copy()
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            
            counts[class_name] = counts.get(class_name, 0) + 1
            
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Draw rectangle
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated_img, counts

# Main app
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Construction Drawing (PDF)",
        type=['pdf'],
        help="Upload a SWPP construction drawing in PDF format"
    )
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Convert PDF to image
            pdf_bytes = uploaded_file.read()
            image = pdf_to_image(pdf_bytes)
            
        with st.spinner("Detecting SWPP elements..."):
            # Run detection
            annotated_img, counts = detect_and_visualize(model, image)
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Detection Results")
            
            # Create metrics
            total = sum(counts.values())
            st.metric("Total Items Detected", total)
            
            st.markdown("---")
            
            # Individual counts
            for class_name, count in counts.items():
                color_hex = '#{:02x}{:02x}{:02x}'.format(
                    CLASS_COLORS[class_name][2],
                    CLASS_COLORS[class_name][1],
                    CLASS_COLORS[class_name][0]
                )
                st.markdown(
                    f"<span style='color:{color_hex}'>●</span> **{class_name}**: {count}",
                    unsafe_allow_html=True
                )
            
            # Download summary
            st.markdown("---")
            df = pd.DataFrame([{
                'Item': k,
                'Quantity': v,
                'Unit': 'EA'
            } for k, v in counts.items()])
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV Summary",
                csv,
                "swpp_detection_results.csv",
                "text/csv"
            )
        
        with col2:
            st.subheader("🎯 Annotated Drawing")
            
            # Convert BGR to RGB for display
            display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)
    
    else:
        # Instructions
        st.info("👆 Upload a PDF to get started")
        
        st.markdown("""
        ### How to use:
        1. Upload a construction drawing in PDF format
        2. Wait for detection to complete
        3. View results and download CSV summary
        
        ### Detected Elements:
        - 🟣 **Inlet Protection** (EA)
        - 🟢 **Washout Pit** (EA)
        - 🔵 **Skimmer** (EA)
        - 🔴 **Sediment Post** (EA)
        """)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Make sure the trained model is in the 'models' folder")

# Footer
st.markdown("---")
st.caption("SWPP Detection System v1.0 | Powered by YOLOv8")