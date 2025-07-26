import streamlit as st
import cv2
import torch
from detector import ObjectDetector
from utils import draw_boxes
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Image Detection", page_icon="🖼️", layout="wide")
st.title("Image Object Detection")

# --- Sidebar for Settings ---
st.sidebar.header("Detection Settings")
model_descriptions = {
    'yolov8n': 'Nano model - Fastest but less accurate',
    'yolov8s': 'Small model - Good balance of speed and accuracy',
    'yolov8m': 'Medium model - Higher accuracy, moderate speed',
    'yolov8l': 'Large model - Very high accuracy, slower speed',
    'yolov8x': 'Extra Large model - Highest accuracy, slowest speed'
}
model_options = [f"{model} - {desc}" for model, desc in model_descriptions.items()]
selected_option = st.sidebar.selectbox("Select YOLOv8 Model", model_options, index=0)
selected_model = selected_option.split(' - ')[0]
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.01)

@st.cache_resource
def get_detector(model_name):
    return ObjectDetector(model_name)
detector = get_detector(selected_model)

all_class_names = list(detector.names.values())
options = ["All classes"] + all_class_names
if 'selections_img' not in st.session_state:
    st.session_state.selections_img = ["All classes"]
selected_options = st.sidebar.multiselect("What to detect", options, key='multiselect_img', default=st.session_state.selections_img)
if selected_options != st.session_state.selections_img:
    if "All classes" in selected_options and len(selected_options) > 1:
        st.session_state.selections_img = ["All classes"]
    elif len(selected_options) > 1 and "All classes" in selected_options:
        st.session_state.selections_img.remove("All classes")
    elif not selected_options:
        st.session_state.selections_img = ["All classes"]
    else:
        st.session_state.selections_img = selected_options
    st.rerun()
selected_class_ids = None
if "All classes" not in st.session_state.selections_img:
    selected_class_ids = [k for k, v in detector.names.items() if v in st.session_state.selections_img]

# --- Main Page ---
st.info(f"Model: **{selected_model}** | Device: **{next(detector.model.model.parameters()).device}**")
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Processing image..."):
        results = detector.detect(frame, conf_threshold=confidence_threshold, classes=selected_class_ids)
        processed_frame = draw_boxes(frame, results, detector.names)
        st.image(processed_frame, channels="BGR", caption="Processed Image", use_container_width=True)
    
    st.success("Processing complete!")

    save_button = st.button("Save Image", key="save_image_button")
    if save_button:
        base, ext = os.path.splitext(image_file.name)
        output_filename = f"{base}_processed.png"
        
        # --- This is the fix: Save to the parent directory ---
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_path = os.path.join(project_dir, output_filename)
        cv2.imwrite(output_path, processed_frame)
        st.success(f"Image saved as {output_filename} in your main project folder.")
else:
    st.info("Upload an image to get started.")