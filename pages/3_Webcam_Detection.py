import streamlit as st
import cv2
import torch
from detector import ObjectDetector
from utils import draw_boxes
import time
import os

st.set_page_config(page_title="Webcam Detection", page_icon="📸", layout="wide")
st.title("Webcam Live Detection")

# --- Sidebar Settings ---
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
if 'selections_cam' not in st.session_state:
    st.session_state.selections_cam = ["All classes"]
selected_options = st.sidebar.multiselect("What to detect", options, key='multiselect_cam', default=st.session_state.selections_cam)
if selected_options != st.session_state.selections_cam:
    if "All classes" in selected_options and len(selected_options) > 1:
        st.session_state.selections_cam = ["All classes"]
    elif len(selected_options) > 1 and "All classes" in selected_options:
        st.session_state.selections_cam.remove("All classes")
    elif not selected_options:
        st.session_state.selections_cam = ["All classes"]
    else:
        st.session_state.selections_cam = selected_options
    st.rerun()
selected_class_ids = None
if "All classes" not in st.session_state.selections_cam:
    selected_class_ids = [k for k, v in detector.names.items() if v in st.session_state.selections_cam]

# --- Main Page ---
st.info(f"Model: **{selected_model}** | Device: **{next(detector.model.model.parameters()).device}**")
run = st.checkbox('Start Webcam')

if run:
    record_video = st.checkbox('Record Video')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    st.sidebar.subheader("Performance")
    fps_placeholder = st.sidebar.empty()
    prev_time = time.time()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from webcam. Please check camera connection.")
            break
            
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_placeholder.markdown(f"**FPS:** `{fps:.2f}`")

        # Recording logic
        if record_video and 'is_recording' not in st.session_state:
            st.session_state.is_recording = True
            
            # --- This is the fix: Save to the parent directory ---
            project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            filename = os.path.join(project_dir, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi")
            
            st.session_state.video_filename = filename
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            record_fps = 20.0
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            st.session_state.video_writer = cv2.VideoWriter(filename, fourcc, record_fps, frame_size)
            st.toast(f"🔴 Recording started!")

        if not record_video and 'is_recording' in st.session_state:
            st.session_state.video_writer.release()
            del st.session_state.is_recording
            st.toast(f"⚫ Recording stopped! Saved as `{os.path.basename(st.session_state.video_filename)}`.")

        results = detector.detect(frame, conf_threshold=confidence_threshold, classes=selected_class_ids)
        processed_frame = draw_boxes(frame, results, detector.names)

        if record_video and 'is_recording' in st.session_state:
            st.session_state.video_writer.write(processed_frame)

        FRAME_WINDOW.image(processed_frame, channels='BGR')
    
    cap.release()
    if 'is_recording' in st.session_state:
        st.session_state.video_writer.release()
        del st.session_state.is_recording
else:
    st.info("Check the 'Start Webcam' box to begin detection.")