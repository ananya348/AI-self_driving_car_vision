import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Load model
model = YOLO("yolov8n.pt")

REAL_WIDTH = {
    "car": 1.8,
    "person": 0.5,
    "truck": 2.5,
    "bus": 2.5,
    "motorbike": 0.8
}

FOCAL_LENGTH = 700

# Distance estimation
def estimate_distance(box_width, label):
    if label in REAL_WIDTH and box_width > 0:
        return round((REAL_WIDTH[label] * FOCAL_LENGTH) / box_width, 2)
    return None

# Scene reasoning + risk level
def analyze_scene(objects):
    alerts = []
    risk = "LOW"

    for obj in objects:
        label = obj["label"]
        dist = obj["distance"]

        if label == "person" and dist and dist < 10:
            alerts.append("🚶 Pedestrian ahead → Slow down")
            risk = "MEDIUM"

        if label == "car" and dist and dist < 8:
            alerts.append("🚨 Collision risk with car")
            risk = "HIGH"

        if label == "traffic light":
            alerts.append("🚦 Traffic signal detected")

    if not alerts:
        alerts.append("✅ Safe driving conditions")

    return alerts, risk

# UI config
st.set_page_config(page_title="AI-Self-driving-Car-vision", layout="wide")

# 🔥 Tesla-style header
st.markdown(
    "<h1 style='text-align:center; color:#00FFCC;'>🚗 AI Driver Dashboard</h1>",
    unsafe_allow_html=True
)

uploaded_video = st.file_uploader("Upload Driving Video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    col1, col2 = st.columns([3,1])

    stframe = col1.empty()
    dashboard = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detected_objects = []

        # Object counters
        counts = {"car":0, "person":0, "truck":0, "bus":0}

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in counts:
                counts[label] += 1

            box_width = x2 - x1
            distance = estimate_distance(box_width, label)

            detected_objects.append({
                "label": label,
                "distance": distance
            })

            color = (0, 255, 0)
            if distance and distance < 8:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label}"
            if distance:
                text += f" {distance}m"

            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        alerts, risk = analyze_scene(detected_objects)

        # 🎯 Dashboard UI
        dashboard.markdown("## 📊 Dashboard")

        # Risk meter color
        if risk == "HIGH":
            dashboard.error(f"🚨 Risk Level: {risk}")
        elif risk == "MEDIUM":
            dashboard.warning(f"⚠️ Risk Level: {risk}")
        else:
            dashboard.success(f"✅ Risk Level: {risk}")

        dashboard.markdown("### 🚘 Object Count")
        dashboard.write(f"Cars: {counts['car']}")
        dashboard.write(f"People: {counts['person']}")
        dashboard.write(f"Trucks: {counts['truck']}")
        dashboard.write(f"Buses: {counts['bus']}")

        dashboard.markdown("### 🧠 Alerts")
        for alert in alerts:
            dashboard.write(alert)

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
