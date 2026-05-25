"""
object_detection.py  —  Task 4: Object Detection & Tracking
-------------------------------------------------------------
✅ Real-time webcam input          (OpenCV VideoCapture)
✅ Pre-trained MobileNet-SSD       (Caffe model)
✅ Bounding boxes per frame        (cv2.dnn SSD inference)
✅ Object tracking with IDs        (Simple centroid tracker — no extra install)
✅ Labels + tracking IDs on screen (drawn with cv2.putText)

Usage:
    streamlit run object_detection.py

Requirements:
    pip install streamlit opencv-python-headless numpy requests
"""

import os
import math
import urllib.request
from collections import OrderedDict

import cv2
import numpy as np
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Object Detection & Tracking", page_icon="🎯", layout="wide")
st.title("🎯 Object Detection & Tracking — MobileNet-SSD")
st.caption("Real-time detection with centroid-based object tracking IDs.")

# ── VOC classes ────────────────────────────────────────────────────────────────
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

# ── Centroid Tracker ───────────────────────────────────────────────────────────
class CentroidTracker:
    """
    Assigns a unique integer ID to each detected object and tracks it
    across frames using centroid distance — no extra library needed.
    """
    def __init__(self, max_disappeared=30):
        self.next_id       = 0
        self.objects       = OrderedDict()   # id → centroid
        self.disappeared   = OrderedDict()   # id → frames unseen
        self.labels        = OrderedDict()   # id → class label
        self.max_disappeared = max_disappeared

    def _register(self, centroid, label):
        self.objects[self.next_id]    = centroid
        self.disappeared[self.next_id] = 0
        self.labels[self.next_id]     = label
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.labels[oid]

    def update(self, detections):
        """
        detections: list of (label, cx, cy)
        Returns:    dict  id → (cx, cy, label)
        """
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self._output()

        if not self.objects:
            for (label, cx, cy) in detections:
                self._register((cx, cy), label)
            return self._output()

        obj_ids    = list(self.objects.keys())
        obj_cents  = list(self.objects.values())
        det_cents  = [(cx, cy) for (_, cx, cy) in detections]
        det_labels = [label    for (label, _, __) in detections]

        # Build distance matrix
        D = np.zeros((len(obj_cents), len(det_cents)))
        for r, oc in enumerate(obj_cents):
            for c, dc in enumerate(det_cents):
                D[r, c] = math.hypot(oc[0]-dc[0], oc[1]-dc[1])

        # Greedy match: closest pairs first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = obj_ids[r]
            self.objects[oid]    = det_cents[c]
            self.labels[oid]     = det_labels[c]
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        # Handle unmatched existing objects
        for r in range(len(obj_ids)):
            if r not in used_rows:
                oid = obj_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        # Register brand-new detections
        for c in range(len(det_cents)):
            if c not in used_cols:
                self._register(det_cents[c], det_labels[c])

        return self._output()

    def _output(self):
        return {oid: (cx, cy, self.labels[oid])
                for oid, (cx, cy) in self.objects.items()}


# ── Prototxt (auto-download) ───────────────────────────────────────────────────
PROTOTXT   = "MobileNetSSD_deploy.prototxt"
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/"
    "master/MobileNetSSD_deploy.prototxt"
)

@st.cache_data(show_spinner="Downloading prototxt…")
def ensure_prototxt():
    if not os.path.exists(PROTOTXT):
        try:
            urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT)
        except Exception as e:
            st.error(f"❌ Could not download prototxt: {e}\n\n"
                     f"Download manually from:\n`{PROTOTXT_URL}`")
            st.stop()
    return PROTOTXT

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MobileNet-SSD…")
def load_model(proto):
    model = "mobilenet_iter_73000.caffemodel"
    if not os.path.exists(model):
        st.error(f"❌ `{model}` not found in the current folder.")
        st.stop()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def run_detection(net, frame_bgr, conf_thresh):
    """Returns list of (class_label, cx, cy, x1, y1, x2, y2, confidence)."""
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False, crop=False,
    )
    net.setInput(blob)
    dets = net.forward()

    results = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        idx = int(dets[0, 0, i, 1])
        if idx >= len(CLASSES):
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        results.append((CLASSES[idx], cx, cy, x1, y1, x2, y2, conf))
    return results


def draw_frame(frame_bgr, detections, tracked):
    """Draw bounding boxes + tracking IDs on a copy of the frame."""
    out = frame_bgr.copy()

    # Draw bounding boxes from raw detections
    for (label, cx, cy, x1, y1, x2, y2, conf) in detections:
        idx   = CLASSES.index(label)
        color = [int(c) for c in COLORS[idx]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Confidence badge
        badge = f"{label} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, badge, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Draw tracking IDs at centroids
    for oid, (cx, cy, label) in tracked.items():
        cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(out, f"ID {oid}", (cx+8, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    return out


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Min confidence (%)", 1, 100, 35) / 100.0
    max_disap   = st.slider("Tracker memory (frames)", 5, 60, 30)
    cam_idx     = st.number_input("Camera index", 0, 10, 0, step=1)
    st.divider()
    st.markdown(
        "**Model:** MobileNet-SSD (VOC)\n\n"
        "**Tracker:** Centroid tracker\n\n"
        "**ID colours:** yellow dot = centroid\n\n"
        "**Box colours:** per VOC class"
    )
    st.divider()
    st.markdown("**Detectable objects:**")
    st.markdown(", ".join(f"`{c}`" for c in CLASSES[1:]))

# ── Load assets ────────────────────────────────────────────────────────────────
proto   = ensure_prototxt()
net     = load_model(proto)
tracker = CentroidTracker(max_disappeared=30)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_vid, col_info = st.columns([3, 2])

with col_vid:
    run              = st.checkbox("▶️ Start webcam", value=False)
    frame_slot       = st.empty()

with col_info:
    st.subheader("📦 Tracked Objects")
    info_slot = st.empty()

# ── Main loop ──────────────────────────────────────────────────────────────────
if run:
    # Re-create tracker with current sidebar setting each run
    tracker = CentroidTracker(max_disappeared=max_disap)
    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("❌ Cannot open camera. Try a different camera index.")
        st.stop()

    try:
        while run:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # 1. Detect
            dets = run_detection(net, frame, conf_thresh)

            # 2. Track  (feed centroids + labels)
            tracker_input = [(label, cx, cy) for (label,cx,cy,*_) in dets]
            tracked = tracker.update(tracker_input)

            # 3. Draw
            annotated = draw_frame(frame, dets, tracked)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            # 4. Info panel
            if tracked:
                rows = ["| ID | Label | Centroid |", "|-----|-------|----------|"]
                for oid, (cx, cy, label) in sorted(tracked.items()):
                    rows.append(f"| **{oid}** | `{label}` | ({cx}, {cy}) |")
                info_slot.markdown("\n".join(rows))
            else:
                info_slot.info("No objects tracked yet.\n\n"
                               "Try: lower confidence, better lighting, "
                               "hold a bottle/chair in frame.")
    finally:
        cap.release()

else:
    frame_slot.info("☝️ Check **Start webcam** to begin.")
    info_slot.empty()