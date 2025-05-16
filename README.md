# 🧠 AI Person Detection on Screen using YOLOv8 + OpenCV

This project captures your **entire screen in real time** and uses **YOLOv8 (via Ultralytics)** to detect people. Bounding boxes are drawn around detected persons, and the frame is displayed with FPS information.

---

## 🚀 Features

- ✅ Real-time **screen capture** using `mss`
- ✅ **YOLOv8** model for person detection
- ✅ FPS display
- ✅ Bounding boxes + confidence % labels
- ✅ Lightweight (supports yolov8n/yolov8s)

---

## 🧰 Requirements

Install dependencies using `pip`:

```bash
pip install opencv-python mss pyautogui numpy ultralytics
```

---

## 🛠️ How It Works

1. Captures your primary screen in real-time.
2. Passes each frame to a YOLOv8 model.
3. Detects only **persons (class 0)**.
4. Draws bounding boxes + confidence % on screen.
5. Displays FPS and live preview.
6. Press **`q`** to quit.

---

## 📁 File Structure

```bash
ai_detection.py   # Main script
yolov8n.pt        # YOLOv8 weights (downloaded separately)
```

---

## 📦 YOLOv8 Model

Make sure you have the YOLOv8 model (e.g., `yolov8n.pt`) in the same directory or provide the correct path.

To download it:

```bash
from ultralytics import YOLO
YOLO('yolov8n.pt')  # Downloads and caches the model
```

You can switch to other versions like:
- `yolov8s.pt` (small)
- `yolov8m.pt` (medium)

---

## ⚠️ Notes

- This script **scans the full screen**. Performance may vary based on screen resolution and model size.
- Requires a relatively modern GPU or CPU for smooth performance.
