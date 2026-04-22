# 🚗 Vehicle Tracking, Counting & Number Plate Detection

> **Powered by FastAPI + YOLO11 + ByteTracker + Supervision + EasyOCR**

### 🟢 Live Demo: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/tejesh05/vehicle-tracker)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 **Web Interface** | User-friendly web UI to upload videos and view results |
| 🔢 **Per-class counting** | Separate IN/OUT counts for Car, Motorcycle, Bus, Truck |
| 🎯 **Interactive Line Drawing** | Draw your counting line directly on the video preview in the browser |
| 🔤 **Number plate OCR** | Automatic license plate detection + text reading (EasyOCR) |
| 🧠 **State-of-the-art Tracking** | Powered by YOLO11 and ByteTracker |

---

## 📦 Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR will download its language models (~100 MB) on first run automatically.  
> The vehicle detection model (`yolo11s.pt`) and license plate model will be downloaded automatically on first run.

---

## 🚀 Usage

### Start the Web Server

Run the FastAPI application using Python:

```bash
python app.py
```

Alternatively, you can run it with uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Web Interface

Open your web browser and go to:
[http://localhost:8000](http://localhost:8000)

1. **Upload a video**: Select and upload your target video.
2. **Draw Counting Line**: Click and drag on the video preview to draw the exact line for counting vehicles (IN/OUT).
3. **Process**: Click the process button and wait for the AI to track vehicles, count them, and read license plates.
4. **View Results**: The processed video with annotations and counts will be displayed in the browser.

---

## 📁 Project Structure

```
├── app.py                     ← FastAPI backend application
├── requirements.txt           ← Python dependencies
├── static/
│   └── index.html             ← Frontend web interface
├── uploads/                   ← Directory for uploaded user videos
└── output/                    ← Processed videos and generated preview frames
```

---

## 📝 Notes

- **Models**: The system defaults to using `yolo11s.pt` for vehicle detection. You can modify `app.py` to use other models (e.g. `yolo11n.pt` for speed, `yolo11x.pt` for maximum accuracy).
- **GPU Acceleration**: For better performance, particularly with YOLO and EasyOCR, running on a system with a CUDA-enabled GPU is recommended.
- **License Plate OCR**: License plate reading runs every 5 frames to balance performance.

---

## 🛠️ Troubleshooting

**EasyOCR download fails on first run?**  
→ Check your internet connection. EasyOCR downloads ~100 MB of language model files.

**Cannot access web interface?**  
→ Ensure the terminal shows the server is running on `http://0.0.0.0:8000` and no firewall is blocking the port.
