import os
import cv2
import uuid
import numpy as np
import imageio
import supervision as sv
from ultralytics import YOLO
import easyocr
import asyncio

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Make directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("output", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Globals for models so they load once
print("[INFO] Initializing Models...")
vehicle_model = YOLO("yolo11s.pt")
plate_model = YOLO("models/license_plate_detector.pt") if os.path.exists("models/license_plate_detector.pt") else None
reader = easyocr.Reader(['en'], gpu=True)
print("[INFO] Models Ready.")

TARGET_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

class ProcessRequest(BaseModel):
    filename: str
    line: list

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    safe_name = f"{uid}{ext}"
    in_path = os.path.join("uploads", safe_name)
    
    with open(in_path, "wb") as f:
        f.write(await file.read())
        
    # Extract first frame
    cap = cv2.VideoCapture(in_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return JSONResponse(status_code=400, content={"error": "Invalid video file. Could not read frames."})
        
    preview_name = f"preview_{uid}.jpg"
    preview_path = os.path.join("output", preview_name)
    cv2.imwrite(preview_path, frame)
    
    return {
        "filename": safe_name,
        "preview_url": f"/output/{preview_name}"
    }

@app.post("/api/process")
async def process_video(req: ProcessRequest):
    # Offload processing to a thread so we don't block the fastAPI loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_processing, req)
    return result

def run_processing(req: ProcessRequest):
    in_path = os.path.join("uploads", req.filename)
    if not os.path.exists(in_path):
        return {"error": "Uploaded file not found"}
        
    out_name = f"tracked_{os.path.splitext(req.filename)[0]}.mp4"
    out_path = os.path.join("output", out_name)
    
    # Setup coordinates
    start_pt = sv.Point(*req.line[0])
    end_pt = sv.Point(*req.line[1])
    
    video_info = sv.VideoInfo.from_video_path(in_path)
    generator = sv.get_video_frames_generator(in_path)
    
    line_zone = sv.LineZone(start=start_pt, end=end_pt)
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25, 
        lost_track_buffer=90, 
        minimum_matching_threshold=0.8, 
        frame_rate=video_info.fps
    )
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=1.0)
    
    plate_cache = {}
    
    # Use imageio with libx264 codec for web compatibility!
    writer = imageio.get_writer(
        out_path, 
        fps=video_info.fps, 
        codec='libx264',
        quality=7  # Decent quality vs size tradeoff
    )
    
    try:
        for frame_idx, frame in enumerate(generator):
            # Model inference
            results = vehicle_model(frame, verbose=False, conf=0.15, iou=0.6, imgsz=1024)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter classes
            mask = np.array([class_id in TARGET_CLASSES for class_id in detections.class_id], dtype=bool)
            detections = detections[mask]
            
            # Tracking
            detections = byte_tracker.update_with_detections(detections)
            line_zone.trigger(detections=detections)
            
            labels = []
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                tracker_id = detections.tracker_id[i]
                bbox = detections.xyxy[i]
                class_name = TARGET_CLASSES[class_id]
                
                # License plate scanning every 5 frames
                if plate_model and tracker_id not in plate_cache:
                    if frame_idx % 5 == 0:
                        x1, y1, x2, y2 = map(int, bbox)
                        vehicle_crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if vehicle_crop.size > 0:
                            plate_results = plate_model(vehicle_crop, verbose=False, conf=0.15)[0]
                            p_dets = sv.Detections.from_ultralytics(plate_results)
                            if len(p_dets) > 0:
                                best_idx = np.argmax(p_dets.confidence)
                                px1, py1, px2, py2 = map(int, p_dets.xyxy[best_idx])
                                plate_crop = vehicle_crop[max(0, py1):py2, max(0, px1):px2]
                                if plate_crop.size > 0:
                                    # Convert to grayscale for better OCR
                                    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                    # OCR
                                    ocr_result = reader.readtext(gray_plate, detail=0)
                                    if ocr_result:
                                        t = "".join(ocr_result).replace(" ", "").strip()
                                        if len(t) >= 4:
                                            plate_cache[tracker_id] = t
                
                found_plate = plate_cache.get(tracker_id, "")
                if found_plate:
                    labels.append(f"#{tracker_id} {class_name} [{found_plate}]")
                else:
                    labels.append(f"#{tracker_id} {class_name}")

            # Drawing annotations
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frame = line_zone_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
            
            # Draw HUD
            bg_rect_end = (annotated_frame.shape[1] // 3, 140)
            cv2.rectangle(annotated_frame, (10, 10), bg_rect_end, (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"IN Count:  {line_zone.in_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 3)
            cv2.putText(annotated_frame, f"OUT Count: {line_zone.out_count}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 3)
            
            # Convert BGR (OpenCV default) to RGB (imageio default)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        return {"error": str(e)}
    finally:
        writer.close()
        
    return {
        "in_count": line_zone.in_count,
        "out_count": line_zone.out_count,
        "output_video_url": f"/output/{out_name}"
    }

if __name__ == "__main__":
    import uvicorn
    # Optional auto-start script if run directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
