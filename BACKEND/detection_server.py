from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, threading, queue, time, psycopg2, requests
from datetime import datetime
from merger import run_merged_detector  # YOLO + InsightFace
import numpy as np

# -----------------------------
# QUEUES
# -----------------------------
input_queue = queue.Queue(maxsize=2)
output_queue = queue.Queue(maxsize=2)
alert_queue = queue.Queue()
write_queue = queue.Queue()

# -----------------------------
# CAMERA CAPTURE THREAD
# -----------------------------
def camera_capture(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            input_queue.put(frame, timeout=0.5)
        except queue.Full:
            continue

threading.Thread(target=camera_capture, daemon=True).start()

# -----------------------------
# DATABASE SAVE FUNCTION
# -----------------------------
def save_detection_to_db(event_data):
    try:
        conn = psycopg2.connect(
            dbname="SurveillanceDB",
            user="postgres",
            password="Aryan@211",
            host="localhost"
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO DetectionEvents (timestamp, screenshot, camera_id, person_id, weapon_id, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            event_data['timestamp'],
            psycopg2.Binary(event_data['screenshot']),
            event_data['camera_id'],
            event_data.get('person_id'),
            event_data.get('weapon_id'),
            event_data.get('notes', '')
        ))
        conn.commit()
        cur.close()
        conn.close()
        print(f"DB_SAVED: {event_data['notes']}")
    except Exception as e:
        print(f"DB_ERROR: {str(e)}")

# -----------------------------
# HELPER: FORMAT DETECTION TO HUMAN-READABLE
# -----------------------------
def format_detection(detection: dict) -> list[str]:
    messages = []

    # FACE MATCH
    if detection.get("face_embedding") is not None:
        try:
            response = requests.post(
                "http://localhost:8001/match_face",
                json={"embedding": detection["face_embedding"]},
                timeout=3
            )
            if response.status_code == 200:
                match = response.json()
                detection["person_id"] = match.get("person_id")
                similarity = float(match.get("similarity", 0))
                msg = f"Face matched: {match.get('name')} (Similarity: {similarity:.2f})"
            else:
                msg = "No face match found"
        except Exception as e:
            msg = f"FACE_MATCH_ERROR: {str(e)}"
        messages.append(msg)

    # WEAPON DETECTION
    if detection.get("weapon_id"):
        confidence = float(detection.get("confidence", 0))
        msg = f"Weapon detected: {detection['weapon_id']} (Confidence: {confidence:.2f})"
        messages.append(msg)

    # LOITERING / RUNNING
    if detection.get("activity"):
        speed = float(detection.get("speed", 0))
        track_id = detection.get("track_id", detection.get("id", ""))
        msg = f"Activity detected: {detection['activity']} | Track ID: {track_id} | Speed: {speed:.2f}"
        messages.append(msg)

    return messages if messages else ["Detection event occurred"]

# -----------------------------
# DETECTION WORKER THREAD
# -----------------------------
def detection_worker():
    for detection in run_merged_detector(input_queue, output_queue, alert_queue, write_queue):
        # Format messages human-readable
        messages = format_detection(detection)

        # Save to DB
        save_detection_to_db({
            'timestamp': datetime.now(),
            'screenshot': detection['screenshot'],
            'camera_id': detection.get('camera_id', 1),
            'person_id': detection.get('person_id'),
            'weapon_id': detection.get('weapon_id'),
            'notes': " | ".join(messages)
        })

        # Send each message to alert queue
        for msg in messages:
            alert_queue.put(msg)

threading.Thread(target=detection_worker, daemon=True).start()

# -----------------------------
# ALERT STREAM (SSE)
# -----------------------------
def alert_stream():
    while True:
        try:
            msg = alert_queue.get(timeout=1)
            yield f"data: {msg}\n\n"
        except queue.Empty:
            continue

# -----------------------------
# MJPEG STREAM
# -----------------------------
def gen_frames():
    while True:
        try:
            frame = output_queue.get(timeout=1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/video_feed")
def video_feed(cameraId: str = "cam-1"):
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/alerts/stream")
def stream_alerts():
    """Server-Sent Events endpoint for live detection alerts"""
    return StreamingResponse(alert_stream(), media_type="text/event-stream")

@app.get("/")
def root():
    return {"status": "Detection service running"}



