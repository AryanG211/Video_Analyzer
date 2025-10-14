import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import math
import time
import os
import queue
import torch
import threading
import faiss
import psycopg2
import insightface

# ---------------- Config ----------------
CAM_INDEX = 0
POSE_MODEL_PATH = "yolov8n-pose.pt"
WEAPON_MODEL_PATH = r"C:\Users\hp\OneDrive\Desktop\2ND_PS\BACKEND\best (3).pt"
WEAPON_DISPLAY_FRAMES = 30  # Keep weapon boxes visible for N frames
CAP_QUEUE_MAX = 2
PROCESS_IMG_SIZE_GPU = 640
PROCESS_IMG_SIZE_CPU = 320

# Motion / heuristic params
RUNNING_SPEED_PIX_PER_S = 350
LOITER_FRAMES_THRESHOLD = 90
LOITER_MOVEMENT_TOLERANCE = 20
ATTACK_FORWARD_LEAN_DEG = 20
ATTACK_ARM_EXTENDED_DIST = 100
KEYPOINT_CONF_THR = 0.4
TIME_WINDOW = 30

OUTPUT_DIR = "alerts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- device / model ----------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE = PROCESS_IMG_SIZE_GPU if device.startswith("cuda") else PROCESS_IMG_SIZE_CPU
print(f"[INFO] Device: {device}  IMG_SIZE: {IMG_SIZE}")

pose_model = YOLO(POSE_MODEL_PATH)
weapon_model = YOLO(WEAPON_MODEL_PATH)

pose_model.to(device)
weapon_model.to(device)

# Warmup models
print("[INFO] Warming up models...")
dummy_frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
pose_model(dummy_frame, verbose=False, device=device, imgsz=IMG_SIZE)
weapon_model(dummy_frame, verbose=False, device=device, imgsz=IMG_SIZE)
print("[INFO] Models ready!")

# ---------------- supervision setup ----------------
byte_tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ---------------- shared states ----------------
history = {}
frames_stationary = {}
frame_idx = 0

# ---------------- utility math ----------------
def vec(a, b): return (b[0]-a[0], b[1]-a[1])
def length(v): return math.hypot(v[0], v[1])
def angle_between(p1, p2):
    dot = p1[0]*p2[0] + p2[1]*p1[1]
    l1, l2 = length(p1), length(p2)
    if l1*l2 == 0: return 0
    cos = max(-1.0, min(1.0, dot/(l1*l2)))
    return math.degrees(math.acos(cos))
def torso_angle(shoulder_mid, hip_mid):
    return angle_between(vec(shoulder_mid, hip_mid), (0,1))
def is_running(track_centers, fps):
    if len(track_centers) < 2: return False, 0.0
    N = min(len(track_centers)-1, TIME_WINDOW-1)
    dx = sum(length(vec(track_centers[i], track_centers[i+1])) for i in range(-N-1, -1))
    avg_speed = (dx / max(N,1)) * fps
    return (avg_speed > RUNNING_SPEED_PIX_PER_S), avg_speed
def is_loitering(track_centers, stationary_count):
    return stationary_count >= LOITER_FRAMES_THRESHOLD
def detect_attack_by_pose(kps):
    try:
        k = np.array(kps)
    except:
        return False, []
    info = []
    if np.sum(k[:,2] > KEYPOINT_CONF_THR) < 6:
        return False, info
    shoulder_mid = ((k[5,0]+k[6,0])/2, (k[5,1]+k[6,1])/2)
    hip_mid = ((k[11,0]+k[12,0])/2, (k[11,1]+k[12,1])/2)
    lean_deg = torso_angle(shoulder_mid, hip_mid)
    if abs(lean_deg) > ATTACK_FORWARD_LEAN_DEG:
        info.append("torso_lean")
    for side in [("left",5,7,9), ("right",6,8,10)]:
        _, s_i, e_i, w_i = side
        if k[s_i,2]>KEYPOINT_CONF_THR and k[w_i,2]>KEYPOINT_CONF_THR:
            dist = length(vec((k[s_i,0],k[s_i,1]),(k[w_i,0],k[w_i,1])))
            if dist > ATTACK_ARM_EXTENDED_DIST:
                info.append(f"{side[0]}_arm_extended")
    if ("torso_lean" in info) and any("arm_extended" in x for x in info):
        return True, info
    return False, info

# ---------------- heatmap ----------------
heatmap = np.zeros((480, 640), dtype=np.float32)
def add_heatmap(center, radius=40, intensity=1.0):
    x, y = int(center[0]), int(center[1])
    if 0 <= x < 640 and 0 <= y < 480:
        cv2.circle(heatmap, (x, y), radius, intensity, -1)
def overlay_heatmap(frame, heatmap, alpha=0.35):
    hm = cv2.GaussianBlur(np.clip(heatmap,0,255),(0,0),sigmaX=25,sigmaY=25)
    hm_norm = cv2.normalize(hm,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame,1-alpha,hm_color,alpha,0)

# ---------------- writer thread ----------------
def writer_thread_fn(write_queue):
    writer = None
    while True:
        item = write_queue.get()
        if item is None: break
        cmd = item.get("action") if isinstance(item, dict) else item[0]

        if cmd == "START":
            filename = f"{OUTPUT_DIR}/alert_{int(time.time())}.mp4"
            size = item.get("size",(640,480))
            fps = item.get("fps",30)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            if writer is not None: writer.release()
            writer = cv2.VideoWriter(filename, fourcc, fps, size)
            print(f"[REC] Started recording: {filename}")

        elif cmd == "FRAME" and writer is not None:
            frame = item["frame"]
            writer.write(frame)

        elif cmd == "STOP" and writer is not None:
            writer.release()
            writer = None
            print("[REC] Stopped recording")

# ---------------- DB helpers ----------------
def get_embeddings_from_db():
    try:
        conn = psycopg2.connect(dbname="SurveillanceDB", user="postgres", password="Aryan@211", host="localhost")
        cur = conn.cursor()
        cur.execute("SELECT person_id, name, face_embedding FROM Watchlist;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        person_ids, names, embeddings = [], [], []
        for row in rows:
            pid, name, emb_bytes = row
            emb = np.frombuffer(emb_bytes, dtype="float32")
            person_ids.append(pid)
            names.append(name)
            embeddings.append(emb)
        return np.array(embeddings), names, person_ids
    except Exception as e:
        print(f"[DB] Failed to get embeddings: {e}")
        return np.array([]), [], []

def insert_detection_event(screenshot_bytes, camera_id=1, person_id=None, weapon_id=None, notes=""):
    try:
        conn = psycopg2.connect(dbname="SurveillanceDB", user="postgres", password="Aryan@211", host="localhost")
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO DetectionEvents (screenshot, camera_id, person_id, weapon_id, notes)
            VALUES (%s, %s, %s, %s, %s)
        """, (screenshot_bytes, camera_id, person_id, weapon_id, notes))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Event saved: {notes}")
    except Exception as e:
        print(f"[DB] Failed to insert event: {e}")

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# ---------------- MERGED DETECTOR ----------------
def run_merged_detector(input_queue, final_output_queue, alert_queue, write_queue):
    global frame_idx, history, frames_stationary

    frame_idx = 0
    recording = False
    RECORD_GRACE_PERIOD = 3.0
    last_alert_time = 0
    FACE_ALERT_COOLDOWN = 7.0
    WEAPON_ALERT_COOLDOWN = 7.0
    last_face_alert_time = {}
    last_weapon_alert_time = 0

    writer_thread = threading.Thread(target=writer_thread_fn, args=(write_queue,), daemon=True)
    writer_thread.start()

    # ---------------- Face model ----------------
    try:
        face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        print("[INFO] Using GPU for face detection")
    except:
        face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        print("[INFO] Using CPU for face detection")
    face_model.prepare(ctx_id=0, det_size=(640,640))

    # ---------------- Load embeddings & FAISS ----------------
    embeddings, names, person_ids = get_embeddings_from_db()
    if embeddings.shape[0] == 0:
        raise RuntimeError("No embeddings in DB!")
    faiss_index = build_faiss_index(embeddings)

    print("[MERGED] Detector thread started.")

    while True:
        try:
            frame = input_queue.get(timeout=0.5)
            if frame is None:
                break
            frame_idx += 1
            annotated = frame.copy()
            alert_triggered = False
            h_frame, w_frame = annotated.shape[:2]

            # ---------------- Pose Detection ----------------
            results = pose_model(frame, verbose=False, device=device, imgsz=IMG_SIZE)[0]
            dets = sv.Detections.from_ultralytics(results)
            dets = dets[dets.class_id == 0]
            tracked = byte_tracker.update_with_detections(dets)
            kps_array = getattr(results.keypoints, 'data', None)
            if kps_array is not None:
                kps_array = kps_array.cpu().numpy()

            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i]) + 1
                bbox = tracked.xyxy[i]
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

                hist = history.setdefault(track_id, [])
                hist.append((center, frame_idx))
                hist = [h for h in hist if frame_idx - h[1] <= TIME_WINDOW]
                history[track_id] = hist

                if len(hist) >= 2:
                    motion = length(vec(hist[-2][0], hist[-1][0]))
                    frames_stationary[track_id] = frames_stationary.get(track_id, 0) + 1 if motion < LOITER_MOVEMENT_TOLERANCE else 0

                action_labels = []
                centers_only = [c for c, _ in hist]

                # Running
                is_run, speed = is_running(centers_only, fps=30)
                if is_run:
                    action_labels.append(f"RUNNING ({int(speed)} px/s)")
                    alert_queue.put({"type": "RUNNING", "track_id": track_id, "speed": float(speed)})
                    ret, buf = cv2.imencode('.jpg', annotated)
                    if ret:
                        insert_detection_event(buf.tobytes(), camera_id=1, notes=f"RUNNING ID{track_id} speed {speed:.1f}px/s")
                    alert_triggered = True

                # Loitering
                if is_loitering(centers_only, frames_stationary.get(track_id, 0)):
                    action_labels.append("LOITERING")
                    add_heatmap(center, radius=40, intensity=2.0)
                    alert_queue.put({"type": "LOITERING", "track_id": track_id})
                    ret, buf = cv2.imencode('.jpg', annotated)
                    if ret:
                        insert_detection_event(buf.tobytes(), camera_id=1, notes=f"LOITERING ID{track_id}")
                    alert_triggered = True

                # Pose-based attack
                if kps_array is not None:
                    best_idx, best_dist = None, 1e9
                    for idx, person_kps in enumerate(kps_array):
                        valid = person_kps[:, 2] > KEYPOINT_CONF_THR
                        if not np.any(valid):
                            continue
                        px, py = int(person_kps[valid, 0].mean()), int(person_kps[valid, 1].mean())
                        d = math.hypot(px - center[0], py - center[1])
                        if d < best_dist:
                            best_dist, best_idx = d, idx
                    if best_idx is not None:
                        attack_flag, attack_info = detect_attack_by_pose(kps_array[best_idx])
                        if attack_flag:
                            action_labels.append("POSSIBLE_ATTACK")
                            alert_queue.put({"type": "POSSIBLE_ATTACK", "track_id": track_id})
                            ret, buf = cv2.imencode('.jpg', annotated)
                            if ret:
                                insert_detection_event(buf.tobytes(), camera_id=1, notes=f"POSSIBLE_ATTACK ID{track_id} details: {', '.join(attack_info)}")
                            alert_triggered = True

                # Annotate pose
                display_label = f"ID {track_id}" + (" | " + " | ".join(action_labels) if action_labels else "")
                annotated = box_annotator.annotate(
                    annotated,
                    sv.Detections(xyxy=np.array([bbox]), confidence=np.array([1.0]), class_id=np.array([0]))
                )
                annotated = label_annotator.annotate(
                    annotated,
                    sv.Detections(xyxy=np.array([bbox]), confidence=np.array([1.0]), class_id=np.array([0])),
                    labels=[display_label]
                )

            # ---------------- Weapon Detection ----------------
            results_weapon = weapon_model(frame, verbose=False, device=device, imgsz=IMG_SIZE, conf=0.5, iou=0.25)[0]
            if results_weapon.boxes is not None and len(results_weapon.boxes) > 0:
                for box, conf, cls in zip(results_weapon.boxes.xyxy.cpu().numpy(),
                                          results_weapon.boxes.conf.cpu().numpy(),
                                          getattr(results_weapon.boxes, 'cls', np.zeros(len(results_weapon.boxes.xyxy)))):
                    if conf > 0.5 and time.time() - last_weapon_alert_time > WEAPON_ALERT_COOLDOWN:
                        last_weapon_alert_time = time.time()
                        alert_queue.put({"type": "WEAPON", "confidence": float(conf)})
                        ret, buf = cv2.imencode('.jpg', annotated)
                        if ret:
                            insert_detection_event(buf.tobytes(), camera_id=1, weapon_id=1, notes=f"WEAPON conf {conf:.2f} class {int(cls)}")
                        alert_triggered = True

                    # Draw weapon box in red
                    x1, y1, x2, y2 = map(int, box)
                    annotated = cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"WEAPON {conf:.2f}"
                    annotated = cv2.putText(annotated, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ---------------- Face Detection ----------------
            try:
                resized_frame = cv2.resize(annotated, (640, 640))
                faces = face_model.get(resized_frame)

                for face in faces:
                    # Scale bbox back to original frame
                    x1, y1, x2, y2 = face.bbox
                    x1 = int(x1 * w_frame / 640)
                    x2 = int(x2 * w_frame / 640)
                    y1 = int(y1 * h_frame / 640)
                    y2 = int(y2 * h_frame / 640)

                    # Compute face embedding and compare
                    embedding = face.normed_embedding.astype("float32").reshape(1, -1)
                    faiss.normalize_L2(embedding)
                    D, I = faiss_index.search(embedding, k=1)
                    similarity, idx = D[0][0], I[0][0]

                    # Only annotate known faces (in watchlist)
                    if similarity > 0.20:
                        pid, name = person_ids[idx], names[idx]
                        if pid not in last_face_alert_time or time.time() - last_face_alert_time[pid] > FACE_ALERT_COOLDOWN:
                            last_face_alert_time[pid] = time.time()
                            alert_queue.put({"type": "FACE_MATCH", "name": name, "similarity": similarity})
                            ret, buf = cv2.imencode('.jpg', annotated)
                            if ret:
                                insert_detection_event(buf.tobytes(), camera_id=1, person_id=pid,
                                                       notes=f"FACE_MATCH {name} sim {similarity:.2f}")
                            alert_triggered = True

                        # Draw green box and label only for known faces
                        label = f"{name} ({similarity:.2f})"
                        annotated = cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        annotated = cv2.putText(annotated, label, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Unknown faces → ignored (no annotation, no label)

            except Exception as e:
                print(f"[FACE] Error: {e}")

            # ---------------- Heatmap ----------------
            annotated = overlay_heatmap(annotated, heatmap, alpha=0.35)

            # ---------------- Recording ----------------
            if alert_triggered:
                last_alert_time = time.time()
                if not recording:
                    recording = True
                    write_queue.put({"action": "START", "size": (640, 480)})
            elif recording and time.time() - last_alert_time > RECORD_GRACE_PERIOD:
                recording = False
                write_queue.put({"action": "STOP"})

            # ---------------- Output ----------------
            final_output_queue.put(annotated)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[MERGED] Detector loop error: {e}")
            continue

