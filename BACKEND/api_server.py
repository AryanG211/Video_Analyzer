# api_service/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psycopg2, base64
from psycopg2.extras import RealDictCursor
from fastapi.responses import FileResponse
from report import export_events_to_pdf  

# -----------------------------
# FASTAPI APP CONFIG
# -----------------------------
app = FastAPI(title="Database API Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_db_conn():
    return psycopg2.connect(
        dbname="SurveillanceDB",
        user="postgres",
        password="Aryan@211",
        host="localhost"
    )

# -----------------------------
# /api/people
# -----------------------------

@app.get("/api/reports")
def download_report():
    output_file = "DetectionReport.pdf"
    export_events_to_pdf(output_file)  # generate PDF
    return FileResponse(output_file, media_type="application/pdf", filename="events-report.pdf")


@app.get("/api/people")
def get_detected_people():
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT event_id, screenshot, timestamp, camera_id
            FROM DetectionEvents
            WHERE screenshot IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        people = []
        for row in rows:
            img_base64 = base64.b64encode(row['screenshot']).decode('utf-8')
            people.append({
                "id": str(row['event_id']),
                "photoUrl": f"data:image/jpeg;base64,{img_base64}",
                "timestamp": row['timestamp'].isoformat(),
                "cameraId": str(row['camera_id'])
            })
        cur.close()
        conn.close()
        return JSONResponse(people)
    except Exception as e:
        print("Error fetching people:", e)
        return JSONResponse([])

# -----------------------------
# /api/events
# -----------------------------
@app.get("/api/events")
def get_detection_events():
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT event_id, timestamp, camera_id, person_id, weapon_id, notes
            FROM DetectionEvents
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        events = []
        for row in rows:
            events.append({
                "id": str(row['event_id']),
                "date": row['timestamp'].date().isoformat(),
                "time": row['timestamp'].time().strftime("%H:%M:%S"),
                "location": f"Camera {row['camera_id']}" if row['camera_id'] else "—",
                "summary": row['notes'] or "—",
                "personId": str(row['person_id']) if row['person_id'] else None,
                "weaponId": str(row['weapon_id']) if row['weapon_id'] else None
            })
        cur.close()
        conn.close()
        return JSONResponse(events)
    except Exception as e:
        print("Error fetching events:", e)
        return JSONResponse([])

@app.get("/")
def root():
    return {"status": "Database API service running"}

# Run using: uvicorn api_service.main:app --host 0.0.0.0 --port 8001
