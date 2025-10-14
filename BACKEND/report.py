import psycopg2
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io

def export_events_to_pdf(output_file="DetectionReport.pdf"):
    conn = psycopg2.connect(
        dbname="SurveillanceDB",
        user="postgres",
        password="Aryan@211",
        host="localhost"
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT event_id, timestamp, person_id, camera_id, notes, screenshot 
        FROM DetectionEvents
        ORDER BY timestamp DESC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Setup PDF
    doc = SimpleDocTemplate(output_file, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph("Detection Events Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 20))

    for row in rows:
        event_id, timestamp, person_id, camera_id, notes, screenshot_bytes = row

        # Metadata table
        data = [
            ["Event ID", event_id],
            ["Timestamp", str(timestamp)],
            ["Person ID", str(person_id) if person_id else "N/A"],
            ["Camera ID", str(camera_id) if camera_id else "N/A"],
            ["Notes", notes if notes else ""]
        ]
        details_table = Table(data, colWidths=[100, 250])
        details_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        # Convert screenshot bytes to image
        if screenshot_bytes:
            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to RGB for reportlab
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode(".png", img_rgb)
            img_stream = io.BytesIO(img_encoded.tobytes())

            img_obj = Image(img_stream, width=200, height=150)

            # Combine image (left) and details (right) in one row
            combined_table = Table(
                [[img_obj, details_table]],
                colWidths=[220, 300]  # Adjust widths as needed
            )
            combined_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ]))

            elements.append(combined_table)
            elements.append(Spacer(1, 20))

    doc.build(elements)
    print(f"✅ PDF report generated: {output_file}")


# Example usage
export_events_to_pdf("DetectionReport.pdf")
