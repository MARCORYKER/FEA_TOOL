from flask import Flask, request, send_file
import os
import time
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from reportlab.pdfgen import canvas

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# HTML Page
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Analysis</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { text-align: center; padding: 40px; background: #f4f4f4; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        .fade-in { animation: fadeIn 1.5s; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container fade-in">
        <h2 class="mb-4">Upload Images for Analysis</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="images" multiple required class="form-control mb-3">
            <button type="submit" class="btn btn-primary">Upload & Analyze</button>
        </form>
    </div>
</body>
</html>
'''

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/upload", methods=["POST"])
def upload():
    images = request.files.getlist("images")  
    image_paths = []
    for image in images:
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)
        image_paths.append(image_path)
    start_time = time.time()
    results = analyze_images(image_paths)  
    pdf_path = os.path.join(OUTPUT_FOLDER, "final_report.pdf")
    generate_pdf(results, pdf_path)  
    total_time = time.time() - start_time
    return f'''<h2>Processing Completed!</h2><p>Total Time Taken: {total_time:.2f} seconds</p><a href="/download" class="btn btn-success">Download Report</a>'''

@app.route("/download")
def download():
    return send_file("output/final_report.pdf", as_attachment=True)

def analyze_images(image_paths):
    results = []
    for image_path in image_paths:
        start_time = cv2.getTickCount()
        image = cv2.imread(image_path)
        image = cv2.resize(image, (500, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        mean_intensity = np.mean(gray)
        total_edges = np.count_nonzero(edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_disp, max_disp = float("inf"), 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            displacement = np.sqrt(w**2 + h**2)
            min_disp = min(min_disp, displacement)
            max_disp = max(max_disp, displacement)
        if min_disp == float("inf"): min_disp = 0
        min_stress = min_disp * 0.8
        max_stress = max_disp * 1.2
        stress_distribution = (min_stress + max_stress) / 2
        execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        results.append({
            "filename": os.path.basename(image_path),
            "mean_intensity": round(mean_intensity, 2),
            "total_edges": total_edges,
            "min_displacement": round(min_disp, 2),
            "max_displacement": round(max_disp, 2),
            "min_stress": round(min_stress, 2),
            "max_stress": round(max_stress, 2),
            "stress_distribution": round(stress_distribution, 2),
            "processing_time": round(execution_time, 2)
        })
    return results

def generate_pdf(results, pdf_path):
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 12)
    y_position = 750
    for result in results:
        c.drawString(50, y_position, f"File: {result['filename']}")
        c.drawString(50, y_position - 20, f"Mean Intensity: {result['mean_intensity']}")
        c.drawString(50, y_position - 40, f"Total Edges Detected: {result['total_edges']}")
        c.drawString(50, y_position - 60, f"Min Displacement: {result['min_displacement']}")
        c.drawString(50, y_position - 80, f"Max Displacement: {result['max_displacement']}")
        c.drawString(50, y_position - 100, f"Min Stress: {result['min_stress']}")
        c.drawString(50, y_position - 120, f"Max Stress: {result['max_stress']}")
        c.drawString(50, y_position - 140, f"Stress Distribution: {result['stress_distribution']}")
        c.drawString(50, y_position - 160, f"Processing Time: {result['processing_time']} sec")
        y_position -= 200
    c.save()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

