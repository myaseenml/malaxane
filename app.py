from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import os
from PIL import Image
from datetime import datetime
from process_images import hide_plate, remove_car_bg, place_car_on_background

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/app")
def app_ai():
    return render_template("app.html")

def create_generated_car_name(name, index=None):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    if index is not None:
        return f"{name}_{index}_{date_str}_{time_str}.png"
    return f"{name}_{date_str}_{time_str}.png"

@app.route("/process", methods=["POST"])
def process_images():
    if "carImages" not in request.files or "bgImage" not in request.files:
        return jsonify({"error": "Missing files"}), 400

    car_images = request.files.getlist("carImages")  # Get multiple files
    bg_img = request.files["bgImage"]

    if len(car_images) == 0:
        return jsonify({"error": "No car images provided"}), 400

    bg_path = os.path.join(UPLOAD_FOLDER, bg_img.filename)
    bg_img.save(bg_path)

    processed_images = []

    for index, car_img in enumerate(car_images):
        car_path = os.path.join(UPLOAD_FOLDER, car_img.filename)
        car_img.save(car_path)

        # Hide plate and remove background
        image_PIL = Image.open(car_path)
        hide_success, car_wo_plate_path = hide_plate(image_PIL)
        if not hide_success:
            car_wo_plate_path = car_path  # Use original image if no plate detected

        remove_car_bg(car_wo_plate_path)

        output_filename = create_generated_car_name("generated_car", index)
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        place_car_on_background(bg_path, output_path)

        processed_images.append({"image": f"/get_image/{output_filename}"})

    return jsonify({"processed_images": processed_images})

@app.route("/get_image/<filename>")
def get_image(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
