# ──────────────────────────────────────────────────────────────────────────────
#  DiaVision — DR Detection Flask Backend
#  K Group · EfficientNet-B4 + Grad-CAM (model.py)
# ──────────────────────────────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import base64, io, cv2, traceback, os

from model import load_model, predict, NotFundusImageError

app = Flask(__name__)

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE      = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']        = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']   = MAX_FILE_SIZE

# ── Load model ONCE at startup ──
print("=" * 55)
print("  DiaVision — Loading DR model, please wait...")
print("=" * 55)
try:
    model = load_model()
    print("Model ready!\n")
except Exception as e:
    print(f"Model loading failed: {e}")
    traceback.print_exc()
    model = None


def numpy_to_base64(img_rgb):
    """Convert numpy RGB image to base64 PNG string."""
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── PAGES ──
@app.route('/')
def index():
    return render_template('dr_detection_gui.html')

@app.route('/info')
def info():
    return render_template('eye_info.html')


# ── PREDICT ENDPOINT ──
@app.route("/predict", methods=["POST"])
def predict_route():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded. Server error."}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File type not allowed. Use PNG, JPG, or JPEG."}), 400

    try:
        file_bytes = file.read()
        pil_img    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array  = np.array(pil_img)

        # predict() will raise NotFundusImageError if it's not a retinal image
        img_rgb, cam, prediction, prob = predict(model, img_array, threshold=0.35)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)

        original_b64 = numpy_to_base64(img_rgb)
        heatmap_b64  = numpy_to_base64(heatmap)
        overlay_b64  = numpy_to_base64(overlay)

        # Borderline logic
        if prediction == "DR":
            status = "DR"
        elif prob >= 0.20:
            status = "Borderline"
        else:
            status = "No DR"

        return jsonify({
            "label":      status,
            "confidence": prob,
            "original":   original_b64,
            "heatmap":    heatmap_b64,
            "overlay":    overlay_b64,
        })

    except NotFundusImageError as e:
        # Return a clean user-facing error — not a 500
        return jsonify({
            "success": False,
            "error":   f"Invalid image: {str(e)} Please upload a valid fundus camera retinal photograph."
        }), 422

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


# ── HEALTH CHECK ──
@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status, "model": "EfficientNet-B4 DR"}), 200


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))
    print(f"Server running at: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
