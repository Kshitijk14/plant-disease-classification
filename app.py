import base64
import logging
import yaml
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from utils import load_model, preprocess_image, predict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load parameters from YAML file
with open('params.yaml') as f:
    params = yaml.safe_load(f)

app = Flask(__name__)
CORS(app)

# Initialize model once at app startup
logger.info("Initializing model...")
model = load_model()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    logger.info("Home route accessed.")
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image_data = request.json.get('image')
        logger.info("Received image data: %s", image_data)  # Debugging line
        
        if not image_data:
            logger.warning("No image data received.")
            return jsonify({"status": "error", "message": "No image data received"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(image_data)
        
        # Write the binary data to a file
        with open("uploaded_image.jpg", "wb") as img_file:
            img_file.write(image_data)
            logger.info("Image written to file: uploaded_image.jpg")

        # Preprocess the image
        img_array = preprocess_image("uploaded_image.jpg")
        predicted_class, confidence = predict(model, img_array)

        # Return JSON response with class and confidence
        logger.info("Returning prediction result.")
        return jsonify({
            "status": "success",
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
