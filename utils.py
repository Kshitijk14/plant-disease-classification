# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Configuration variables
# IMAGE_SIZE = 256
# BATCH_SIZE = 32
# MODEL_PATH = r'artifacts/model/model.keras'
# CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# # Load the model
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# # Preprocess image for prediction
# # def preprocess_image(img_path):
# #     img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
# #     img_array = image.img_to_array(img) / 255.0  # Normalize to [0,1] range
# #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# #     return img_array

# def preprocess_image(img):
#     """
#     Preprocess image for prediction
#     Args:
#         img: PIL Image object or path to image
#     Returns:
#         Preprocessed numpy array ready for prediction
#     """
#     if isinstance(img, str):  # If path is provided
#         img = Image.open(img)
    
#     # Resize the image
#     img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
#     # Convert to numpy array and normalize
#     img_array = np.array(img) / 255.0
    
#     # Add batch dimension if not present
#     if len(img_array.shape) == 3:
#         img_array = np.expand_dims(img_array, axis=0)
    
#     print("Input shape:", img_array.shape)
#     print("Input array:", img_array)
#     return img_array

# # Predict on a single image
# def predict(model, img_array):
#     predictions = model.predict(img_array)
#     print("Raw predictions:", predictions)
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)
#     return predicted_class, confidence

# # Test the functions individually
# if __name__ == "__main__":
#     # Load the model once and reuse it
#     model = load_model()

#     # Path to the test image
#     test_image_path = '../path/to/your/test/image.jpg'
#     img_array = preprocess_image(test_image_path)
#     predicted_class, confidence = predict(model, img_array)

#     print(f"Predicted class: {predicted_class}, Confidence: {confidence}%")


import numpy as np
from PIL import Image
import tensorflow as tf
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load parameters from YAML file
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Configuration variables
IMAGE_SIZE = params['data_ingestion']['image_size']
BATCH_SIZE = params['data_ingestion']['batch_size']
MODEL_PATH = r'artifacts/model/model.keras'
CLASS_NAMES = params['data_ingestion']['class_names']

# Load the model
def load_model():
    logger.info("Loading model from: %s", MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# Preprocess image for prediction
def preprocess_image(img):
    """
    Preprocess image for prediction
    Args:
        img: PIL Image object or path to image
    Returns:
        Preprocessed numpy array ready for prediction
    """
    if isinstance(img, str):  # If path is provided
        img = Image.open(img)

    logger.info("Resizing image to %s", (IMAGE_SIZE, IMAGE_SIZE))
    
    # Resize the image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension if not present
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    logger.debug("Input shape: %s", img_array.shape)
    return img_array

# Predict on a single image
def predict(model, img_array):
    logger.info("Predicting on image with shape: %s", img_array.shape)
    predictions = model.predict(img_array)
    logger.debug("Raw predictions: %s", predictions)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    logger.info("Predicted class: %s, Confidence: %s%%", predicted_class, confidence)
    return predicted_class, confidence
