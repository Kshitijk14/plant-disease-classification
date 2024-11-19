import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from logger import logger

class Predictor:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def predict(self, img):
        try:
            logger.info("Prediction started for a new image.")
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = self.model.predict(img_array)
            predicted_class = self.class_names[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)
            logger.info(f"Prediction completed: Class - {predicted_class}, Confidence - {confidence}%")
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def display_predictions(self, test_ds):
        plt.figure(figsize=(15, 15))
        for images, labels in test_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                predicted_class, confidence = self.predict(images[i].numpy())
                actual_class = self.class_names[labels[i]]
                plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Confidence: {confidence}%")
                plt.axis("off")
        plt.show()
