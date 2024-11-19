from pathlib import Path
import tensorflow as tf
from tensorflow.keras import models, layers
import yaml
from logger import logger

class ModelTraining:
    def __init__(self, params_path=Path('params.yaml'), class_names=None):
        with open(params_path) as f:
            self.params = yaml.safe_load(f)
        self.IMAGE_SIZE = self.params['data_ingestion']['image_size']
        self.CHANNELS = self.params['model_training']['channels']
        self.class_names = class_names
        self.model = None

    def build_model(self, augmentation_layer):
        resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(self.IMAGE_SIZE, self.IMAGE_SIZE),
            layers.Rescaling(1.0 / 255)
        ])

        self.model = models.Sequential([resize_and_rescale, augmentation_layer])

        # Add convolutional layers dynamically
        for conv_layer in self.params['model_training']['conv_layers']:
            self.model.add(layers.Conv2D(conv_layer['filters'], conv_layer['kernel_size'], activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
        
        # Dense and output layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.params['model_training']['dense_units'], activation='relu'))
        self.model.add(layers.Dense(len(self.class_names), activation='softmax'))
        
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        return self.model

    def train_model(self, model, train_ds, val_ds):
        try:
            logger.info("Model training started.")
            epochs = self.params['model_training']['epochs']
            history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
            logger.info("Model training completed successfully.")
            return history
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
