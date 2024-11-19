import tensorflow as tf
from tensorflow.keras import layers
import yaml
from logger import logger
from pathlib import Path

class DataPreprocessing:
    def __init__(self, params_path=Path('params.yaml')):
        with open(params_path) as f:
            self.params = yaml.safe_load(f)['augmentation']
        
    def preprocess(self, train_ds, val_ds, test_ds):
        try:
            logger.info("Preprocessing started.")
            shuffle_size = yaml.safe_load(Path('params.yaml').open())['data_preprocessing']['shuffle_size']
            train_ds = train_ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_ds = test_ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            logger.info("Preprocessing completed successfully.")
            return train_ds, val_ds, test_ds
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def augmentation_layer(self):
        return tf.keras.Sequential([
            layers.RandomFlip(self.params['flip']),
            layers.RandomRotation(self.params['rotation']),
        ])
