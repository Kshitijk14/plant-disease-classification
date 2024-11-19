import tensorflow as tf
import yaml
from logger import logger
from pathlib import Path

class DataIngestion:
    def __init__(self, params_path=Path('params.yaml')):
        with open(params_path) as f:
            params = yaml.safe_load(f)
        ingestion_params = params['data_ingestion']
        
        self.directory_path = Path(ingestion_params['directory_path'])
        self.IMAGE_SIZE = ingestion_params['image_size']
        self.BATCH_SIZE = ingestion_params['batch_size']
        self.class_names = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_data(self):
        try:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.directory_path,
                shuffle=True,
                image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                batch_size=self.BATCH_SIZE
            )
            logger.info("Data loading started.")
            self.class_names = dataset.class_names
            self.train_ds, self.val_ds, self.test_ds = self.get_dataset_partitions_tf(dataset)
            logger.info("Data loading completed successfully.")
            return self.train_ds, self.val_ds, self.test_ds, self.class_names
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            raise

    def get_dataset_partitions_tf(self, ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
        ds_size = len(ds)
        shuffle_size = yaml.safe_load(Path('params.yaml').open())['data_preprocessing']['shuffle_size']
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        return train_ds, val_ds, test_ds
