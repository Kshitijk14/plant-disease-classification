import yaml
from pathlib import Path
from pipeline.stage_01_data_ingestion import DataIngestion
from pipeline.stage_02_pre_processing import DataPreprocessing
from pipeline.stage_03_model_training import ModelTraining
from pipeline.stage_04_model_evaluation import ModelEvaluation
from pipeline.save_model import SaveModel
from pipeline.predict import Predictor

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(Path('params.yaml').open())

# Stage 1: Data Ingestion
data_ingestion = DataIngestion()
train_ds, val_ds, test_ds, class_names = data_ingestion.load_data()

# Stage 2: Data Preprocessing
data_preprocessing = DataPreprocessing()
train_ds, val_ds, test_ds = data_preprocessing.preprocess(train_ds, val_ds, test_ds)
augmentation_layer = data_preprocessing.augmentation_layer()

# Stage 3: Model Training
model_training = ModelTraining(class_names=class_names)
model = model_training.build_model(augmentation_layer)
history = model_training.train_model(model, train_ds, val_ds)

# Stage 4: Model Evaluation
model_evaluation = ModelEvaluation()
model_evaluation.evaluate_model(model, test_ds)
model_evaluation.plot_training_history(history)

# Save the trained model
model_saver = SaveModel(model)  # No need to pass parameters anymore
model_saver.save()

# Prediction
predictor = Predictor(model, class_names)
predictor.display_predictions(test_ds)
