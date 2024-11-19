import matplotlib.pyplot as plt
from logger import logger

class ModelEvaluation:
    def evaluate_model(self, model, test_ds):
        try:
            logger.info("Model evaluation started.")
            scores = model.evaluate(test_ds)
            print("Model Accuracy and Loss Scores:", scores)
            logger.info("Model evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(acc)), acc, label='Training Accuracy')
        plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(loss)), loss, label='Training Loss')
        plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
