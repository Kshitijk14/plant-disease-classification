from pathlib import Path
import yaml
from logger import logger

class SaveModel:
    def __init__(self, model, params_path='params.yaml'):
        self.model = model
        self.params_path = Path(params_path)
        
        # Load model parameters from the params.yaml file
        with self.params_path.open() as f:
            self.params = yaml.safe_load(f)['output']
        
        self.model_dir = Path(self.params['model_dir'])
        self.version = self.params['version']
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        try:
            model_save_path = self.model_dir / f'model_v{self.version}'
            self.model.save(model_save_path)
            logger.info(f"Model saved successfully at: {model_save_path}")
        except Exception as e:
            logger.error(f"Error during model saving: {e}")
            raise
