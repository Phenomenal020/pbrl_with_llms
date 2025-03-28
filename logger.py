import logging
import os

class Logger:
    def __init__(self, base_dir="./frozen_lake/reward_model/logs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.queries_logger      = self._create_logger("queries", "queries.log")
        self.data_logger         = self._create_logger("add_data", "add_data.log")
        self.train_logger        = self._create_logger("train", "train.log")
        self.put_logger          = self._create_logger("put_queries", "put_queries.log")
        self.true_labels_logger  = self._create_logger("get_true_labels", "get_true_labels.log")
        self.pred_labels_logger  = self._create_logger("get_pred_labels", "get_pred_labels.log")
        self.llm_labels_logger   = self._create_logger("get_llm_labels", "get_llm_labels.log")
        self.save_model_logger   = self._create_logger("save_model", "savemodel.log")
        self.load_model_logger   = self._create_logger("load_model", "load_model.log")

    def _create_logger(self, name, filename, level=logging.DEBUG, 
                       fmt='%(asctime)s - %(levelname)s - %(message)s'):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicate logs if re-imported.
        if logger.hasHandlers():
            logger.handlers.clear()

        file_path = os.path.join(self.base_dir, filename)
        handler = logging.FileHandler(file_path)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

# Create a global instance that can be imported elsewhere
logger = Logger()
