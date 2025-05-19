import logging
import os

# A common logging format using pipe characters for legibility.
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class RewardLogger:
    def __init__(self, base_dir="./logs/reward_model/ppo"):
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
                       fmt=LOG_FORMAT):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicate logs if re-imported.
        if logger.hasHandlers():
            logger.handlers.clear()

        file_path = os.path.join(self.base_dir, filename)
        handler = logging.FileHandler(file_path)
        formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    
class TrainLogger:
    def __init__(self, base_dir="./logs/model/ppo"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.standard_logger = self._create_logger("train_model", "standard_ppo.log")
        self.naive_logger = self._create_logger("naive_train_model", "naive_ppo.log")
        self.enhanced_logger = self._create_logger("enhanced_train_model", "enhanced_ppo.log")

    def _create_logger(self, name, filename, level=logging.DEBUG, 
                       fmt=LOG_FORMAT):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicate logs if re-imported.
        if logger.hasHandlers():
            logger.handlers.clear()

        file_path = os.path.join(self.base_dir, filename)
        handler = logging.FileHandler(file_path)
        formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
class PretrainLogger:
    def __init__(self, base_dir="./logs/pretrain/ppo"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.gail_logger    = self._create_logger("gail_train", "gail.log")
        self.bc_logger      = self._create_logger("bc_train", "bc.log")

    def _create_logger(self, name, filename, level=logging.DEBUG, 
                       fmt=LOG_FORMAT):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicate logs if re-imported.
        if logger.hasHandlers():
            logger.handlers.clear()

        file_path = os.path.join(self.base_dir, filename)
        handler = logging.FileHandler(file_path)
        formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

# Create a global instance that can be imported elsewhere
reward_logger = RewardLogger()
pretrain_logger = PretrainLogger()
train_logger = TrainLogger()