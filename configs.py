import stow
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_name = 'captcha-3'
        self.model_path = 'models/captcha-3.keras'
        self.data_dir = 'samples/captcha-3'
        self.vocab = []
        self.height = 40
        self.width = 400
        self.char_height = 20
        self.char_width = 20
        self.max_text_length = 15
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.train_epochs = 50
        self.train_workers = 20
        self.downsample_factor = 4
        self.shuffle = True
        self.samples = 4
        self.needs_object_detection = False
        self.custom_char_map = True
        self.continue_training = True
        self.base_model = 'models/captcha-3.keras'
