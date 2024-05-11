import json


class AnchorConfig:
    def __init__(self, settings):
        for key, value in settings.items():
            setattr(self, key, value)
        

class Config:
    def __init__(self, config_file, model, **kwargs):
        self.config_file = config_file
        self.model = model
        self.load_config()
        
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        
        
    def load_config(self):
        with open(self.config_file, 'r') as file:
            config_data = json.load(file)

            # Load and set common settings
            for key, value in config_data.get('common', {}).items():
                setattr(self, key, value)

            # Load and set model-specific settings
            for key, value in config_data.get(self.model, {}).items():
                if key == "anchor":
                    setattr(self, key, AnchorConfig(value)) 
                else:
                    setattr(self, key, value) 
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}