import yaml
import argparse
import json
from pathlib import Path

class BaseTrainConfig():
    def __init__(self, **config):
        self.__dict__.update(config) 
        self.setup()
        self.print()
    def setup(self):
        try:
            result_dir = Path(self.result_dir)/self.exp_name
            result_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = Path(self.checkpoint_dir)/self.exp_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except:
            pass
        
    def print(self):
        filename = Path(self.result_dir)/Path(self.exp_name)/'config.txt'
        with open(filename, 'w') as file:
            file.write(json.dumps(self.__dict__)) 
            
            
def read_yaml(path):
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    return config

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/train.yaml')
    # parser.add_argument('--log_dir', '-l', type=str, default='log')
    args = parser.parse_args()
    config_file = args.config
    config = BaseTrainConfig(**read_yaml(config_file))
    
    return config