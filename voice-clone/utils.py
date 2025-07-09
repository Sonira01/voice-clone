import json
import torch

class HParams:
    def __init__(self, d):
        self.__dict__.update(d)
    def __getattr__(self, name):
        return self.__dict__[name]

def dict_to_obj(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_obj(v)
    return HParams(d)

def get_hparams_from_file(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return dict_to_obj(data)

def load_checkpoint(path, model, _):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
