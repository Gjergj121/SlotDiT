import json
from copy import deepcopy
from pathlib import Path
from CONFIG import DEFAULTS


class Config(dict):
    def __init__(self, exp_path):
        super().__init__(deepcopy(DEFAULTS))
        self.exp_path = Path(exp_path)

    def load_exp_config_file(self, exp_path=None):
        path = Path(exp_path) if exp_path is not None else self.exp_path
        with open(path / "experiment_params.json") as stream:
            params = json.load(stream)
        for section, defaults in DEFAULTS.items():
            if isinstance(defaults, dict):
                params[section] = {**deepcopy(defaults), **params.get(section, {})}
            else:
                params.setdefault(section, deepcopy(defaults))
        return params

    def save_exp_config_file(self, exp_path=None, exp_params=None):
        path = Path(exp_path) if exp_path is not None else self.exp_path
        with open(path / "experiment_params.json", "w") as stream:
            json.dump(self if exp_params is None else exp_params, stream, indent=2)
