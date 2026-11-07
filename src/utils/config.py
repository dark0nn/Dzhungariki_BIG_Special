import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path) as f: cfg = yaml.safe_load(f)
    if '_base_' in cfg:
        base_cfg = load_config(str(Path(config_path).parent / cfg['_base_']))
        for k, v in cfg.items():
            if k != '_base_': base_cfg[k] = v
        cfg = base_cfg
    return cfg
