import yaml
import logging.config

def setup_logging():
    with open("config/logging.yaml", "r", encoding="utf-8") as f:
        logging.config.dictConfig(yaml.safe_load(f))