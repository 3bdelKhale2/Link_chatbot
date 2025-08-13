# logger.py
import logging
from logging.handlers import RotatingFileHandler


def init_logger():
    logger = logging.getLogger("football_ticket_assistant")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        "assistant.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
