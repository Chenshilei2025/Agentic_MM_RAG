"""Structured logger. Every tool call, action, and state change is logged."""
import logging, sys

_CONFIGURED = False

def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"))
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)
