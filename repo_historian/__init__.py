"""Repo Historian — narrative Git history generator."""

import logging

logger = logging.getLogger("repo_historian")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_handler)
