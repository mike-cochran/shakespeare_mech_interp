"""Wandb / tensorboard wrappers.

Owner:
"""

from __future__ import annotations


def init_run(project: str, name: str, config: dict):
    raise NotImplementedError


def log(metrics: dict, step: int):
    raise NotImplementedError
