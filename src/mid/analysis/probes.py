"""Supervised probes for character / scene / act / verse concepts. Labels come from Folger XML tagger.

Owner: 
"""

from __future__ import annotations


def train_probe(features, labels):
    raise NotImplementedError


def evaluate_probe(probe, features, labels) -> dict:
    raise NotImplementedError
