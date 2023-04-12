"""Helpers for querying pair-index payloads."""


def _pairs(payload):
    return list(payload.get("pairs", []))


def _missing(payload):
    return list(payload.get("missing_masks", []))


def index_pairs(payload):
    return _pairs(payload)


def index_missing_masks(payload):
    return _missing(payload)
