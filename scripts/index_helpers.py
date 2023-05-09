"""Helpers for querying pair-index payloads."""


def _pairs(payload):
    return list(payload.get("pairs", []))


def _missing(payload):
    return list(payload.get("missing_masks", []))


def index_pairs(payload):
    return _pairs(payload)


def index_missing_masks(payload):
    return _missing(payload)


def index_pair_count(payload):
    return len(_pairs(payload))


def index_missing_count(payload):
    return len(_missing(payload))


def index_origin_names(payload):
    return [pair[0] for pair in _pairs(payload)]


def index_mask_names(payload):
    return [pair[1] for pair in _pairs(payload)]


def index_has_duplicate_origins(payload):
    origins = index_origin_names(payload)
    return len(origins) != len(set(origins))


def index_has_duplicate_masks(payload):
    masks = index_mask_names(payload)
    return len(masks) != len(set(masks))


def index_same_stem_pairs(payload):
    return [pair for pair in _pairs(payload) if pair[0] == pair[1]]


def index_mismatched_stem_pairs(payload):
    return [pair for pair in _pairs(payload) if pair[0] != pair[1]]


def index_missing_ratio(payload):
    total = index_pair_count(payload) + index_missing_count(payload)
    if total == 0:
        return 0.0
    return index_missing_count(payload) / total


def index_same_stem_ratio(payload):
    total = index_pair_count(payload)
    if total == 0:
        return 0.0
    return len(index_same_stem_pairs(payload)) / total
