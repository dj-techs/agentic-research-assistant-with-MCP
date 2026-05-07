"""Programmatic metrics for the eval harness."""

from __future__ import annotations

from typing import Iterable


def keyword_recall(answer: str, expected: Iterable[str]) -> float:
    """Fraction of expected keywords that appear in the answer (case-insensitive)."""
    expected = list(expected)
    if not expected:
        return 1.0
    a = answer.lower()
    hits = sum(1 for k in expected if k.lower() in a)
    return hits / len(expected)


def specialist_accuracy(used: list[str], expected: list[str]) -> float:
    """Set-overlap (Jaccard) between specialists used and those expected."""
    u, e = set(used), set(expected)
    if not e and not u:
        return 1.0
    if not e:
        return 0.0
    return len(u & e) / len(u | e)


def first_specialist_correct(used: list[str], expected: list[str]) -> int:
    """1 if the first specialist used matches the first expected, else 0."""
    if not used or not expected:
        return 0
    return int(used[0] == expected[0])
