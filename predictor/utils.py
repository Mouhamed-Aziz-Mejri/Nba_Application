"""Utility helpers for the predictor app."""

from __future__ import annotations

from typing import Iterable, List, TypeVar

T = TypeVar("T")


def supprime_redondance(items: Iterable[T]) -> List[T]:
    """Supprime les éléments dupliqués tout en conservant l'ordre.

    Args:
        items: Un itérable d'éléments comparables.

    Returns:
        Une liste avec chaque élément unique, dans l'ordre de première apparition.

    Exemple:
        >>> supprime_redondance(['f', 'h', 'f', 'a', 'a', 'a'])
        ['f', 'h', 'a']
    """

    # dict.fromkeys conserve l'ordre d'insertion (Python 3.7+)
    return list(dict.fromkeys(items))
