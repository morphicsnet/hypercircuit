from __future__ import annotations

from typing import Any, Dict, Mapping, Type


_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {}


def register_adapter(kind: str, name: str, cls: Type[Any]) -> None:
    """Register an adapter class under a (kind, name)."""
    k = str(kind)
    n = str(name)
    _REGISTRY.setdefault(k, {})[n] = cls


def get_adapter(kind: str, name: str) -> Type[Any]:
    k = str(kind)
    n = str(name)
    if k not in _REGISTRY or n not in _REGISTRY[k]:
        raise KeyError(f"Adapter not registered: kind={k} name={n}")
    return _REGISTRY[k][n]


def list_adapters(kind: str) -> Mapping[str, Type[Any]]:
    return dict(_REGISTRY.get(str(kind), {}))

