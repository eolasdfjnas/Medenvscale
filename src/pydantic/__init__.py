from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints


def _load_real_pydantic() -> object | None:
    """Prefer the installed pydantic package over this local fallback shim."""
    current_file = Path(__file__).resolve()
    search_paths: list[str] = []
    for raw_path in sys.path:
        path = Path(raw_path or ".").resolve()
        try:
            local_init = (path / "pydantic" / "__init__.py").resolve()
        except OSError:
            search_paths.append(raw_path)
            continue
        if local_init == current_file:
            continue
        search_paths.append(raw_path)

    spec = importlib.machinery.PathFinder.find_spec(__name__, search_paths)
    if spec is None or spec.loader is None or spec.origin is None:
        return None
    try:
        if Path(spec.origin).resolve() == current_file:
            return None
    except OSError:
        pass

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(__name__)
    sys.modules[__name__] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is not None:
            sys.modules[__name__] = previous
        else:
            sys.modules.pop(__name__, None)
        return None
    return module


_real_pydantic = _load_real_pydantic()

if _real_pydantic is not None:
    globals().update(_real_pydantic.__dict__)
else:

    class ValidationError(ValueError):
        pass


    class ConfigDict(dict):
        pass


    class BaseModel:
        model_config = ConfigDict()

        def __init__(self, **kwargs: Any) -> None:
            annotations = self._all_annotations()
            for name, annotation in annotations.items():
                if name in kwargs:
                    value = kwargs[name]
                elif hasattr(self.__class__, name):
                    default = getattr(self.__class__, name)
                    value = deepcopy(default)
                else:
                    raise TypeError(f"Missing required field: {name}")
                setattr(self, name, self._coerce_value(annotation, value))

        @classmethod
        def _all_annotations(cls) -> dict[str, Any]:
            annotations: dict[str, Any] = {}
            for base in reversed(cls.__mro__):
                try:
                    annotations.update(get_type_hints(base))
                except Exception:
                    annotations.update(getattr(base, "__annotations__", {}))
            return annotations

        @classmethod
        def _coerce_value(cls, annotation: Any, value: Any) -> Any:
            origin = get_origin(annotation)
            args = get_args(annotation)

            if value is None:
                return None
            if origin is list and args:
                return [cls._coerce_value(args[0], item) for item in value]
            if origin is dict and len(args) == 2:
                return {item_key: cls._coerce_value(args[1], item_value) for item_key, item_value in value.items()}
            if origin is tuple and args:
                return tuple(cls._coerce_value(args[0], item) for item in value)
            if origin is not None and type(None) in args:
                non_none = [arg for arg in args if arg is not type(None)]
                return cls._coerce_value(non_none[0], value) if non_none else value
            try:
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    if isinstance(value, annotation):
                        return value
                    return annotation.model_validate(value)
            except TypeError:
                pass
            return value

        def model_dump(self) -> dict[str, Any]:
            payload = {}
            for name in self._all_annotations():
                value = getattr(self, name)
                if isinstance(value, BaseModel):
                    payload[name] = value.model_dump()
                elif isinstance(value, list):
                    payload[name] = [item.model_dump() if isinstance(item, BaseModel) else item for item in value]
                elif isinstance(value, dict):
                    payload[name] = {
                        item_key: item_value.model_dump() if isinstance(item_value, BaseModel) else item_value
                        for item_key, item_value in value.items()
                    }
                else:
                    payload[name] = value
            return payload

        @classmethod
        def model_validate(cls, payload: Any):
            if isinstance(payload, cls):
                return payload
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict for {cls.__name__}.")
            return cls(**payload)

        def model_copy(self, update: dict[str, Any] | None = None, deep: bool = False):
            payload = deepcopy(self.model_dump()) if deep else self.model_dump()
            if update:
                payload.update(update)
            return self.__class__.model_validate(payload)

        def __repr__(self) -> str:
            args = ", ".join(f"{key}={value!r}" for key, value in self.model_dump().items())
            return f"{self.__class__.__name__}({args})"
