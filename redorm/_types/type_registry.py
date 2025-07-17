# redorm/_types/type_registry.py
from datetime import datetime
from typing import Any, Callable, Type
from _exceptions import (
    SerializationError,
    DeserializationError,
)

class TypeRegistry:
    _registry: dict[Type, tuple[Callable[[Any], str], Callable[[str], Any]]] = {}

    @classmethod
    def register(cls, field_type: Type, serializer: Callable, deserializer: Callable):
        cls._registry[field_type] = (serializer, deserializer)

    @classmethod
    def serialize(cls, value: Any) -> str:
        if value is None:
            return ""
        value_type = type(value)
        if value_type not in cls._registry:
            raise SerializationError(value)
        return cls._registry[value_type][0](value)

    @classmethod
    def deserialize(cls, field_type: Type, value: str) -> Any:
        if value == "":
            return None
        if field_type not in cls._registry:
            raise DeserializationError(field_type, value)
        return cls._registry[field_type][1](value)

# # === טיפוסים בסיסיים ===
# TypeRegistry.register(str, str, str)
# TypeRegistry.register(int, str, int)
# TypeRegistry.register(float, str, float)
# TypeRegistry.register(bool, lambda x: "1" if x else "0", lambda x: x == "1")
# TypeRegistry.register(datetime, lambda dt: dt.isoformat(), lambda s: datetime.fromisoformat(s))

# # === טיפוסים מורכבים (מהירים) ===

# def serialize_list(lst: list) -> str:
#     return "|".join(str(i) for i in lst)

# def deserialize_list(s: str) -> list:
#     return s.split("|") if s else []

# TypeRegistry.register(list, serialize_list, deserialize_list)

# def serialize_tuple(t: tuple) -> str:
#     return "|".join(str(i) for i in t)

# def deserialize_tuple(s: str) -> tuple:
#     return tuple(s.split("|")) if s else ()

# TypeRegistry.register(tuple, serialize_tuple, deserialize_tuple)

# def serialize_set(s: set) -> str:
#     return "|".join(str(i) for i in s)

# def deserialize_set(s: str) -> set:
#     return set(s.split("|")) if s else set()

# TypeRegistry.register(set, serialize_set, deserialize_set)

# def serialize_dict(d: dict) -> str:
#     return ";".join(f"{k}={v}" for k, v in d.items())

# def deserialize_dict(s: str) -> dict:
#     if not s:
#         return {}
#     return dict(item.split("=", 1) for item in s.split(";"))

# TypeRegistry.register(dict, serialize_dict, deserialize_dict)
