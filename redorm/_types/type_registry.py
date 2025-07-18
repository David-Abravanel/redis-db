from ast import literal_eval
from datetime import datetime
from typing import Any, Callable, Type, Union, get_origin, get_args
from .._exceptions import SerializationError, DeserializationError


class TypeRegistry:
    _registry: dict[Type, tuple[Callable[[Any], str], Callable[[str], Any]]] = {}
    _instance: "TypeRegistry" = None

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



# === טיפוסים בסיסיים ===
TypeRegistry.register(str, lambda x: x, lambda x: x) 
TypeRegistry.register(int, str, int)
TypeRegistry.register(float, str, float)
TypeRegistry.register(bool, lambda x: "True" if x else "False", lambda x: x == "True")
TypeRegistry.register(datetime, lambda dt: dt.isoformat(), lambda s: datetime.fromisoformat(s))

# טיפוסים מורכבים
TypeRegistry.register(list, repr, literal_eval)
TypeRegistry.register(tuple, repr, literal_eval)
TypeRegistry.register(set, repr, literal_eval)
TypeRegistry.register(dict, repr, literal_eval)



# def validate_type(value: Any, expected_type: type) -> bool:
#     origin = get_origin(expected_type)
#     args = get_args(expected_type)

#     if origin is None:
#         return isinstance(value, expected_type)

#     if origin is list:
#         return isinstance(value, list) and all(validate_type(item, args[0]) for item in value)

#     if origin is tuple:
#         return isinstance(value, tuple) and all(validate_type(item, args[i]) for i, item in enumerate(value))

#     if origin is set:
#         return isinstance(value, set) and all(validate_type(item, args[0]) for item in value)

#     if origin is dict:
#         return (
#             isinstance(value, dict) and
#             all(validate_type(k, args[0]) and validate_type(v, args[1]) for k, v in value.items())
#         )

#     # fallback
#     return isinstance(value, expected_type)

def compile_validator(expected_type):
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        return lambda x: isinstance(x, expected_type)

    if origin is list:
        inner_validator = compile_validator(args[0])
        return lambda x: isinstance(x, list) and all(inner_validator(i) for i in x)

    if origin is tuple:
        validators = [compile_validator(arg) for arg in args]
        return lambda x: isinstance(x, tuple) and len(x) == len(validators) and all(v(i) for v, i in zip(validators, x))

    if origin is set:
        inner_validator = compile_validator(args[0])
        return lambda x: isinstance(x, set) and all(inner_validator(i) for i in x)

    if origin is dict:
        key_validator = compile_validator(args[0])
        val_validator = compile_validator(args[1])
        return lambda x: isinstance(x, dict) and all(key_validator(k) and val_validator(v) for k, v in x.items())

    # fallback
    return lambda x: isinstance(x, expected_type)


def is_optional_type(tp):
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        return type(None) in args
    return False