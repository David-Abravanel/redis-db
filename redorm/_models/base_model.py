# redorm/_models/base_model.py
import uuid
from typing import Any, ClassVar, get_origin
from _types import TypeRegistry
from _fields import Field, AutoField
from _exceptions import InvalidFieldError

class BaseModelMeta(type):
    def __new__(cls, name, bases, attrs):
        declared_fields = {}

        annotations = attrs.get('__annotations__', {})
        if "id" not in annotations:
            annotations["id"] = str
            attrs["id"] = AutoField()

        for field_name, field_type in annotations.items():
            value = attrs.get(field_name, None)

            if isinstance(value, Field):
                value.field_type = field_type  # Inject type if not provided
                declared_fields[field_name] = value
            else:
                # Create implicit field
                declared_fields[field_name] = Field(field_type)

        attrs["_fields"] = declared_fields
        return super().__new__(cls, name, bases, attrs)

class BaseModel(metaclass=BaseModelMeta):
    def __init__(self, **kwargs):
        self.__data__ = {}

        for field_name, field in self._fields.items():
            if field_name in kwargs:
                value = kwargs[field_name]
            elif field.default is not None:
                value = field.default
            elif isinstance(field, AutoField):
                value = str(uuid.uuid4())
            else:
                value = None

            # טיפוס קשיח
            if value is not None and not isinstance(value, field.field_type):
                raise TypeError(f"Field '{field_name}' expects {field.field_type}, got {type(value)}")

            self.__data__[field_name] = value

    def to_dict(self) -> dict[str, Any]:
        return self.__data__

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        deserialized = {}
        for field_name, field in cls._fields.items():
            raw_value = data.get(field_name)
            deserialized[field_name] = TypeRegistry.deserialize(field.field_type, raw_value)
        return cls(**deserialized)

    def serialize(self) -> dict[str, str]:
        return {
            key: TypeRegistry.serialize(value)
            for key, value in self.__data__.items()
        }

    def __getattr__(self, item):
        return self.__data__.get(item)

    def __setattr__(self, key, value):
        if key in ("__data__", "_fields"):
            super().__setattr__(key, value)
        elif key in self._fields:
            expected_type = self._fields[key].field_type
            if not isinstance(value, expected_type):
                raise TypeError(f"Field '{key}' expects {expected_type}, got {type(value)}")
            self.__data__[key] = value
        else:
            raise InvalidFieldError(key)

    def __repr__(self):
        fields = ", ".join(f"{k}={v}" for k, v in self.__data__.items())
        return f"<{self.__class__.__name__}({fields})>"
