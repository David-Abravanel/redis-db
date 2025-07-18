import uuid
from typing import Any
from .._types import TypeRegistry, compile_validator, is_optional_type
from .._fields import Field, AutoField
from .._exceptions import InvalidFieldError, NoneFieldError


class BaseModelMeta(type):
    def __new__(cls, name, bases, attrs):
        declared_fields = {}
        annotations = attrs.get('__annotations__', {}).copy()

        if "id" not in annotations:
            annotations["id"] = str
            attrs["id"] = AutoField()

        # הכנה לשדות
        for field_name, field_type in annotations.items():
            value = attrs.get(field_name, None)

            if isinstance(value, Field):
                value.field_type = field_type
                value.validator = compile_validator(field_type)
                declared_fields[field_name] = value

            else:
                field = Field(field_type=field_type, default=value)
                field.validator = compile_validator(field_type)
                declared_fields[field_name] = field

        attrs["_fields"] = declared_fields

        def make_init(fields: dict[str, Field]):
            args = ", ".join(f"{name}=None" for name in fields)
            args += ", allow_optional_fields=False"
            body = ["self.__data__ = {}"]

            for name, field in fields.items():
                line = (
                    f"value = {name} if {name} is not None else "
                    f"{repr(field.default) if field.default is not None else 'str(uuid.uuid4())' if isinstance(field, AutoField) else 'None'}"
                )
                optional_none_check = (
                    f"if not allow_optional_fields and value is None and not is_optional_type(fields['{name}'].field_type):\n"
                    f"    raise NoneFieldError('{name}', fields['{name}'].field_type)"
                )
                check = (
                    f"if value is not None and not fields['{name}'].validator(value):\n"
                    f"    raise InvalidFieldError('{name}', fields['{name}'].field_type, value)"
                )
                assign = f"self.__data__['{name}'] = value"
                body.append(line)
                body.extend(optional_none_check.split('\n'))
                body.extend(check.split('\n'))
                body.append(assign)

            src = f"def __init__(self, {args}):\n    " + "\n    ".join(body)
            ns = {
                'uuid': uuid,
                'InvalidFieldError': InvalidFieldError,
                'NoneFieldError': NoneFieldError,
                'fields': fields,
                'is_optional_type': is_optional_type,
            }
            exec(src, ns)
            return ns['__init__']

        init_func = make_init(declared_fields)
        attrs['__init__'] = init_func
        return super().__new__(cls, name, bases, attrs)


class BaseModel(metaclass=BaseModelMeta):
    id: str
    __slots__ = ('__data__',)

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
        items = {"id": TypeRegistry.serialize(self.__data__["id"])}
        items.update(
            {
                key: TypeRegistry.serialize(value)
                for key, value in self.__data__.items()
                if key != "id"
            }
        )
        return items

    # def __getattr__(self, item):
    #     if item in self.__data__:
    #         return self.__data__[item]
    #     raise AttributeError(f"{item} not found")
    
    def __getattribute__(self, name):
        _fields = super().__getattribute__('_fields')
        __data__ = super().__getattribute__('__data__')
        if name in _fields:
            return __data__[name]
        return super().__getattribute__(name)


    def __setattr__(self, key, value):
        if key in ("__data__", "_fields"):
            super().__setattr__(key, value)
        elif key in self._fields:
            field = self._fields[key]
            if not field.validator(value):
                raise InvalidFieldError(key, field.field_type, value)
            self.__data__[key] = value
        else:
            raise InvalidFieldError(key)

    def __repr__(self):
        indent = " " * (len(self.__class__.__name__) + 1)
        id_line = f"{indent}id = {self.__data__['id']!r},\n" if "id" in self.__data__ else ""
        other_fields = "\n".join(f"{indent}{k} = {v!r}," for k, v in self.__data__.items() if k != "id")
        return f"{self.__class__.__name__}(\n{id_line}{other_fields}\n)"

    def __str__(self):
        return self.__repr__()
