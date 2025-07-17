# redorm/_exceptions/exceptions.py

class RedORMException(Exception):
    """Base exception for all RedORM-related errors."""
    pass

class UnknownFieldTypeError(RedORMException):
    def __init__(self, field_type):
        super().__init__(
            f"[Type Error] The field type '{field_type}' is not registered in the TypeRegistry.\n"
            f"💡 Suggestion: Register this type using `TypeRegistry.register({field_type}, serializer, deserializer)`"
        )

class SerializationError(RedORMException):
    def __init__(self, value):
        super().__init__(
            f"[Serialization Error] Cannot serialize value: {value} (type: {type(value)})\n"
            f"💡 Suggestion: Ensure this type is registered in TypeRegistry, or provide a custom serializer."
        )

class DeserializationError(RedORMException):
    def __init__(self, field_type, value):
        super().__init__(
            f"[Deserialization Error] Cannot deserialize value '{value}' for type '{field_type}'\n"
            f"💡 Suggestion: Check if value is a valid string for that type or re-register a safer deserializer."
        )

class InvalidFieldError(RedORMException):
    def __init__(self, field_name):
        super().__init__(
            f"[Model Error] Attempted to access or set invalid field '{field_name}'\n"
            f"💡 Suggestion: Ensure the field is defined in the model class, and spelled correctly."
        )
