# redorm/_exceptions/exceptions.py

class RedORMException(Exception):
    """Base exception for all RedORM-related errors."""
    pass

class UnknownFieldTypeError(RedORMException):
    def __init__(self, field_type):
        super().__init__(
            f"[Type Error] The field type '{field_type}' is not registered in the TypeRegistry."
        )

class SerializationError(RedORMException):
    def __init__(self, value):
        super().__init__(
            f"[Serialization Error] Cannot serialize value: {value} (type: {type(value)})\n"
        )

class DeserializationError(RedORMException):
    def __init__(self, field_type, value):
        super().__init__(
            f"[Deserialization Error] Cannot deserialize value '{value}' for type '{field_type}'\n"
            f"💡 Suggestion: Check if value is a valid string for that type or re-register a safer deserializer."
        )

class NoneFieldError(RedORMException):
    def __init__(self, field_name, filed_type):
        super().__init__(
            f"[None filed Error] Expect value of type '{filed_type}' for field: '{field_name}', got: 'None'\n"
            f"💡 Suggestion: use the 'Optional' type for Optional types or add attribute allow_optional_fields=True to the class."
        )


class InvalidFieldError(RedORMException):
    def __init__(self, field_name: str, expected_type=None, value=None):
        if expected_type is not None and value is not None:
            message = (
                f"[Type Error] Field '{field_name}' expects type: '{expected_type}', "
                f"got: '{value!r}'\n"
                f"💡 Make sure you assign a value of the correct type."
            )
        else:
            message = (
                f"[Model Error] Attempted to access or set invalid field '{field_name}'\n"
                f"💡 Suggestion: Ensure the field is defined in the model class and spelled correctly."
            )

        super().__init__(message)
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_type = type(value) if value is not None else None
