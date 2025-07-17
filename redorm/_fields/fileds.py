# redorm/_fields/fields.py
class Field:
    def __init__(self, field_type=None, default=None, indexed=False):
        self.field_type = field_type
        self.default = default
        self.indexed = indexed

class AutoField(Field):
    def __init__(self):
        super().__init__(str, default=None)
