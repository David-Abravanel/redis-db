import operator
import re
from typing import Any, Generic, List, Optional, Dict, TypeVar, Type

T = TypeVar('T', bound='RedisBaseModel')

# Mapping of supported filter operators to their corresponding Python functions or lambdas.
FILTER_OPERATORS = {
    "eq": operator.eq,
    "not_eq": operator.ne,
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
    "starts_with": lambda val, test: isinstance(val, str) and val.startswith(test),
    "ends_with": lambda val, test: isinstance(val, str) and val.endswith(test),
    "contains": lambda val, test: isinstance(val, str) and test in val,
}

def split_filter_key(filter_key: str) -> (str, str):
    if "__" in filter_key:
        field_name, operator_name = filter_key.split("__", 1)
        if operator_name not in FILTER_OPERATORS:
            raise ValueError(f"Unknown operator '{operator_name}' in filter key '{filter_key}'")
        return field_name, operator_name
    return filter_key, "eq"

class RedisQuerySet(Generic[T]):
    def __init__(self, model_class: Type[T], fields: Optional[List[str]] = None):
        self.model_class = model_class
        self.filters: Dict[str, Any] = {}
        self.order_field: Optional[str] = None
        self.order_descending: bool = False
        self.limit_count: Optional[int] = None
        self.offset_count: int = 0
        self.selected_fields: Optional[List[str]] = fields

    def select(self, fields: Optional[List[str]] = None) -> "RedisQuerySet":
        self.selected_fields = fields
        return self

    def where(self, **filters) -> "RedisQuerySet":
        self.filters.update(filters)
        return self

    def order_by(self, field_name: str) -> "RedisQuerySet":
        if field_name.startswith("-"):
            self.order_descending = True
            self.order_field = field_name[1:]
        else:
            self.order_descending = False
            self.order_field = field_name
        return self

    def limit(self, count: int) -> "RedisQuerySet":
        self.limit_count = count
        return self

    def offset(self, count: int) -> "RedisQuerySet":
        self.offset_count = count
        return self

    async def all(self) -> List[T]:
        redis_connection = self.model_class.get_redis()
        key_prefix = self.model_class.get_key_prefix()

        simple_eq_filters = {
            key: val for key, val in self.filters.items()
            if "__" not in key or key.endswith("__eq")
        }
        complex_filters = {
            key: val for key, val in self.filters.items()
            if key not in simple_eq_filters
        }

        if not simple_eq_filters:
            matched_ids = await self.model_class._resolve_ids(None)
        else:
            sets_of_ids = []
            for filter_key, filter_value in simple_eq_filters.items():
                field, _ = split_filter_key(filter_key)
                index_key = f"{key_prefix}:by_{field}:{filter_value}"
                members = await redis_connection.smembers(index_key)
                members_decoded = {
                    member.decode() if isinstance(member, bytes) else member
                    for member in members
                }
                sets_of_ids.append(members_decoded)
            if not sets_of_ids:
                return []
            matched_ids = list(set.intersection(*sets_of_ids))

        if not matched_ids:
            return []

        matched_objects = await self.model_class.get(ids=matched_ids, fields=self.selected_fields)

        filtered_objects = []
        for obj in matched_objects:
            if obj is None:
                continue
            passes_filters = True
            for filter_key, expected_value in complex_filters.items():
                field, operator_name = split_filter_key(filter_key)
                actual_value = getattr(obj, field, None)
                operator_func = FILTER_OPERATORS.get(operator_name)
                if operator_func is None:
                    raise ValueError(f"Unsupported operator: {operator_name}")
                if not operator_func(actual_value, expected_value):
                    passes_filters = False
                    break
            if passes_filters:
                filtered_objects.append(obj)

        if self.order_field:
            def get_sort_key(obj):
                val = getattr(obj, self.order_field, None)
                if val is None and hasattr(obj, "_meta"):
                    val = getattr(obj._meta, self.order_field, None)
                if isinstance(val, str):
                    return natural_sort_key(val)
                return val
            filtered_objects.sort(key=get_sort_key, reverse=self.order_descending)

        start_index = self.offset_count
        end_index = start_index + self.limit_count if self.limit_count is not None else None

        return filtered_objects[start_index:end_index]
    
    async def count(self) -> int:
        """Count matching objects without retrieving them."""
        # For count, we only need IDs
        redis_connection = self.model_class.get_redis()
        key_prefix = self.model_class.get_key_prefix()
        
        simple_eq_filters = {
            key: val for key, val in self.filters.items()
            if "__" not in key or key.endswith("__eq")
        }
        
        if not simple_eq_filters:
            # Count all keys
            pattern = f"{key_prefix}:*"
            count = 0
            async for key in redis_connection.scan_iter(match=pattern):
                count += 1
            return count
        
        # Use intersection for count
        sets_of_ids = []
        for filter_key, filter_value in simple_eq_filters.items():
            field, _ = split_filter_key(filter_key)
            index_key = f"{key_prefix}:by_{field}:{filter_value}"
            cardinality = await redis_connection.scard(index_key)
            if cardinality == 0:
                return 0
            sets_of_ids.append(index_key)
        
        if len(sets_of_ids) == 1:
            return await redis_connection.scard(sets_of_ids[0])
        
        # For multiple sets, we need to compute intersection
        # This is a simplified approach - for exact count with complex filters,
        # we'd need to retrieve and filter
        matched_ids = await self._get_indexed_ids(key_prefix, simple_eq_filters, {})
        return len(matched_ids)

    async def first(self) -> Optional[T]:
        self.limit(1)
        results = await self.all()
        return results[0] if results else None

def natural_sort_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', value)
    ]
