# redisorm/core/query.py
import operator
import re
import asyncio
from typing import Any, Generic, List, Optional, Dict, TypeVar, Type, Set

T = TypeVar('T', bound='RedisBaseModel')

# Enhanced filter operators with optimizations
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
    "in": lambda val, test: val in test,
    "not_in": lambda val, test: val not in test,
    "isnull": lambda val, test: (val is None) == test,
    "regex": lambda val, test: isinstance(val, str) and re.search(test, val) is not None,
}

def split_filter_key(filter_key: str) -> (str, str):
    """Split filter key into field name and operator."""
    if "__" in filter_key:
        field_name, operator_name = filter_key.split("__", 1)
        if operator_name not in FILTER_OPERATORS:
            raise ValueError(f"Unknown operator '{operator_name}' in filter key '{filter_key}'")
        return field_name, operator_name
    return filter_key, "eq"

def natural_sort_key(value: str):
    """Natural sorting key for strings with numbers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', value)
    ]

class RedisQuerySet(Generic[T]):
    """
    Optimized QuerySet with enhanced filtering and performance improvements.
    """
    
    def __init__(self, model_class: Type[T], fields: Optional[List[str]] = None):
        self.model_class = model_class
        self.filters: Dict[str, Any] = {}
        self.order_field: Optional[str] = None
        self.order_descending: bool = False
        self.limit_count: Optional[int] = None
        self.offset_count: int = 0
        self.selected_fields: Optional[List[str]] = fields
        self._prefetch_related: Set[str] = set()

    def select(self, fields: Optional[List[str]] = None) -> "RedisQuerySet":
        """Select specific fields to retrieve."""
        self.selected_fields = fields
        return self

    def where(self, **filters) -> "RedisQuerySet":
        """Add WHERE conditions to the query."""
        self.filters.update(filters)
        return self

    def order_by(self, field_name: str) -> "RedisQuerySet":
        """Order results by field (use '-field' for descending)."""
        if field_name.startswith("-"):
            self.order_descending = True
            self.order_field = field_name[1:]
        else:
            self.order_descending = False
            self.order_field = field_name
        return self

    def limit(self, count: int) -> "RedisQuerySet":
        """Limit number of results."""
        self.limit_count = count
        return self

    def offset(self, count: int) -> "RedisQuerySet":
        """Skip first N results."""
        self.offset_count = count
        return self

    def prefetch_related(self, *fields: str) -> "RedisQuerySet":
        """Prefetch related fields (for future use)."""
        self._prefetch_related.update(fields)
        return self

    async def all(self) -> List[T]:
        """
        Execute query and return all matching results.
        Optimized for performance with index usage and batch operations.
        """
        redis_connection = self.model_class.get_redis()
        key_prefix = self.model_class.get_key_prefix()

        # Separate simple equality filters from complex ones
        simple_eq_filters = {}
        range_filters = {}
        complex_filters = {}
        
        for key, val in self.filters.items():
            field, operator = split_filter_key(key)
            
            if operator == "eq":
                simple_eq_filters[field] = val
            elif operator in ("lt", "lte", "gt", "gte"):
                if field not in range_filters:
                    range_filters[field] = {}
                range_filters[field][operator] = val
            else:
                complex_filters[key] = val

        # Use Redis indexes for efficient filtering
        matched_ids = await self._get_indexed_ids(
            key_prefix, simple_eq_filters, range_filters
        )
        
        if not matched_ids:
            return []

        # Batch retrieve objects
        matched_objects = await self._batch_get_objects(matched_ids)
        
        # Apply complex filters in memory
        if complex_filters:
            matched_objects = self._apply_complex_filters(matched_objects, complex_filters)

        # Apply ordering
        if self.order_field:
            matched_objects = self._apply_ordering(matched_objects)

        # Apply offset and limit
        return self._apply_pagination(matched_objects)

    async def _get_indexed_ids(self, key_prefix: str, simple_filters: Dict, range_filters: Dict) -> List[str]:
        """Get IDs using Redis indexes efficiently."""
        redis_connection = self.model_class.get_redis()
        
        if not simple_filters and not range_filters:
            return await self.model_class._resolve_ids(None)

        id_sets = []
        
        # Handle simple equality filters
        for field, value in simple_filters.items():
            index_key = f"{key_prefix}:by_{field}:{value}"
            members = await redis_connection.smembers(index_key)
            id_set = {
                member.decode() if isinstance(member, bytes) else member
                for member in members
            }
            id_sets.append(id_set)

        # Handle range filters using sorted sets
        for field, conditions in range_filters.items():
            z_key = f"{key_prefix}:z_by_{field}"
            
            # Determine range bounds
            min_val = float('-inf')
            max_val = float('inf')
            
            for op, val in conditions.items():
                if op == "gte":
                    min_val = max(min_val, float(val))
                elif op == "gt":
                    min_val = max(min_val, float(val) + 0.000001)  # Small epsilon for exclusion
                elif op == "lte":
                    max_val = min(max_val, float(val))
                elif op == "lt":
                    max_val = min(max_val, float(val) - 0.000001)
            
            # Query sorted set
            members = await redis_connection.zrangebyscore(z_key, min_val, max_val)
            id_set = {
                member.decode() if isinstance(member, bytes) else member
                for member in members
            }
            id_sets.append(id_set)

        # Intersect all sets
        if id_sets:
            return list(set.intersection(*id_sets))
        return []

    async def _batch_get_objects(self, ids: List[str], batch_size: int = 1000) -> List[T]:
        """Retrieve objects in batches for better performance."""
        if not ids:
            return []
        
        objects = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_objects = await self.model_class.get(
                ids=batch_ids, 
                fields=self.selected_fields
            )
            
            # Filter out None objects
            objects.extend([obj for obj in batch_objects if obj is not None])
        
        return objects

    def _apply_complex_filters(self, objects: List[T], complex_filters: Dict) -> List[T]:
        """Apply complex filters in memory."""
        filtered_objects = []
        
        for obj in objects:
            passes_all_filters = True
            
            for filter_key, expected_value in complex_filters.items():
                field, operator_name = split_filter_key(filter_key)
                actual_value = getattr(obj, field, None)
                
                operator_func = FILTER_OPERATORS.get(operator_name)
                if operator_func is None:
                    raise ValueError(f"Unsupported operator: {operator_name}")
                
                if not operator_func(actual_value, expected_value):
                    passes_all_filters = False
                    break
            
            if passes_all_filters:
                filtered_objects.append(obj)
        
        return filtered_objects

    def _apply_ordering(self, objects: List[T]) -> List[T]:
        """Apply ordering to objects."""
        def get_sort_key(obj):
            val = getattr(obj, self.order_field, None)
            if val is None and hasattr(obj, "_meta"):
                val = getattr(obj._meta, self.order_field, None)
            if isinstance(val, str):
                return natural_sort_key(val)
            return val if val is not None else 0
        
        return sorted(objects, key=get_sort_key, reverse=self.order_descending)

    def _apply_pagination(self, objects: List[T]) -> List[T]:
        """Apply offset and limit to objects."""
        start_index = self.offset_count
        end_index = start_index + self.limit_count if self.limit_count is not None else None
        return objects[start_index:end_index]

    async def first(self) -> Optional[T]:
        """Get first matching object."""
        self.limit(1)
        results = await self.all()
        return results[0] if results else None

    async def last(self) -> Optional[T]:
        """Get last matching object."""
        if not self.order_field:
            raise ValueError("last() requires order_by() to be set")
        
        # Reverse the ordering
        original_desc = self.order_descending
        self.order_descending = not self.order_descending
        
        self.limit(1)
        results = await self.all()
        
        # Restore original ordering
        self.order_descending = original_desc
        
        return results[0] if results else None

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

    async def exists(self) -> bool:
        """Check if any matching objects exist."""
        count = await self.count()
        return count > 0

    async def values(self, *fields: str) -> List[Dict[str, Any]]:
        """Return dictionaries with specified field values."""
        if not fields:
            raise ValueError("values() requires at least one field")
        
        self.selected_fields = list(fields)
        objects = await self.all()
        
        return [
            {field: getattr(obj, field, None) for field in fields}
            for obj in objects
        ]

    async def values_list(self, *fields: str, flat: bool = False) -> List:
        """Return tuples/lists with specified field values."""
        if not fields:
            raise ValueError("values_list() requires at least one field")
        
        if flat and len(fields) != 1:
            raise ValueError("flat=True requires exactly one field")
        
        self.selected_fields = list(fields)
        objects = await self.all()
        
        if flat:
            return [getattr(obj, fields[0], None) for obj in objects]
        
        return [
            tuple(getattr(obj, field, None) for field in fields)
            for obj in objects
        ]

    def __aiter__(self):
        """Async iterator support."""
        return self._async_iterator()

    async def _async_iterator(self):
        """Async iterator implementation."""
        objects = await self.all()
        for obj in objects:
            yield obj

    def __repr__(self):
        return f"<RedisQuerySet: {self.model_class.__name__}>"