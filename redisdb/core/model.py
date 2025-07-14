# Optimized redisdb/core/model.py
import asyncio
import re
import json
import orjson
import uuid
import time
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from typing import Any, List, Optional, ClassVar, Type, TypeVar, get_type_hints, Tuple, Dict, Union
from pydantic import BaseModel, PrivateAttr, ValidationError, create_model

from .query import RedisQuerySet, natural_sort_key
from .engine import RedisDB

T = TypeVar('T', bound='RedisBaseModel')
UUID_REGEX = re.compile(r'^[0-9a-fA-F\-]{36}$')

class MetaFields(BaseModel):
    exp: Optional[int] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    indexable_fields: Optional[set] = None

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'


class RedisBaseModel(BaseModel):
    id: Optional[str] = None
    _meta: MetaFields = PrivateAttr(default_factory=MetaFields)
    _include_meta_for_repr: bool = PrivateAttr(default=False)

    _redis: ClassVar[Optional[Redis]] = None
    _type_hints_cache: ClassVar[Dict[str, Any]] = {}

    class Meta:
        key_prefix: ClassVar[Optional[str]] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._apply_meta_defaults_from_model()

    def _apply_meta_defaults_from_model(self):
        meta_cls = getattr(self.__class__, "Meta", None)
        if not meta_cls:
            return
        for field in MetaFields.model_fields:
            if getattr(self._meta, field, None) is None:
                value = getattr(meta_cls, field, None)
                if value is not None:
                    setattr(self._meta, field, value)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Cache type hints for performance
        cls._type_hints_cache = get_type_hints(cls)
        
        if RedisDB._instance is None:
            RedisDB._instance = RedisDB()
        RedisDB._instance.add_table(cls)

    @classmethod
    def get_redis(cls) -> Redis:
        if cls._redis:
            return cls._redis
        if RedisDB._instance is None:
            raise RuntimeError("RedisDB not initialized. Create/start RedisDB first.")
        cls._redis = RedisDB._instance.get_redis()
        return cls._redis

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @classmethod
    def get_key_prefix(cls) -> str:
        key_prefix = getattr(cls.Meta, "key_prefix", None)
        return key_prefix or cls._camel_to_snake(cls.__name__)

    def _serialize_data(self) -> Tuple[str, Dict[str, str]]:
        now = int(time.time())
        self.id = self.id or str(uuid.uuid4())
        self._meta.created_at = self._meta.created_at or now
        self._meta.updated_at = now

        key = f"{self.get_key_prefix()}:{self.id}"
        model_data = self.model_dump(exclude_none=True, exclude_defaults=True)
        meta_data = self._meta.model_dump(exclude_none=True)
        combined = {**model_data, **meta_data}
        
        # Optimized serialization - avoid encoding simple types
        serialized = {}
        for k, v in combined.items():
            if isinstance(v, (dict, list)):
                serialized[k] = orjson.dumps(v).decode("utf-8")
            else:
                serialized[k] = str(v)

        return key, serialized

    @classmethod
    def _prepare_update_data(cls, fields: dict) -> Dict[str, str]:
        now = int(time.time())
        data = {}
        meta_fields = {}

        for k, v in fields.items():
            if k in ("exp", "created_at", "updated_at"):
                meta_fields[k] = v
            else:
                data[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

        meta_fields['updated_at'] = now
        data.update({k: str(v) for k, v in meta_fields.items()})
        return data

    @classmethod
    def _parse_redis_data(cls, raw_data: dict) -> Tuple[dict, MetaFields]:
        decoded = {k.decode(): v.decode() for k, v in raw_data.items()}
        type_hints = cls._type_hints_cache  # Use cached type hints
        model_data = {}
        meta_data = {}

        for k, v in decoded.items():
            if k in ("exp", "created_at", "updated_at"):
                meta_data[k] = int(v)
            else:
                if type_hints.get(k) in (dict, list):
                    try:
                        model_data[k] = orjson.loads(v)  # Use orjson for faster parsing
                    except (orjson.JSONDecodeError, ValueError):
                        model_data[k] = v
                else:
                    model_data[k] = v

        for field in MetaFields.model_fields:
            if field not in meta_data:
                meta_data[field] = None

        return model_data, MetaFields(**meta_data)

    @classmethod
    def _validate_partial_fields(cls, fields: dict) -> dict:
        optional_fields = {k: (Optional[Any], None) for k in cls.model_fields.keys()}
        UpdateModel = create_model('UpdateModel', **optional_fields)
        try:
            validated_instance = UpdateModel(**fields)
            return {k: getattr(validated_instance, k) for k in fields.keys()}
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

    async def save(self) -> None:
        redis_conn = self.get_redis()
        key, data = self._serialize_data()
        await redis_conn.hmset(key, data)
        if self._meta.exp:
            await redis_conn.expire(key, self._meta.exp)

    @classmethod
    async def _resolve_ids(cls, ids: Union[str, List[str], None]) -> List[str]:
        redis_conn = cls.get_redis()

        if ids is None:
            pattern = f"{cls.get_key_prefix()}:*"
            resolved_ids = []
            async for key in redis_conn.scan_iter(match=pattern):
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) != 2:
                    continue
                candidate_id = parts[1]

                if UUID_REGEX.match(candidate_id):
                    resolved_ids.append(candidate_id)
        elif isinstance(ids, str):
            resolved_ids = [ids]
        else:
            resolved_ids = ids

        return resolved_ids

    @classmethod
    def _get_index_key(cls, field: str, value: Any) -> str:
        return f"{cls.get_key_prefix()}:by_{field}:{value}"

    # @classmethod
    # async def _batch_index_fields(cls, instances: List[T]) -> None:
    #     """Optimized batch indexing for multiple instances"""
    #     redis = cls.get_redis()
    #     prefix = cls.get_key_prefix()
        
    #     indexable_fields = getattr(getattr(cls, "Meta", None), "indexable_fields", None)
    #     type_hints = cls._type_hints_cache

    #     async with redis.pipeline(transaction=False) as pipe:
    #         for instance in instances:
    #             for field, typ in type_hints.items():
    #                 if field.startswith("_") or field == "id":
    #                     continue
                    
    #                 if indexable_fields is not None and field not in indexable_fields:
    #                     continue

    #                 if typ not in (str, int, float, bool):
    #                     continue
                    
    #                 value = getattr(instance, field, None)
    #                 if value is None:
    #                     continue

    #                 if isinstance(value, (str, int, float, bool)):
    #                     set_key = f"{prefix}:by_{field}:{value}"
    #                     pipe.sadd(set_key, instance.id)

    #                 if isinstance(value, (int, float)):
    #                     z_key = f"{prefix}:z_by_{field}"
    #                     pipe.zadd(z_key, {instance.id: float(value)})

    #         await pipe.execute()

    @classmethod
    async def _index_fields(cls, instance: T, pipe: Optional[Pipeline] = None) -> None:
        redis = cls.get_redis()
        pipe = pipe or redis.pipeline()

        prefix = cls.get_key_prefix()
        
        indexable_fields = getattr(getattr(cls, "Meta", None), "indexable_fields", None)
        type_hints = cls._type_hints_cache

        for field, typ in type_hints.items():
            if field.startswith("_") or field == "id":
                continue
            
            if indexable_fields is not None and field not in indexable_fields:
                continue

            if typ not in (str, int, float, bool):
                continue
            
            value = getattr(instance, field, None)
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                set_key = f"{prefix}:by_{field}:{value}"
                pipe.sadd(set_key, instance.id)

            if isinstance(value, (int, float)):
                z_key = f"{prefix}:z_by_{field}"
                pipe.zadd(z_key, {instance.id: float(value)})

        if pipe is not None and isinstance(pipe, Pipeline):
            await pipe.execute()

    @classmethod
    async def _remove_from_indexes(cls, instance: T) -> None:
        redis_conn = cls.get_redis()
        simple_types = (str, int, float, bool)
        type_hints = cls._type_hints_cache
        
        async with redis_conn.pipeline(transaction=True) as pipe:
            for field_name, field_type in type_hints.items():
                value = getattr(instance, field_name, None)
                if value is not None and isinstance(value, simple_types):
                    index_key = cls._get_index_key(field_name, value)
                    pipe.srem(index_key, instance.id)
            await pipe.execute()

    @classmethod
    def _cast_partial_fields(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        type_hints = cls._type_hints_cache
        parsed = {}

        for field, val in raw_data.items():
            if val is None:
                parsed[field] = None
                continue

            typ = type_hints.get(field)
            try:
                if typ == int:
                    parsed[field] = int(val)
                elif typ == float:
                    parsed[field] = float(val)
                elif typ == bool:
                    parsed[field] = val.lower() in ("true", "1", "yes")
                elif typ in (dict, list):
                    parsed[field] = orjson.loads(val)  # Use orjson
                else:
                    parsed[field] = val
            except Exception:
                parsed[field] = val

        return parsed

    @classmethod
    def select(cls, fields: Optional[List[str]] = None) -> RedisQuerySet[T]:
        return RedisQuerySet(cls, fields=fields)

    @classmethod
    async def update(
        cls: Type[T],
        ids: Union[str, List[str], None] = None,
        return_updated: bool = False,
        include_meta: bool = False,
        exp: Optional[int] = None,
        **fields
    ) -> Union[Optional[T], List[Optional[T]], None]:
        if not fields:
            return None

        redis_conn = cls.get_redis()
        ids = await cls._resolve_ids(ids)
        full_keys = [f"{cls.get_key_prefix()}:{id_}" for id_ in ids]

        current_instances = await cls.get(ids)

        validated_fields = cls._validate_partial_fields(fields)
        if exp is not None:
            validated_fields['exp'] = exp
        data = cls._prepare_update_data(validated_fields)

        async with redis_conn.pipeline(transaction=True) as pipe:
            for i, key in enumerate(full_keys):
                pipe.hmset(key, data)
                if exp is not None:
                    pipe.expire(key, int(exp))

                current = current_instances[i]
                if current:
                    await cls._remove_from_indexes(current, pipe=pipe)
                    model_data = current.model_dump()
                    model_data.update(fields)
                    updated_instance = cls.model_validate(model_data)
                    updated_instance.id = current.id
                    await cls._index_fields(updated_instance, pipe=pipe)

            await pipe.execute()

        if return_updated:
            return await cls.get(ids, include_meta=include_meta)
        return None

    @classmethod
    async def get(
        cls: Type[T],
        ids: Union[str, List[str], None] = None,
        include_meta: bool = False,
        fields: Optional[List[str]] = None
    ) -> Union[Optional[T], List[Optional[T]]]:
        redis_conn = cls.get_redis()
        ids = await cls._resolve_ids(ids)

        # Batch Redis operations
        async with redis_conn.pipeline(transaction=False) as pipe:
            for id_ in ids:
                key = f"{cls.get_key_prefix()}:{id_}"
                if fields is None:
                    pipe.hgetall(key)
                else:
                    pipe.hmget(key, *fields)
                if include_meta:
                    pipe.ttl(key)
            results = await pipe.execute()

        output = []
        step = 2 if include_meta else 1
        for i, id_ in enumerate(ids):
            raw_data = results[i * step]
            if not raw_data:
                output.append(None)
                continue

            if fields is None:
                model_data, meta = cls._parse_redis_data(raw_data)
            else:
                field_values = raw_data
                raw_dict = {
                    field: (val.decode() if isinstance(val, bytes) else val)
                    for field, val in zip(fields, field_values)
                }
                model_data = cls._cast_partial_fields(raw_dict)
                meta = MetaFields()

            if include_meta:
                ttl_val = results[i * step + 1]
                meta.exp = ttl_val if isinstance(ttl_val, int) and ttl_val >= 0 else None
                
            if fields is not None:
                instance = cls.model_construct(**model_data)
            else:
                instance = cls.model_validate(model_data)
            instance.id = id_
            instance._meta = meta
            instance._include_meta_for_repr = include_meta
            output.append(instance)

        return output[0] if len(output) == 1 else output

    @classmethod
    async def delete(cls: Type[T], ids: Union[str, List[str], None] = None) -> Union[bool, List[bool]]:
        redis_conn = cls.get_redis()
        ids = await cls._resolve_ids(ids)

        instances = await cls.get(ids)
        full_keys = [f"{cls.get_key_prefix()}:{id_}" for id_ in ids]

        async with redis_conn.pipeline(transaction=True) as pipe:
            for inst in instances:
                if inst:
                    await cls._remove_from_indexes(inst, pipe=pipe)
            pipe.delete(*full_keys)
            results = await pipe.execute()

        deleted_count = results[-1]
        if len(ids) == 1:
            return deleted_count == 1
        else:
            return [i < deleted_count for i in range(len(ids))]

    @classmethod
    async def create(
        cls: Type[T],
        items: Union[T, dict, List[Union[T, dict]]],
        *,
        exp: Optional[int] = None,
        batch_size: int = 1000  # Add batch processing
    ) -> Union[T, List[T]]:
        if not isinstance(items, list):
            items = [items]
            return_single = True
        else:
            return_single = False

        # Process in batches for very large datasets
        all_instances = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_instances = await cls._create_batch(batch, exp=exp)
            all_instances.extend(batch_instances)

        return all_instances[0] if return_single else all_instances

    @classmethod
    async def _create_batch(
        cls: Type[T],
        items: List[Union[T, dict]],
        *,
        exp: Optional[int] = None
    ) -> List[T]:
        # Validation
        instances: List[T] = []
        for item in items:
            instance = cls.model_validate(item) if isinstance(item, dict) else item
            if exp is not None:
                instance._meta.exp = exp
            instances.append(instance)

        # Batch serialization
        serialized_items = [inst._serialize_data() for inst in instances]

        # Batch Redis operations
        redis_conn = cls.get_redis()
        async with redis_conn.pipeline(transaction=False) as pipe:
            for inst, (key, data) in zip(instances, serialized_items):
                pipe.hmset(key, data)
                if inst._meta.exp:
                    pipe.expire(key, inst._meta.exp)
            await pipe.execute()

        # Index fields after creation
        await asyncio.gather(*[cls._index_fields(inst) for inst in instances])
        return instances

    async def save(self) -> None:
        redis_conn = self.get_redis()
        key, data = self._serialize_data()
        await redis_conn.hmset(key, data)
        if self._meta.exp:
            await redis_conn.expire(key, self._meta.exp)
        await self.__class__._index_fields(self)

    def __repr__(self):
        fields = self.model_dump(exclude_none=True)
        if self._include_meta_for_repr:
            meta_fields = self._meta.model_dump(exclude_none=True)
            fields["_meta"] = meta_fields
        json_str = json.dumps(fields, indent=2, ensure_ascii=False)
        return f"{self.__class__.__name__}: {json_str}"

    def __str__(self):
        return self.__repr__()