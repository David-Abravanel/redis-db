import json
import uuid
import time
import re
import pydantic
from typing import Any, Optional, ClassVar, Type, TypeVar, get_type_hints, Tuple, Dict

import redis.asyncio as redis_async
from pydantic import BaseModel, PrivateAttr, ValidationError, create_model

from .engine import RedisDB

T = TypeVar('T', bound='RedisBaseModel')


class MetaFields(BaseModel):
    exp: Optional[int] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    exp: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'


class RedisBaseModel(BaseModel):
    id: Optional[str] = None
    _meta: MetaFields = PrivateAttr(default_factory=MetaFields)
    _include_meta_for_repr: bool = PrivateAttr(default=False)

    _redis: ClassVar[Optional[redis_async.Redis]] = None

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
        if RedisDB._instance is None:
            RedisDB._instance = RedisDB()
        RedisDB._instance.add_table(cls)

    @classmethod
    def get_redis(cls) -> redis_async.Redis:
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
        model_data = self.model_dump(exclude_none=True)
        meta_data = self._meta.model_dump(exclude_none=True)

        combined = {**model_data, **meta_data}

        serialized = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in combined.items()
        }
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
        type_hints = get_type_hints(cls)

        model_data = {}
        meta_data = {}

        for k, v in decoded.items():
            if k in ("exp", "created_at", "updated_at"):
                meta_data[k] = int(v)
            else:
                if type_hints.get(k) in (dict, list):
                    try:
                        model_data[k] = json.loads(v)
                    except json.JSONDecodeError:
                        model_data[k] = v
                else:
                    model_data[k] = v

        for field in MetaFields.__fields__:
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
    async def get(cls: Type[T], id: str, include_meta: bool = False) -> Optional[T]:
        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        raw_data = await redis_conn.hgetall(key)
        if not raw_data:
            return None
        model_data, meta = cls._parse_redis_data(raw_data)
        if include_meta:
            exp = await redis_conn.ttl(key)
            meta.exp = exp if exp >= 0 else None
        instance = cls.model_validate(model_data)
        instance._meta = meta
        instance._include_meta_for_repr = include_meta
        return instance

    @classmethod
    async def delete(cls: Type[T], id: str) -> bool:
        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        res = await redis_conn.delete(key)
        return res > 0

    @classmethod
    async def update(
        cls: Type[T],
        id: str,
        return_updated: bool = False,
        include_meta: bool = False,
        exp: Optional[int] = None,
        **fields
    ) -> Optional[T]:
        if not fields:
            return None

        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"

        validated_fields = cls._validate_partial_fields(fields)

        if exp is not None:
            validated_fields['exp'] = exp

        data = cls._prepare_update_data(validated_fields)

        await redis_conn.hmset(key, data)
        if exp is not None:
            await redis_conn.expire(key, int(exp))

        if return_updated:
            return await cls.get(id, include_meta=include_meta)
        return None
    
    @classmethod
    async def create(cls: Type[T], obj: T | dict, *, exp: Optional[int] = None) -> T:
        """
        Create and save a new instance to Redis.
        Accepts either a dict or a model instance.
        Automatically validates the input and handles meta.
        """
        if isinstance(obj, dict):
            instance = cls.model_validate(obj)
        elif isinstance(obj, cls):
            instance = obj
        else:
            raise TypeError(f"create() expected dict or {cls.__name__} instance, got {type(obj)}")

        # Set expiration if passed
        if exp is not None:
            instance._meta.exp = exp

        await instance.save()
        return instance


    def __repr__(self):
        fields = self.model_dump(exclude_none=True)
        if self._include_meta_for_repr:
            fields["_meta"] = self._meta.model_dump(exclude_none=True)
        json_str = json.dumps(fields, indent=2, ensure_ascii=False)
        return f"{self.__class__.__name__}({json_str})"

    def __str__(self):
        return self.__repr__()
