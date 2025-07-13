import json
import uuid
import time
import re
from typing import Optional, ClassVar, Type, TypeVar, get_type_hints, Tuple, Dict, Any

import redis.asyncio as redis_async
from pydantic import BaseModel, PrivateAttr, ValidationError

from .engine import RedisDB

T = TypeVar('T', bound='RedisBaseModel')


class MetaFields(BaseModel):
    """
    Metadata fields for Redis entries: expiration, creation and update timestamps.
    """
    exp: Optional[int] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'


class RedisBaseModel(BaseModel):
    """
    Base Pydantic model for Redis hash storage with async Redis client.

    Features:
    - Auto-generate ID if missing
    - Manage meta data (exp, timestamps) internally via PrivateAttr
    - JSON serialize dicts and lists transparently on save/update
    - Async CRUD operations with Redis (save/get/delete/update)
    - Validation of partial updates (compatible with Pydantic 1.x)
    - Allows meta defaults from inner Meta class per model subclass
    """

    id: Optional[str] = None
    _meta: MetaFields = PrivateAttr(default_factory=MetaFields)

    _redis: ClassVar[Optional[redis_async.Redis]] = None

    class Meta:
        # Optional Redis key prefix override
        key_prefix: ClassVar[Optional[str]] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._apply_meta_defaults_from_model()

    def _apply_meta_defaults_from_model(self):
        """
        Apply meta defaults from the subclass Meta class to the instance's _meta
        if the meta fields are not already set.
        """
        meta_cls = getattr(self.__class__, "Meta", None)
        if not meta_cls:
            return
        for field in MetaFields.__fields__:
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
        """Return Redis connection instance."""
        if cls._redis:
            return cls._redis
        if RedisDB._instance is None:
            raise RuntimeError("RedisDB not initialized. Create/start RedisDB first.")
        cls._redis = RedisDB._instance.get_redis()
        return cls._redis

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """Convert CamelCase string to snake_case."""
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @classmethod
    def get_key_prefix(cls) -> str:
        """Return Redis key prefix for the model, default is snake_case class name."""
        key_prefix = getattr(cls.Meta, "key_prefix", None)
        return key_prefix or cls._camel_to_snake(cls.__name__)

    def _serialize_data(self) -> Tuple[str, Dict[str, str]]:
        """
        Prepare model data and meta for Redis storage as dict[str, str].
        Serializes dict/list as JSON strings.
        Sets timestamps and id if needed.
        """
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
        """
        Serialize update fields and meta fields (exp, timestamps).
        """
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
        """
        Decode raw redis bytes and deserialize JSON fields.
        Separate model data from meta data.
        """
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

        return model_data, MetaFields(**meta_data)

    @classmethod
    def _validate_partial_fields(cls, fields: dict) -> dict:
        """
        Validate only the fields being updated against the model fields.
        Uses Pydantic's parse_obj for partial validation (Pydantic 1.x compatible).
        Raises ValueError if validation fails.
        """
        try:
            # parse_obj allows partial data, no required fields needed here
            validated_instance = cls.parse_obj(fields)
            # Return only the fields present in input
            return {k: getattr(validated_instance, k) for k in fields.keys()}
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

    async def save(self) -> None:
        """
        Save the current instance to Redis.
        """
        redis_conn = self.get_redis()
        key, data = self._serialize_data()
        await redis_conn.hmset(key, data)
        if self._meta.exp:
            await redis_conn.expire(key, self._meta.exp)

    @classmethod
    async def get(cls: Type[T], id: str, include_meta: bool = False) -> Optional[T]:
        """
        Get a model instance by ID from Redis.
        If include_meta=True, populate the _meta PrivateAttr.
        """
        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        raw_data = await redis_conn.hgetall(key)
        if not raw_data:
            return None
        model_data, meta = cls._parse_redis_data(raw_data)
        instance = cls.parse_obj(model_data)
        if include_meta:
            instance._meta = meta
        return instance

    @classmethod
    async def delete(cls: Type[T], id: str) -> bool:
        """
        Delete a model instance from Redis by ID.
        Returns True if the key was deleted.
        """
        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        res = await redis_conn.delete(key)
        return res > 0

    @classmethod
    async def update(cls: Type[T], id: str, return_updated: bool = False, exp: int = None, **fields) -> Optional[T]:
        """
        Partially update fields of a model in Redis by ID.
        Validates updated fields.
        Optionally returns the updated model instance.
        """
        if not fields:
            return None

        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"

        # Validate fields before updating
        validated_fields = cls._validate_partial_fields(fields)
        data = cls._prepare_update_data(validated_fields)

        await redis_conn.hmset(key, data)
        if exp:
            await redis_conn.expire(key, int(exp))

        if return_updated:
            return await cls.get(id)
        return None

    def __repr__(self):
        """
        String representation excluding None fields.
        """
        fields = self.model_dump(exclude_none=True)
        fields_str = ', '.join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fields_str})"
