import json
import uuid
import time
import re
from typing import Optional, ClassVar, Type, TypeVar, get_type_hints

import redis.asyncio as redis_async
from pydantic import BaseModel

from .engine import RedisDB

T = TypeVar('T', bound='RedisBaseModel')


class RedisBaseModel(BaseModel):
    """
    Base model for Redis hash-based storage using Pydantic and redis.asyncio.
    
    Provides:
    - Auto ID generation
    - Timestamps (created_at, updated_at)
    - TTL support
    - JSON serialization for lists and dicts
    - Async save/get/delete methods
    """
    id: Optional[str] = None
    ttl: Optional[int] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    _redis: ClassVar[Optional[redis_async.Redis]] = None

    class Meta:
        # Optional key prefix override
        key_prefix: ClassVar[Optional[str]] = None

    def __init_subclass__(cls, **kwargs):
        """
        Called automatically when subclassed.
        Registers the model in the RedisDB singleton.
        """
        super().__init_subclass__(**kwargs)
        if RedisDB._instance is None:
            RedisDB._instance = RedisDB()
        RedisDB._instance.add_table(cls)

    @classmethod
    def get_redis(cls) -> redis_async.Redis:
        """
        Get the shared Redis connection from RedisDB.
        Raises an error if RedisDB is not initialized.
        """
        if cls._redis:
            return cls._redis
        if RedisDB._instance is None:
            raise RuntimeError("RedisDB not initialized. Please create and start it before using models.")
        cls._redis = RedisDB._instance.get_redis()
        return cls._redis

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """
        Convert CamelCase to snake_case.
        Used for automatic key prefix generation.
        """
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @classmethod
    def get_key_prefix(cls) -> str:
        """
        Return the Redis key prefix for the model.
        Defaults to snake_case of class name unless overridden in Meta.
        """
        return cls.Meta.key_prefix or cls._camel_to_snake(cls.__name__)

    def _prepare_data_for_storage(self) -> tuple[str, dict[str, str]]:
        """
        Prepare the Redis key and the field-value mapping for storage.
        - Generates ID and timestamps if needed.
        - Serializes dicts/lists to JSON.
        Returns: (key, {field: stringified_value})
        """
        now = int(time.time())
        self.id = self.id or str(uuid.uuid4())
        self.created_at = self.created_at or now
        self.updated_at = now

        key = f"{self.get_key_prefix()}:{self.id}"
        raw_data = self.dict(exclude_none=True)

        serialized_data = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in raw_data.items()
        }
        return key, serialized_data
    
    @classmethod
    def _prepare_update_data(cls, fields: dict) -> dict[str, str]:
        now = int(time.time())
        data = {}
        for k, v in fields.items():
            if isinstance(v, (dict, list)):
                data[k] = json.dumps(v)
            else:
                data[k] = str(v)
        data['updated_at'] = str(now)
        return data


    @classmethod
    def _parse_redis_data(cls, raw_data: dict) -> dict:
        """
        Deserialize raw data from Redis.
        - Decodes bytes to strings
        - Parses JSON fields if their expected type is dict or list
        """
        decoded = {k.decode(): v.decode() for k, v in raw_data.items()}
        type_hints = get_type_hints(cls)

        for k, v in decoded.items():
            if type_hints.get(k) in (dict, list):
                try:
                    decoded[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
        return decoded

    async def save(self) -> None:
        """
        Save the model instance to Redis.
        - Uses HMSET to store fields
        - Applies TTL if defined
        """
        redis_conn = self.get_redis()
        key, data = self._prepare_data_for_storage()
        await redis_conn.hmset(key, data)
        if self.ttl: 
            await redis_conn.expire(key, self.ttl)

    @classmethod
    async def get(cls: Type[T], id: str) -> Optional[T]:
        """
        Retrieve an instance by ID from Redis.
        - Returns parsed model instance or None if not found
        """
        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        raw_data = await redis_conn.hgetall(key)
        if not raw_data:
            return None
        return cls.model_validate(cls._parse_redis_data(raw_data))

    @classmethod
    async def delete(cls: Type[T], id: str) -> bool:
        """
        Delete a model instance from Redis by ID.
        Returns True if a key was deleted.
        """
        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        res = await redis_conn.delete(key)
        return res > 0

    @classmethod
    async def update(cls: Type[T], id: str, **fields) -> None:
        """
        Update specific fields of a model instance in Redis by ID, without loading the full object.
        - Updates the fields in Redis hash.
        - Updates the `updated_at` timestamp automatically.
        - Serializes dict/list fields as JSON strings.
        
        Args:
            id: The Redis key ID of the object to update.
            **fields: Field names and values to update.
        """
        if not fields:
            return  # Nothing to update

        redis_conn = cls.get_redis()
        key = f"{cls.get_key_prefix()}:{id}"
        data = cls._prepare_update_data(fields)
        await redis_conn.hmset(key, data)
