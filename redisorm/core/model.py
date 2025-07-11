# redisorm/core/model.py
from pydantic import BaseModel
from typing import Optional, ClassVar, Type, TypeVar
from .engine import RedisDB
import redis.asyncio as redis_async
import uuid
import time
import re

T = TypeVar('T', bound='AsyncRedisBaseModel')


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class AsyncRedisBaseModel(BaseModel):
    id: Optional[str] = None
    ttl: Optional[int] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    _redis: ClassVar[Optional[redis_async.Redis]] = None

    class Meta:
        key_prefix: ClassVar[Optional[str]] = None
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from .engine import RedisDB
        if RedisDB._instance is not None:
            RedisDB._instance.add_table(cls)

    @classmethod
    def get_redis(cls) -> redis_async.Redis:
        if cls._redis:
            return cls._redis
        if RedisDB._instance is None:
            raise RuntimeError("RedisDB not initialized. Please create and start it before using models.")
        cls._redis = RedisDB._instance.get_redis()
        return cls._redis

    @classmethod
    def get_key_prefix(cls) -> str:
        if cls.Meta.key_prefix:
            return cls.Meta.key_prefix
        return camel_to_snake(cls.__name__)

    async def save(self) -> None:
        redis_conn = self.get_redis()

        now = int(time.time())
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

        key = f"{self.get_key_prefix()}:{self.id}"
        data_str = {k: str(v) for k, v in self.dict().items() if v is not None}
        await redis_conn.hset(key, mapping=data_str)
 
        if self.ttl:
            await redis_conn.expire(key, self.ttl)

    @classmethod
    async def get(cls: Type[T], id: str) -> Optional[T]:
        redis_conn = cls.get_redis()

        key = f"{cls.get_key_prefix()}:{id}"
        data = await redis_conn.hgetall(key)
        if not data:
            return None
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        return cls.parse_obj(decoded)

    async def delete(self) -> None:
        redis_conn = self.get_redis()
        key = f"{self.get_key_prefix()}:{self.id}"
        await redis_conn.delete(key)
