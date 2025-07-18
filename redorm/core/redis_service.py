import logging
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from typing import Any, Optional


class RedisMessenger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if args or kwargs:
            cls._instance._init_connection(*args, **kwargs)
        return cls._instance

    def _init_connection(self, host: str, port: int, db: int, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Connecting to Redis at {host}:{port}, DB: {db}")
        self.redis = Redis(host=host, port=int(port), db=db, decode_responses=True)

    async def close(self):
        if self.redis:
            await self.redis.close()
            self.redis = None
            self.logger.info("Redis connection closed.")

    # רגיל
    async def set(self, key: str, value: Any, ex: Optional[int] = None):
        return await self.redis.set(key, value, ex=ex)

    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)

    async def delete(self, key: str):
        return await self.redis.delete(key)

    # PIPE
    def get_pipe(self) -> Pipeline:
        """Return a new Redis pipeline (synchronous)."""
        return self.redis.pipeline()

    def pipe_set(self, pipe: Pipeline, key: str, value: Any, ex: Optional[int] = None) -> Pipeline:
        if not isinstance(value, bytes):
            raise TypeError(f"Error: Invalid data -> Expected bytes for Redis set() gat: {type(value)}\nvalue:\n    {value}")
        pipe.set(key, value, ex=ex)
        return pipe

    def pipe_get(self, pipe: Pipeline, key: str) -> Pipeline:
        pipe.get(key)
        return pipe

    def pipe_delete(self, pipe: Pipeline, key: str) -> Pipeline:
        pipe.delete(key)
        return pipe

    async def pipe_execute(self, pipe: Pipeline):
        """Execute all commands queued in the pipeline."""
        return await pipe.execute()    
        # def push_to_queue(self, queue_name: str, message: str):
    #     self.redis.lpush(queue_name, message)

    # def append_to_list(self, list_name: str, value: str):
    #     self.redis.rpush(list_name, value)


    # def incr_field(self, hash_name: str, field: str, amount: int = 1):
    #     if self.redis.exists(hash_name):
    #         self.redis.hincrby(hash_name, field, amount)
    
    
    # def get_list(self, list_name: str) -> list[str]:
    #     return self.redis.lrange(list_name, 0, -1)

    # def publish(self, channel: str, message: str):
    #     self.redis.publish(channel, message)
    #     self.logger.info(f"✅ Published to channel '{channel}': {message}")

    # def listen_to_queue(self, queue_name: str, callback, timeout: int = 5):
    #     self.logger.info(f"📡 Listening to queue '{queue_name}'...")
    #     while True:
    #         try:
    #             result = self.redis.brpop(queue_name, timeout=timeout)
    #             if result:
    #                 _, message = result
    #                 self.logger.info(f"📤 Dequeued from '{queue_name}': {message}")
    #                 callback(message)
    #         except Exception as e:
    #             self.logger.info(f"⚠️ Error while listening to queue: {e}")

    # def get(self, key: str) -> Optional[str]:
    #     value = self.redis.get(key)
    #     self.logger.info(f"📥 Get key '{key}'")
    #     return value
