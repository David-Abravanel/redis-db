# redisorm/core/engine.py
import os
import platform
import subprocess
import asyncio
import socket
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional, Dict, Type
import redis.asyncio as redis_async


class RedisDB:
    _instance: Optional['RedisDB'] = None
    REDIS_DIR = Path('/') / 'redis_local'
    REDIS_SERVER_PATH = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisDB, cls).__new__(cls)
        return cls._instance

    def __init__(self, host=None, port=None, db=0, password=None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.host = host or 'localhost'
        self.port = port or 6379
        self.db = db
        self.password = password
        self._redis: Optional[redis_async.Redis] = None
        self._process = None
        self.tables: Dict[str, Type] = {}
        self._initialized = True

    async def start(self):
        if await self._test_redis_connection(self.host, self.port):
            self._redis = redis_async.Redis(host=self.host, port=self.port, db=self.db, password=self.password)
            return

        print("Redis server not running, trying to download, install and start locally asynchronously...")

        self.REDIS_DIR.mkdir(parents=True, exist_ok=True)

        redis_path = shutil.which('redis-server')
        if redis_path is None:
            await asyncio.get_event_loop().run_in_executor(None, self._download_and_install_redis)

        self.REDIS_SERVER_PATH = shutil.which('redis-server') or str(self.REDIS_DIR / 'redis-server')
        if not self.REDIS_SERVER_PATH or not Path(self.REDIS_SERVER_PATH).exists():
            raise RuntimeError("redis-server executable not found after installation.")

        self._process = subprocess.Popen([self.REDIS_SERVER_PATH, '--port', str(self.port)])
        print(f"Started redis-server with PID {self._process.pid}")

        for _ in range(10):
            if await self._test_redis_connection('localhost', self.port):
                self._redis = redis_async.Redis(host='localhost', port=self.port, db=self.db)
                print("Redis server started successfully.")
                return
            await asyncio.sleep(1)

        self._process.terminate()
        raise RuntimeError("Failed to start Redis server")

    def _download_and_install_redis(self):
        system = platform.system()
        arch = platform.machine()

        if system == 'Linux':
            url = 'http://download.redis.io/releases/redis-7.0.12.tar.gz'
            tar_path = self.REDIS_DIR / 'redis.tar.gz'

            print(f"Downloading Redis from {url} ...")
            urlretrieve(url, tar_path)

            print("Extracting Redis...")
            subprocess.run(['tar', '-xzf', str(tar_path), '-C', str(self.REDIS_DIR)], check=True)

            redis_source_dir = next(self.REDIS_DIR.glob('redis-*'))

            print("Compiling Redis...")
            subprocess.run(['make'], cwd=str(redis_source_dir), check=True)

            redis_server_bin = redis_source_dir / 'src' / 'redis-server'
            target_bin = self.REDIS_DIR / 'redis-server'
            shutil.copy2(redis_server_bin, target_bin)
            os.chmod(target_bin, 0o755)

            print(f"Redis installed to {target_bin}")

        elif system == 'Windows':
            raise RuntimeError("Automatic Redis installation not supported on Windows yet. Please install manually.")
        else:
            raise RuntimeError(f"Unsupported OS: {system}")

    async def _test_redis_connection(self, host, port) -> bool:
        loop = asyncio.get_event_loop()
        try:
            fut = loop.run_in_executor(None, socket.create_connection, (host, port), 1)
            s = await asyncio.wait_for(fut, timeout=1)
            s.close()
            return True
        except Exception:
            return False

    def get_redis(self):
        if self._redis is None:
            raise RuntimeError("Redis connection is not initialized")
        return self._redis

    def add_table(self, model_cls):
        key_prefix = model_cls.get_key_prefix()
        model_cls._redis = self.get_redis()
        self.tables[key_prefix] = model_cls
        setattr(self, model_cls.__name__, model_cls)

    def get_table(self, name: str):
        return self.tables.get(name)

    def __getattr__(self, name: str):
        if name in self.tables:
            return self.tables[name]
        raise AttributeError(f"'RedisDB' object has no attribute '{name}'")

    def __dir__(self):
        return super().__dir__() + list(self.tables.keys())
