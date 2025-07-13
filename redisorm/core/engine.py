# redisorm/core/engine.py
"""
RedisDB class for managing Redis connections and local server processes.
This module provides a singleton RedisDB class that handles:
- Connecting to a remote Redis server or starting a local one
- Downloading, building, and installing Redis server (Linux only)
- Maintaining a registry of model classes (tables) using this Redis connection
- Ensuring a single RedisDB instance is used throughout the application
"""

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
    """
    Singleton class managing a Redis connection and local Redis server process.
    
    Features:
    - Singleton pattern to ensure a single RedisDB instance
    - Supports connecting to a remote Redis server or launching a local one asynchronously
    - Automatic download, build, and installation of Redis server (Linux only)
    - Maintains registry of model classes (tables) using this Redis connection
    """
    _instance: Optional['RedisDB'] = None
    REDIS_DIR = Path('/') / 'redis_local'
    REDIS_SERVER_PATH = None

    def __new__(cls, *args, **kwargs):
        """
        Override new to implement singleton pattern.
        """
        if cls._instance is None:
            cls._instance = super(RedisDB, cls).__new__(cls)
        return cls._instance

    def __init__(self, host=None, port=None, db=0, password=None):
        """
        Initialize RedisDB instance.
        
        Args:
            host (str): Redis server hostname or IP. Defaults to 'localhost'.
            port (int): Redis server port. Defaults to 6379.
            db (int): Redis database number. Defaults to 0.
            password (str): Redis password, if set.
        """
        # Avoid re-initialization in singleton pattern
        if self.__dict__.get("_initialized", False):
            return

        self.host = host or 'localhost'
        self.port = port or 6379
        self.db = db
        self.password = password
        self._redis: Optional[redis_async.Redis] = None
        self._process = None  # For local Redis server process
        self.tables: Dict[str, Type] = {}  # Registered model classes keyed by prefix
        self._initialized = True

    async def start(self):
        """
        Start the Redis connection.
        - If Redis is reachable at given host/port, connects to it.
        - Otherwise, tries to download, build and start a local Redis server asynchronously.
        """
        if await self._test_redis_connection(self.host, self.port):
            self._redis = redis_async.Redis(host=self.host, port=self.port, db=self.db, password=self.password)
            return

        print("Redis server not running, trying to download, install and start locally asynchronously...")

        self.REDIS_DIR.mkdir(parents=True, exist_ok=True)

        redis_path = shutil.which('redis-server')
        if redis_path is None:
            # Blocking call in executor to avoid blocking event loop
            await asyncio.get_event_loop().run_in_executor(None, self._download_and_install_redis)

        self.REDIS_SERVER_PATH = shutil.which('redis-server') or str(self.REDIS_DIR / 'redis-server')
        if not self.REDIS_SERVER_PATH or not Path(self.REDIS_SERVER_PATH).exists():
            raise RuntimeError("redis-server executable not found after installation.")

        # Start Redis server subprocess
        self._process = subprocess.Popen([self.REDIS_SERVER_PATH, '--port', str(self.port)])
        print(f"Started redis-server with PID {self._process.pid}")

        # Wait up to ~10 seconds for server to become available
        for _ in range(10):
            if await self._test_redis_connection('localhost', self.port):
                self._redis = redis_async.Redis(host='localhost', port=self.port, db=self.db)
                print("Redis server started successfully.")
                return
            await asyncio.sleep(1)

        # Failed to start Redis within timeout, terminate process
        self._process.terminate()
        raise RuntimeError("Failed to start Redis server")

    def _download_and_install_redis(self):
        """
        Download, compile and install Redis locally (Linux only).
        Raises on unsupported OS or failure.
        """
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
        """
        Check if Redis server is reachable at host:port.
        
        Returns:
            bool: True if connection succeeds, False otherwise.
        """
        loop = asyncio.get_event_loop()
        try:
            fut = loop.run_in_executor(None, socket.create_connection, (host, port), 1)
            s = await asyncio.wait_for(fut, timeout=1)
            s.close()
            return True
        except Exception:
            return False

    @classmethod
    def get_redis(cls) -> redis_async.Redis:
        """
        Get the Redis connection instance from the singleton.
        Raises if RedisDB not initialized or Redis connection not ready.
        """
        from .engine import RedisDB
        if RedisDB._instance is None:
            raise RuntimeError("RedisDB not initialized. Please create and start it before using models.")
        redis_conn = RedisDB._instance._redis
        if redis_conn is None:
            raise RuntimeError("Redis connection not initialized yet. Call RedisDB.start() first.")
        return redis_conn

    def add_table(self, model_cls):
        """
        Register a model class (table) with this RedisDB instance.
        
        Args:
            model_cls: Model class derived from AsyncRedisBaseModel
        """
        key_prefix = model_cls.get_key_prefix()
        self.tables[key_prefix] = model_cls
        setattr(self, model_cls.__name__, model_cls)

    def get_table(self, name: str):
        """
        Retrieve registered model class by key prefix.
        
        Args:
            name (str): Key prefix or model name
        
        Returns:
            Model class or None if not registered.
        """
        return self.tables.get(name)

    def __getattr__(self, name: str):
        """
        Allow attribute access to registered models by their key prefix.
        Raises AttributeError if not found.
        """
        tables = self.__dict__.get('tables', {})
        if name in tables:
            return tables[name]
        raise AttributeError(f"'RedisDB' object has no attribute '{name}'")

    def __dir__(self):
        """
        Include registered model prefixes in dir() output.
        """
        return super().__dir__() + list(self.tables.keys())
