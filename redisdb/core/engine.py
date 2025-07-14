# redisorm/core/engine.py
"""
Optimized RedisDB class with connection pooling and performance improvements.
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
from redis.asyncio.connection import ConnectionPool


class RedisDB:
    """
    Optimized singleton class managing Redis connections with connection pooling.
    """
    _instance: Optional['RedisDB'] = None
    REDIS_DIR = Path('/tmp') / 'redis_local'  # Changed to /tmp for better permissions
    REDIS_SERVER_PATH = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisDB, cls).__new__(cls)
        return cls._instance

    def __init__(self, host=None, port=None, db=0, password=None, 
                 max_connections=50, socket_keepalive=True, 
                 socket_keepalive_options=None):
        """
        Initialize RedisDB with connection pooling optimizations.
        
        Args:
            host (str): Redis server hostname or IP
            port (int): Redis server port
            db (int): Redis database number
            password (str): Redis password
            max_connections (int): Maximum connections in pool
            socket_keepalive (bool): Enable TCP keepalive
            socket_keepalive_options (dict): TCP keepalive options
        """
        if self.__dict__.get("_initialized", False):
            return

        self.host = host or 'localhost'
        self.port = port or 6379
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_keepalive = socket_keepalive
        self.socket_keepalive_options = socket_keepalive_options or {}
        
        self._redis: Optional[redis_async.Redis] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._process = None
        self.tables: Dict[str, Type] = {}
        self._initialized = True

    async def start(self):
        """
        Start Redis connection with optimized connection pool.
        """
        if await self._test_redis_connection(self.host, self.port):
            await self._create_connection_pool()
            return

        print("Redis server not running, starting local instance...")
        self.REDIS_DIR.mkdir(parents=True, exist_ok=True)

        redis_path = shutil.which('redis-server')
        if redis_path is None:
            await asyncio.get_event_loop().run_in_executor(
                None, self._download_and_install_redis
            )

        await self._start_local_redis()
        await self._create_connection_pool()

    async def _create_connection_pool(self):
        """Create optimized connection pool."""
        pool_kwargs = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'password': self.password,
            'max_connections': self.max_connections,
            'socket_keepalive': self.socket_keepalive,
            'socket_keepalive_options': self.socket_keepalive_options,
            'retry_on_timeout': True,
            'health_check_interval': 30,
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
        }
        
        self._connection_pool = ConnectionPool(**pool_kwargs)
        self._redis = redis_async.Redis(connection_pool=self._connection_pool)
        
        # Test connection
        try:
            await self._redis.ping()
            print(f"Connected to Redis at {self.host}:{self.port} with {self.max_connections} max connections")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")

    async def _start_local_redis(self):
        """Start local Redis server with optimized configuration."""
        self.REDIS_SERVER_PATH = shutil.which('redis-server') or str(self.REDIS_DIR / 'redis-server')
        if not self.REDIS_SERVER_PATH or not Path(self.REDIS_SERVER_PATH).exists():
            raise RuntimeError("redis-server executable not found after installation.")

        # Create optimized Redis configuration
        config_path = self.REDIS_DIR / 'redis.conf'
        await self._create_redis_config(config_path)

        # Start Redis with configuration
        self._process = subprocess.Popen([
            self.REDIS_SERVER_PATH, 
            str(config_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"Started redis-server with PID {self._process.pid}")

        # Wait for server to start
        for _ in range(15):  # Increased timeout
            if await self._test_redis_connection('localhost', self.port):
                print("Local Redis server started successfully.")
                return
            await asyncio.sleep(0.5)

        self._process.terminate()
        raise RuntimeError("Failed to start Redis server")

    async def _create_redis_config(self, config_path: Path):
        """Create optimized Redis configuration file."""
        config_content = f"""
# Basic configuration
port {self.port}
bind 127.0.0.1
timeout 0
tcp-keepalive 300

# Memory optimizations
maxmemory-policy allkeys-lru
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Performance optimizations
save ""
appendonly no
stop-writes-on-bgsave-error no
rdbcompression no
rdbchecksum no

# Network optimizations
tcp-backlog 511
tcp-keepalive 300

# Logging
loglevel notice
logfile ""

# Directory
dir {self.REDIS_DIR}
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content.strip())

    def _download_and_install_redis(self):
        """Download and install Redis with optimizations."""
        system = platform.system()
        
        if system == 'Linux':
            url = 'http://download.redis.io/releases/redis-7.2.4.tar.gz'  # Updated version
            tar_path = self.REDIS_DIR / 'redis.tar.gz'

            print(f"Downloading Redis from {url}...")
            urlretrieve(url, tar_path)

            print("Extracting Redis...")
            subprocess.run(['tar', '-xzf', str(tar_path), '-C', str(self.REDIS_DIR)], 
                          check=True, stdout=subprocess.DEVNULL)

            redis_source_dir = next(self.REDIS_DIR.glob('redis-*'))

            print("Compiling Redis with optimizations...")
            # Compile with optimizations
            subprocess.run([
                'make', 
                'PREFIX=' + str(self.REDIS_DIR),
                'OPTIMIZATION=-O3',
                'CFLAGS=-march=native'
            ], cwd=str(redis_source_dir), check=True, 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Install
            subprocess.run(['make', 'install', 'PREFIX=' + str(self.REDIS_DIR)], 
                          cwd=str(redis_source_dir), check=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print(f"Redis installed to {self.REDIS_DIR}")

        elif system == 'Windows':
            raise RuntimeError("Auto-installation not supported on Windows. Please install Redis manually.")
        else:
            raise RuntimeError(f"Unsupported OS: {system}")

    async def _test_redis_connection(self, host, port) -> bool:
        """Test Redis connection with timeout."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    def get_redis(self) -> redis_async.Redis:
        """Get Redis connection from pool."""
        if self._redis is None:
            raise RuntimeError("Redis connection not initialized. Call start() first.")
        return self._redis

    async def close(self):
        """Properly close Redis connections and local server."""
        if self._connection_pool:
            await self._connection_pool.disconnect()
        
        if self._process:
            self._process.terminate()
            self._process.wait()
            print("Local Redis server stopped")

    def add_table(self, model_cls):
        """Register model class."""
        key_prefix = model_cls.get_key_prefix()
        self.tables[key_prefix] = model_cls
        setattr(self, model_cls.__name__, model_cls)

    def get_table(self, name: str):
        """Get registered model class."""
        return self.tables.get(name)

    def __getattr__(self, name: str):
        """Allow attribute access to registered models."""
        tables = self.__dict__.get('tables', {})
        if name in tables:
            return tables[name]
        raise AttributeError(f"'RedisDB' object has no attribute '{name}'")

    def __dir__(self):
        """Include registered models in dir() output."""
        return super().__dir__() + list(self.tables.keys())

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # Health check and monitoring methods
    async def health_check(self) -> Dict[str, any]:
        """Perform health check on Redis connection."""
        try:
            info = await self._redis.info()
            return {
                'status': 'healthy',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'uptime': info.get('uptime_in_seconds', 0),
                'version': info.get('redis_version', 'unknown')
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def get_stats(self) -> Dict[str, any]:
        """Get Redis performance statistics."""
        try:
            info = await self._redis.info()
            return {
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'used_memory_peak_human': info.get('used_memory_peak_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0)
            }
        except Exception as e:
            return {'error': str(e)}