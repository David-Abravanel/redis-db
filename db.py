# model_test.py
# models.py

from redisorm import AsyncRedisBaseModel, RedisDB


class User(AsyncRedisBaseModel):
    name: str
    email: str

class DBTables(RedisDB):
    user = User

db = DBTables(
    host='localhost',
    port=6379,
    db=0,
)