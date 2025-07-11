from typing import Generic, TypeVar
from .engine import RedisApp
from .model import AsyncRedisBaseModel

T = TypeVar("T")

class TypedRedisApp(RedisApp, Generic[T]):
    pass


def create_db_with_models(models_cls: type) -> TypedRedisApp:
    db = TypedRedisApp[models_cls]()
    for name, model in vars(models_cls).items():
        if isinstance(model, type) and issubclass(model, AsyncRedisBaseModel):
            db.add_table(model)
    return db
