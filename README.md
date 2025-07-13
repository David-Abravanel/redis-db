מצוין! הנה גרסה מעודכנת של קובץ `README.md` שמתאימה למבנה הנוכחי שלך, כולל ההגדרה הנדרשת של מחלקת DB שמאפשרת גישה דרך `db.Users`:

---

````markdown
# RedisORM

A lightweight asynchronous ORM-style interface for storing and retrieving Pydantic models in Redis.

## Features

- 🔁 Asynchronous Redis support via `redis.asyncio`
- 🧱 Models based on `pydantic.BaseModel`
- 🆔 Automatic ID generation (UUID4)
- ⏳ Optional TTL per object
- 🕓 Auto-managed `created_at` and `updated_at` timestamps
- 🧩 Automatic JSON serialization for `dict` and `list` fields
- 📦 Simple Redis hash-based storage
- 🧠 Built-in model registry with dynamic access via `db.YourModel`

---

## Quick Start

### 1. Define your model

```python
from redisorm import RedisBaseModel

class Users(RedisBaseModel):
    name: str
    email: str
    age: int
````

### 2. Register models in your DB class

```python
from redisorm import RedisDB

class DB(RedisDB):
    Users = Users  # Capitalized to match class name

db = DB(
    host='localhost',
    port=6379,
    db=0,
)
```

### 3. Start Redis (auto-launches local server on Linux if needed)

```python
await db.start()
```

### 4. Use your model

```python
# Save a new user
user = Users(name="Alice", email="alice@example.com", age=30)
await user.save()

# Get by ID
retrieved = await db.Users.get(user.id)

# Update specific fields
await db.Users.update(user.id, age=31)

# Delete
await db.Users.delete(user.id)
```

---

## Notes

* Redis keys are structured as: `_t:users:{id}` (you can override `_t:users` via `Meta.key_prefix`)
* Uses Redis hashes under the hood (via `HMSET`, `HGETALL`)
* `created_at` and `updated_at` timestamps are stored automatically
* `dict` and `list` fields are serialized to JSON strings

---

## Requirements

* Python 3.8+
* `pydantic>=2.0`
* `redis>=5.0`

Install with:

```bash
pip install redis pydantic
```

---

## License

MIT License
