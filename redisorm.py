# main.py

import asyncio
from .db import db

async def get_user():
    await db.user.get("some_id")

if __name__ == "__main__":
    asyncio.run(db.start())
