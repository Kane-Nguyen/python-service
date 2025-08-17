import motor.motor_asyncio
import os
import asyncio

MONGO_URI = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "db-dev")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]   
users_collection = db["users"]

print("users_collection :", users_collection)

async def check_db_connection():
    try:    
        await db.command("ping")  # MongoDB built-in "ping" command
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")

if __name__ == "__main__":
    asyncio.run(check_db_connection())
