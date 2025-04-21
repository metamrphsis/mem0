# import pytest

import pytest
import os
import shutil
import uuid
from mem0 import Memory
from mem0.configs.base import MemoryConfig

# Define a fixed location for the test database
PERSISTENT_TEST_DB = os.path.expanduser("~/.mem0/persistent_test_db")

@pytest.fixture
def memory_store():
    # Create the directory structure if it doesn't exist
    os.makedirs(PERSISTENT_TEST_DB, exist_ok=True)
    os.makedirs(os.path.join(PERSISTENT_TEST_DB, "qdrant"), exist_ok=True)
    
    # Set proper permissions
    os.chmod(PERSISTENT_TEST_DB, 0o700)  # rwx------
    
    # Define database paths
    history_db_path = os.path.join(PERSISTENT_TEST_DB, "history.db")
    qdrant_path = os.path.join(PERSISTENT_TEST_DB, "qdrant")
    
    # Configure Memory with persistence
    config = MemoryConfig(
        history_db_path=history_db_path,
        vector_store={
            "provider": "qdrant",
            "config": {
                "path": qdrant_path,
                "collection_name": "persistent_collection",
                "on_disk": True
            }
        }
    )
    
    # Return memory instance with persistent storage
    return Memory(config=config)


def test_add_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    retrieved = memory_store.get(memory_id)
    assert retrieved["memory"] == data


def test_get_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    retrieved_data = memory_store.get(memory_id)
    assert retrieved_data["memory"] == data


def test_update_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    
    new_data = "Name is John Kapoor."
    memory_store.update(memory_id, new_data)
    updated = memory_store.get(memory_id)
    assert updated["memory"] == new_data


def test_delete_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    
    memory_store.delete(memory_id)
    deleted = memory_store.get(memory_id)
    assert deleted is None


def test_history(memory_store):
    data = "I like Indian food."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    
    history = memory_store.history(memory_id)
    assert len(history) == 1
    # The correct key is "new_memory" not "content"
    assert history[0]["new_memory"] == data
    
    new_data = "I like Italian food."
    memory_store.update(memory_id, new_data)
    history = memory_store.history(memory_id)
    assert len(history) == 2
    assert history[0]["new_memory"] == data
    assert history[1]["old_memory"] == data
    assert history[1]["new_memory"] == new_data
    assert history[1]["event"] == "UPDATE"

#     assert data2 in memories_content
def test_get_all_memories(memory_store):
    user_id = "test_user"
    data1 = f"Test data 1: {uuid.uuid4()}"  # Make each test run's data unique
    data2 = f"Test data 2: {uuid.uuid4()}"
    
    # Add new test-specific data
    result1 = memory_store.add(data1, user_id=user_id, infer=False)
    result2 = memory_store.add(data2, user_id=user_id, infer=False)
    
    # Get newly added memory IDs
    memory_id1 = result1["results"][0]["id"]
    memory_id2 = result2["results"][0]["id"]
    
    # Retrieve all memories
    memories = memory_store.get_all(user_id=user_id)
    if isinstance(memories, dict) and "results" in memories:
        memories = memories["results"]
    
    # Check that our new entries are in the results
    memory_ids = [memory["id"] for memory in memories]
    assert memory_id1 in memory_ids
    assert memory_id2 in memory_ids


def test_search_memories(memory_store):
    user_id = "test_user"
    data1 = "I love playing tennis."
    data2 = "Tennis is my favorite sport."
    
    memory_store.add(data1, user_id=user_id, infer=False)
    memory_store.add(data2, user_id=user_id, infer=False)
    
    results = memory_store.search("tennis", user_id=user_id)
    
    # API v1.1 returns {"results": [...]} structure
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    
    assert len(results) > 0
    found_memories = [result["memory"] for result in results]
    assert any(memory in found_memories for memory in [data1, data2])

import time

# Add a retry mechanism for Qdrant operations that might encounter locks
def retry_operation(func, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except RuntimeError as e:
            if "already accessed by another instance" in str(e) and attempt < max_retries - 1:
                time.sleep(delay)
                continue
            raise