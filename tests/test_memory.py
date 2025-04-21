# import pytest

# from mem0 import Memory


# @pytest.fixture
# def memory_store():
#     return Memory()
# import pytest
# import os
# from mem0 import Memory
# from mem0.configs.base import MemoryConfig

# @pytest.fixture
# def memory_store():
#     # Get the .mem0 directory path (ensuring it exists with correct permissions)
#     mem0_dir = os.path.expanduser("~/.mem0")
#     os.makedirs(mem0_dir, exist_ok=True)
    
#     # Ensure test-specific subdirectory
#     test_dir = os.path.join(mem0_dir, "test_data")
#     os.makedirs(test_dir, exist_ok=True)
    
#     # Set proper permissions
#     os.chmod(mem0_dir, 0o700)  # rwx------
#     os.chmod(test_dir, 0o700)  # rwx------
    
#     # Create specific paths for databases
#     history_db_path = os.path.join(test_dir, "history.db")
    
#     # Configure Memory with explicit paths
#     config = MemoryConfig(
#         history_db_path=history_db_path,
#         vector_store={
#             "provider": "qdrant",  # Keep using qdrant since that's what the code uses by default
#             "config": {
#                 "path": os.path.join(test_dir, "qdrant"),
#                 "collection_name": "test_collection",
#                 "on_disk": True
#             }
#         }
#     )
    
#     # Return Memory instance with explicit configuration
#     return Memory(config=config)

import pytest
import os
import shutil
import uuid
from mem0 import Memory
from mem0.configs.base import MemoryConfig

# @pytest.fixture
# def memory_store():
#     # Create a unique test directory for each test
#     test_id = str(uuid.uuid4())
#     mem0_dir = os.path.expanduser("~/.mem0")
#     test_dir = os.path.join(mem0_dir, f"test_data_{test_id}")
    
#     # Clean up any existing directory (if somehow it exists)
#     if os.path.exists(test_dir):
#         shutil.rmtree(test_dir)
    
#     # Create fresh directories
#     os.makedirs(test_dir, exist_ok=True)
    
#     # Set proper permissions
#     os.chmod(test_dir, 0o700)  # rwx------
    
#     # Create paths for databases
#     history_db_path = os.path.join(test_dir, "history.db")
#     qdrant_path = os.path.join(test_dir, "qdrant")
#     os.makedirs(qdrant_path, exist_ok=True)
    
#     # Configure Memory
#     config = MemoryConfig(
#         history_db_path=history_db_path,
#         vector_store={
#             "provider": "qdrant",
#             "config": {
#                 "path": qdrant_path,
#                 "collection_name": "test_collection",
#                 "on_disk": True
#             }
#         }
#     )
    
#     # Create memory instance
#     memory = Memory(config=config)
    
#     # Return instance for test use
#     yield memory
    
#     # Clean up after test
#     # try:
#     #     shutil.rmtree(test_dir)
#     # except Exception as e:
#     #     print(f"Warning: Failed to clean up test directory {test_dir}: {e}")


# import pytest
# import os
# import shutil
# import time
# from mem0 import Memory
# from mem0.configs.base import MemoryConfig

# # Create a fixed test directory path - only one for all tests
# TEST_DATA_DIR = os.path.expanduser("~/.mem0/pytest_data")

# @pytest.fixture(scope="function")
# def memory_store():
#     # Create or clean the test directory
#     if os.path.exists(TEST_DATA_DIR):
#         try:
#             # Try to remove existing directory
#             shutil.rmtree(TEST_DATA_DIR)
#             # Wait a moment for file locks to release
#             time.sleep(0.1)
#         except Exception as e:
#             print(f"Warning: Failed to clean up previous test data: {e}")
    
#     # Create fresh directories
#     os.makedirs(TEST_DATA_DIR, exist_ok=True)
#     os.makedirs(os.path.join(TEST_DATA_DIR, "qdrant"), exist_ok=True)
    
#     # Set proper permissions
#     os.chmod(TEST_DATA_DIR, 0o700)  # rwx------
    
#     # Create paths for databases
#     history_db_path = os.path.join(TEST_DATA_DIR, "history.db")
#     qdrant_path = os.path.join(TEST_DATA_DIR, "qdrant")
    
#     # Configure Memory
#     config = MemoryConfig(
#         history_db_path=history_db_path,
#         vector_store={
#             "provider": "qdrant",
#             "config": {
#                 "path": qdrant_path,
#                 "collection_name": "test_collection",
#                 "on_disk": True
#             }
#         }
#     )
    
#     # Create memory instance
#     memory = Memory(config=config)
    
#     # Return instance for test use
#     yield memory
    
#     # Explicitly reset and cleanup connections
#     try:
#         # Close database connections
#         if hasattr(memory, 'db') and hasattr(memory.db, 'connection'):
#             memory.db.connection.close()
            
#         # Close vector store client if it has a close method
#         if hasattr(memory.vector_store, 'client') and hasattr(memory.vector_store.client, 'close'):
#             memory.vector_store.client.close()
            
#         # Allow a moment for connections to properly close
#         time.sleep(0.1)
#     except Exception as e:
#         print(f"Warning: Error closing connections: {e}")

# # Add a session-level fixture to do final cleanup after all tests
# # @pytest.fixture(scope="session", autouse=True)
# # def cleanup_after_all(request):
# #     # This runs at the very end of the test session
# #     def final_cleanup():
# #         if os.path.exists(TEST_DATA_DIR):
# #             try:
# #                 shutil.rmtree(TEST_DATA_DIR)
# #             except Exception as e:
# #                 print(f"Warning: Failed to clean up final test data: {e}")
    
# #     request.addfinalizer(final_cleanup)


import pytest
import os
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


# @pytest.mark.skip(reason="Not implemented")
def test_add_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    retrieved = memory_store.get(memory_id)
    assert retrieved["memory"] == data


# @pytest.mark.skip(reason="Not implemented")
def test_get_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    retrieved_data = memory_store.get(memory_id)
    assert retrieved_data["memory"] == data


# @pytest.mark.skip(reason="Not implemented")
def test_update_memory(memory_store):
    data = "Name is John Doe."
    user_id = "test_user"
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    
    new_data = "Name is John Kapoor."
    memory_store.update(memory_id, new_data)
    updated = memory_store.get(memory_id)
    assert updated["memory"] == new_data


# @pytest.mark.skip(reason="Not implemented")
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


# @pytest.mark.skip(reason="Not implemented")
# def test_get_all_memories(memory_store):
#     user_id = "test_user"
#     data1 = "Name is John Doe."
#     data2 = "I like to code in Python."
    
#     memory_store.add(data1, user_id=user_id, infer=False)
#     memory_store.add(data2, user_id=user_id, infer=False)
    
#     memories = memory_store.get_all(user_id=user_id)
    
#     # API v1.1 returns {"results": [...]} structure
#     if isinstance(memories, dict) and "results" in memories:
#         memories = memories["results"]
    
#     assert len(memories) == 2
#     memories_content = [mem["memory"] for mem in memories]
#     assert data1 in memories_content
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


# @pytest.mark.skip(reason="Not implemented")
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
