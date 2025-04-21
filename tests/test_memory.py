import pytest
import os
import uuid
import time
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.memory.setup import mem0_dir, get_user_id, setup_config

@pytest.fixture
def memory_store():
    # Ensure config exists
    setup_config()
    print(f"[DEBUG] Running setup_config()")
    
    # Use the default mem0 directory structure
    history_db_path = os.path.join(mem0_dir, "history.db")
    qdrant_path = os.path.join(mem0_dir, "qdrant")
    
    print(f"[DEBUG] Using mem0_dir: {mem0_dir}")
    print(f"[DEBUG] Using history_db_path: {history_db_path}")
    print(f"[DEBUG] Using qdrant_path: {qdrant_path}")
    
    # Ensure the directory structure exists
    os.makedirs(qdrant_path, exist_ok=True)
    print(f"[DEBUG] Created directory structure at {qdrant_path}")
    
    # Set proper permissions
    os.chmod(mem0_dir, 0o700)  # rwx------
    print(f"[DEBUG] Set permissions 700 on {mem0_dir}")
    
    # Configure Memory to use the main application database
    config = MemoryConfig(
        history_db_path=history_db_path,
        vector_store={
            "provider": "qdrant",
            "config": {
                "path": qdrant_path,
                "collection_name": "mem0",  # Use the default collection name
                "on_disk": True
            }
        }
    )
    print(f"[DEBUG] Created MemoryConfig with collection_name: 'mem0'")
    
    # Return memory instance with the application's database
    memory = Memory(config=config)
    print(f"[DEBUG] Initialized Memory instance")
    return memory


def test_add_memory(memory_store):
    data = "Name is John Doe."
    user_id = get_user_id()  # Use the real user ID
    print(f"[DEBUG] test_add_memory: Using user_id: {user_id}")
    print(f"[DEBUG] test_add_memory: Adding memory with data: '{data}'")
    
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    print(f"[DEBUG] test_add_memory: Added memory with ID: {memory_id}")
    print(f"[DEBUG] test_add_memory: Add result: {result}")
    
    print(f"[DEBUG] test_add_memory: Getting memory with ID: {memory_id}")
    retrieved = memory_store.get(memory_id)
    print(f"[DEBUG] test_add_memory: Retrieved memory: {retrieved}")
    
    assert retrieved["memory"] == data
    print(f"[DEBUG] test_add_memory: Assert passed - memory matches original data")


def test_get_memory(memory_store):
    data = "Name is John Doe."
    user_id = get_user_id()
    print(f"[DEBUG] test_get_memory: Using user_id: {user_id}")
    print(f"[DEBUG] test_get_memory: Adding memory with data: '{data}'")
    
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    print(f"[DEBUG] test_get_memory: Added memory with ID: {memory_id}")
    print(f"[DEBUG] test_get_memory: Add result: {result}")
    
    print(f"[DEBUG] test_get_memory: Getting memory with ID: {memory_id}")
    retrieved_data = memory_store.get(memory_id)
    print(f"[DEBUG] test_get_memory: Retrieved memory: {retrieved_data}")
    
    assert retrieved_data["memory"] == data
    print(f"[DEBUG] test_get_memory: Assert passed - memory matches original data")


def test_update_memory(memory_store):
    data = "Name is John Doe."
    user_id = get_user_id()
    print(f"[DEBUG] test_update_memory: Using user_id: {user_id}")
    print(f"[DEBUG] test_update_memory: Adding memory with data: '{data}'")
    
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    print(f"[DEBUG] test_update_memory: Added memory with ID: {memory_id}")
    print(f"[DEBUG] test_update_memory: Add result: {result}")
    
    new_data = "Name is John Kapoor."
    print(f"[DEBUG] test_update_memory: Updating memory with ID: {memory_id}")
    print(f"[DEBUG] test_update_memory: New data: '{new_data}'")
    
    memory_store.update(memory_id, new_data)
    print(f"[DEBUG] test_update_memory: Memory updated")
    
    print(f"[DEBUG] test_update_memory: Getting updated memory with ID: {memory_id}")
    updated = memory_store.get(memory_id)
    print(f"[DEBUG] test_update_memory: Retrieved updated memory: {updated}")
    
    assert updated["memory"] == new_data
    print(f"[DEBUG] test_update_memory: Assert passed - memory matches new data")


def test_delete_memory(memory_store):
    data = "Name is John Doe."
    user_id = get_user_id()
    print(f"[DEBUG] test_delete_memory: Using user_id: {user_id}")
    print(f"[DEBUG] test_delete_memory: Adding memory with data: '{data}'")
    
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    print(f"[DEBUG] test_delete_memory: Added memory with ID: {memory_id}")
    print(f"[DEBUG] test_delete_memory: Add result: {result}")
    
    print(f"[DEBUG] test_delete_memory: Deleting memory with ID: {memory_id}")
    memory_store.delete(memory_id)
    print(f"[DEBUG] test_delete_memory: Memory deleted")
    
    print(f"[DEBUG] test_delete_memory: Getting deleted memory with ID: {memory_id}")
    deleted = memory_store.get(memory_id)
    print(f"[DEBUG] test_delete_memory: Result of getting deleted memory: {deleted}")
    
    assert deleted is None
    print(f"[DEBUG] test_delete_memory: Assert passed - deleted memory is None")


def test_history(memory_store):
    data = "I like Indian food."
    user_id = get_user_id()
    print(f"[DEBUG] test_history: Using user_id: {user_id}")
    print(f"[DEBUG] test_history: Adding memory with data: '{data}'")
    
    result = memory_store.add(data, user_id=user_id, infer=False)
    memory_id = result["results"][0]["id"]
    print(f"[DEBUG] test_history: Added memory with ID: {memory_id}")
    print(f"[DEBUG] test_history: Add result: {result}")
    
    print(f"[DEBUG] test_history: Getting history for memory ID: {memory_id}")
    history = memory_store.history(memory_id)
    print(f"[DEBUG] test_history: Initial history: {history}")
    
    assert len(history) == 1
    # The correct key is "new_memory" not "content"
    assert history[0]["new_memory"] == data
    print(f"[DEBUG] test_history: Assert passed - history length is 1 and new_memory matches data")
    
    new_data = "I like Italian food."
    print(f"[DEBUG] test_history: Updating memory with ID: {memory_id}")
    print(f"[DEBUG] test_history: New data: '{new_data}'")
    
    memory_store.update(memory_id, new_data)
    print(f"[DEBUG] test_history: Memory updated")
    
    print(f"[DEBUG] test_history: Getting updated history for memory ID: {memory_id}")
    history = memory_store.history(memory_id)
    print(f"[DEBUG] test_history: Updated history: {history}")
    
    assert len(history) == 2
    assert history[0]["new_memory"] == data
    assert history[1]["old_memory"] == data
    assert history[1]["new_memory"] == new_data
    assert history[1]["event"] == "UPDATE"
    print(f"[DEBUG] test_history: Assert passed - history updated correctly")


def test_get_all_memories(memory_store):
    user_id = get_user_id()
    data1 = f"Test data 1: {uuid.uuid4()}"  # Make each test run's data unique
    data2 = f"Test data 2: {uuid.uuid4()}"
    
    print(f"[DEBUG] test_get_all_memories: Using user_id: {user_id}")
    print(f"[DEBUG] test_get_all_memories: Test data 1: '{data1}'")
    print(f"[DEBUG] test_get_all_memories: Test data 2: '{data2}'")
    
    # Add new test-specific data
    print(f"[DEBUG] test_get_all_memories: Adding first memory")
    result1 = memory_store.add(data1, user_id=user_id, infer=False)
    print(f"[DEBUG] test_get_all_memories: First add result: {result1}")
    
    print(f"[DEBUG] test_get_all_memories: Adding second memory")
    result2 = memory_store.add(data2, user_id=user_id, infer=False)
    print(f"[DEBUG] test_get_all_memories: Second add result: {result2}")
    
    # Get newly added memory IDs
    memory_id1 = result1["results"][0]["id"]
    memory_id2 = result2["results"][0]["id"]
    print(f"[DEBUG] test_get_all_memories: First memory ID: {memory_id1}")
    print(f"[DEBUG] test_get_all_memories: Second memory ID: {memory_id2}")
    
    # Retrieve all memories
    print(f"[DEBUG] test_get_all_memories: Getting all memories for user: {user_id}")
    memories = memory_store.get_all(user_id=user_id)
    print(f"[DEBUG] test_get_all_memories: get_all result: {memories}")
    
    if isinstance(memories, dict) and "results" in memories:
        print(f"[DEBUG] test_get_all_memories: Extracting 'results' from memories")
        memories = memories["results"]
        print(f"[DEBUG] test_get_all_memories: Extracted results: {memories}")
    
    # Check that our new entries are in the results
    memory_ids = [memory["id"] for memory in memories]
    print(f"[DEBUG] test_get_all_memories: All memory IDs: {memory_ids}")
    
    assert memory_id1 in memory_ids
    assert memory_id2 in memory_ids
    print(f"[DEBUG] test_get_all_memories: Assert passed - both memory IDs found in results")


def test_search_memories(memory_store):
    user_id = get_user_id()
    data1 = "I love playing tennis."
    data2 = "Tennis is my favorite sport."
    
    print(f"[DEBUG] test_search_memories: Using user_id: {user_id}")
    print(f"[DEBUG] test_search_memories: Test data 1: '{data1}'")
    print(f"[DEBUG] test_search_memories: Test data 2: '{data2}'")
    
    # Use retry mechanism for vector operations that might encounter lock issues
    def add_data1():
        print(f"[DEBUG] test_search_memories: Adding first memory")
        result = memory_store.add(data1, user_id=user_id, infer=False)
        print(f"[DEBUG] test_search_memories: First add result: {result}")
        return result
    
    def add_data2():
        print(f"[DEBUG] test_search_memories: Adding second memory")
        result = memory_store.add(data2, user_id=user_id, infer=False)
        print(f"[DEBUG] test_search_memories: Second add result: {result}")
        return result
    
    print(f"[DEBUG] test_search_memories: Calling retry_operation for first memory")
    first_result = retry_operation(add_data1)
    print(f"[DEBUG] test_search_memories: retry_operation result for first memory: {first_result}")
    
    print(f"[DEBUG] test_search_memories: Calling retry_operation for second memory")
    second_result = retry_operation(add_data2)
    print(f"[DEBUG] test_search_memories: retry_operation result for second memory: {second_result}")
    
    # Use retry mechanism for search operation
    def search_tennis():
        print(f"[DEBUG] test_search_memories: Searching for 'tennis'")
        search_result = memory_store.search("tennis", user_id=user_id)
        print(f"[DEBUG] test_search_memories: Search result: {search_result}")
        return search_result
    
    print(f"[DEBUG] test_search_memories: Calling retry_operation for search")
    results = retry_operation(search_tennis)
    print(f"[DEBUG] test_search_memories: retry_operation result for search: {results}")
    
    # API v1.1 returns {"results": [...]} structure
    if isinstance(results, dict) and "results" in results:
        print(f"[DEBUG] test_search_memories: Extracting 'results' from search results")
        results = results["results"]
        print(f"[DEBUG] test_search_memories: Extracted results: {results}")
    
    assert len(results) > 0
    found_memories = [result["memory"] for result in results]
    print(f"[DEBUG] test_search_memories: Found memories: {found_memories}")
    
    assert any(memory in found_memories for memory in [data1, data2])
    print(f"[DEBUG] test_search_memories: Assert passed - at least one test memory found in results")


# Add a retry mechanism for Qdrant operations that might encounter locks
def retry_operation(func, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] retry_operation: Executing operation, attempt {attempt+1}/{max_retries}")
            result = func()
            print(f"[DEBUG] retry_operation: Operation succeeded on attempt {attempt+1}")
            return result
        except RuntimeError as e:
            print(f"[DEBUG] retry_operation: Operation failed with error: {str(e)}")
            if "already accessed by another instance" in str(e) and attempt < max_retries - 1:
                print(f"[DEBUG] retry_operation: Database lock detected, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            print(f"[DEBUG] retry_operation: Max retries exceeded or unexpected error. Raising exception.")
            raise