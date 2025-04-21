# to resolve sqlite3.OperationalError: attempt to write a readonly database

```
# Fix directory permissions (make sure it's fully owned by you)
chmod -R 700 ~/.mem0
# Remove any problematic extended attributes
xattr -cr ~/.mem0
```

test_memory.py:
import pytest
import os
import shutil
import uuid
from mem0 import Memory
from mem0.configs.base import MemoryConfig

@pytest.fixture
def memory_store():
    # Create a unique test directory for each test
    test_id = str(uuid.uuid4())
    mem0_dir = os.path.expanduser("~/.mem0")
    test_dir = os.path.join(mem0_dir, f"test_data_{test_id}")
    
    # Clean up any existing directory (if somehow it exists)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Create fresh directories
    os.makedirs(test_dir, exist_ok=True)
    
    # Set proper permissions
    os.chmod(test_dir, 0o700)  # rwx------
    
    # Create paths for databases
    history_db_path = os.path.join(test_dir, "history.db")
    qdrant_path = os.path.join(test_dir, "qdrant")
    os.makedirs(qdrant_path, exist_ok=True)
    
    # Configure Memory
    config = MemoryConfig(
        history_db_path=history_db_path,
        vector_store={
            "provider": "qdrant",
            "config": {
                "path": qdrant_path,
                "collection_name": "test_collection",
                "on_disk": True
            }
        }
    )
    
    # Create memory instance
    memory = Memory(config=config)
    
    # Return instance for test use
    yield memory
    
    # Clean up after test
    try:
        shutil.rmtree(test_dir)
    except Exception as e:
        print(f"Warning: Failed to clean up test directory {test_dir}: {e}")


# to open DB Browser for SQLite
open -a "DB Browser for SQLite" ~/.mem0/persistent_test_db/history.db