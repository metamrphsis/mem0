import os
import pytest
import shutil
import tempfile
import uuid
from datetime import datetime
from unittest.mock import patch, Mock, MagicMock

import pytz

from mem0.configs.base import MemoryConfig
from mem0.configs.enums import MemoryType
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.memory.few_shot_memory import FewShotExample, FewShotMemory, MultiDatabaseManager
from mem0.memory.setup import mem0_dir


@pytest.fixture
def test_directory():
    """Create a temporary test directory and clean it up after the test"""
    # Create a unique test directory for each test
    test_dir = tempfile.mkdtemp(prefix="mem0_test_")
    
    # Set up necessary subdirectories
    os.makedirs(os.path.join(test_dir, "vector_store"), exist_ok=True)
    
    # Override mem0_dir for testing
    original_mem0_dir = mem0_dir
    
    # Patch mem0_dir to use our test directory
    import mem0.memory.setup
    import mem0.memory.few_shot_memory
    
    mem0.memory.setup.mem0_dir = test_dir
    mem0.memory.few_shot_memory.mem0_dir = test_dir
    
    # Update vector_store_dir and history_db_path
    mem0.memory.few_shot_memory.vector_store_dir = os.path.join(test_dir, "vector_store")
    mem0.memory.few_shot_memory.history_db_path = os.path.join(test_dir, "history.db")
    
    # Return the test directory path
    yield test_dir
    
    # Clean up the test directory
    shutil.rmtree(test_dir)
    
    # Restore original mem0_dir
    mem0.memory.setup.mem0_dir = original_mem0_dir
    mem0.memory.few_shot_memory.mem0_dir = original_mem0_dir


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = MagicMock()
    
    # Create sample search results
    all_search_results = []
    for i in range(4):
        mock_result = MagicMock()
        mock_result.id = f"id-{i}"
        mock_result.score = 0.1 * i
        
        if i < 3:
            mock_result.payload = {
                "question": f"What is the capital of country {i}?",
                "answer": f"Capital {i}",
                "metadata": {"category": "geography"}
            }
        else:
            mock_result.payload = {
                "question": "Who wrote Hamlet?",
                "answer": "William Shakespeare",
                "metadata": {"category": "literature"}
            }
        
        all_search_results.append(mock_result)
    
    # Override the search method to respect filters
    def filtered_search(query, vectors, limit=5, filters=None):
        results = all_search_results.copy()
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    # Handle metadata.category special case
                    if key == "metadata.category" and "metadata" in result.payload:
                        if result.payload["metadata"].get("category") != value:
                            match = False
                            break
                    # Handle regular filters
                    elif key in result.payload and result.payload[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            return filtered_results[:limit]
        
        # If no filters, just return all results
        return results[:limit]
    
    # Set the search method
    mock_store.search.side_effect = filtered_search
    
    # Mock insert method
    mock_store.insert.return_value = [str(uuid.uuid4())]
    
    # Mock get method
    mock_get_result = MagicMock()
    mock_get_result.id = "test-id"
    mock_get_result.payload = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "metadata": {"category": "geography"}
    }
    mock_store.get.return_value = mock_get_result
    
    # Mock update method
    mock_store.update.return_value = True
    
    # Mock delete method
    mock_store.delete.return_value = True
    
    # Mock list method
    mock_list_results = []
    for i in range(5):
        mock_result = MagicMock()
        mock_result.id = f"list-id-{i}"
        mock_result.payload = {
            "question": f"Test question {i}?",
            "answer": f"Test answer {i}",
            "metadata": {"category": "test", "index": i}
        }
        mock_list_results.append(mock_result)
    
    mock_store.list.return_value = [mock_list_results]
    
    return mock_store


@pytest.fixture
def mock_openai_client():
    """Mock the OpenAI client for embedding generation"""
    with patch("mem0.embeddings.openai.OpenAI") as mock_openai:
        mock_client = Mock()
        # Create a mock response for the embeddings.create method
        mock_response = Mock()
        # Generate a deterministic mock embedding of the right size
        mock_response.data = [Mock(embedding=[0.1 * i for i in range(384)])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_history_manager():
    """Mock the SQLite history manager"""
    mock_manager = MagicMock()
    
    # Mock add_history method
    mock_manager.add_history.return_value = True
    
    # Mock get_history method
    mock_history = [
        {"memory_id": "test-id", "event": "create", "created_at": "2023-01-01"},
        {"memory_id": "test-id", "event": "update", "created_at": "2023-01-02"}
    ]
    mock_manager.get_history.return_value = mock_history
    
    return mock_manager


@pytest.fixture
def memory(test_directory, mock_openai_client, mock_vector_store, mock_history_manager):
    """Create a FewShotMemory instance with mocked dependencies"""
    # Create a configuration that uses the local FAISS vector store and OpenAI embedder
    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="faiss",
            config={
                "collection_name": f"test_collection_{uuid.uuid4().hex}",
                "embedding_model_dims": 384,
                "path": os.path.join(test_directory, "vector_store"),
            }
        ),
        embedder=EmbedderConfig(
            provider="openai",  # Using OpenAI embedder with mocked client
            config={
                "model": "text-embedding-3-small",
                "embedding_dims": 384,
                "api_key": "sk-test-key"
            }
        )
    )
    
    # Patch VectorStoreFactory.create to return our mock vector store
    with patch("mem0.utils.factory.VectorStoreFactory.create", return_value=mock_vector_store):
        # Patch SQLiteManager to return our mock history manager
        with patch("mem0.memory.storage.SQLiteManager", return_value=mock_history_manager):
            # Create the memory instance
            memory = FewShotMemory(config)
            
            # Manually inject the mock vector store and history manager
            memory.db_manager.vector_stores = {"faiss": mock_vector_store}
            memory.db_manager.history_manager = mock_history_manager
            
            yield memory


class TestFewShotExample:
    """Tests for the FewShotExample class"""
    
    def test_initialization(self):
        """Test basic initialization of FewShotExample"""
        example = FewShotExample(
            question="What is the capital of France?",
            answer="Paris",
            metadata={"category": "geography"}
        )
        
        assert example.question == "What is the capital of France?"
        assert example.answer == "Paris"
        assert example.metadata == {"category": "geography"}
        assert example.id is not None
        
    def test_format_default(self):
        """Test default formatting of examples"""
        example = FewShotExample(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        formatted = example.format()
        assert "Question: What is the capital of France?" in formatted
        assert "Answer: Paris" in formatted
        
    def test_format_custom_template(self):
        """Test custom template formatting"""
        example = FewShotExample(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        template = "Q: {question}\nA: {answer}"
        formatted = example.format(template)
        assert formatted == "Q: What is the capital of France?\nA: Paris"
        
    def test_to_dict(self):
        """Test conversion to dictionary"""
        example = FewShotExample(
            question="What is the capital of France?",
            answer="Paris",
            example_id="test123",
            metadata={"category": "geography"},
            score=0.95
        )
        
        data = example.to_dict()
        assert data["id"] == "test123"
        assert data["question"] == "What is the capital of France?"
        assert data["answer"] == "Paris"
        assert data["metadata"] == {"category": "geography"}
        assert data["score"] == 0.95
        
    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "id": "test123",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "metadata": {"category": "geography"},
            "score": 0.95
        }
        
        example = FewShotExample.from_dict(data)
        assert example.id == "test123"
        assert example.question == "What is the capital of France?"
        assert example.answer == "Paris"
        assert example.metadata == {"category": "geography"}
        assert example.score == 0.95


class TestFewShotMemory:
    """Tests for the FewShotMemory class"""
    
    def test_initialization(self, memory):
        """Test basic initialization of FewShotMemory"""
        assert memory is not None
        assert isinstance(memory.db_manager, MultiDatabaseManager)
        assert "faiss" in memory.db_manager.vector_stores
        
    def test_add_example(self, memory):
        """Test adding examples to memory"""
        # Add a new example
        example_id = memory.add_example(
            question="What is the capital of France?",
            answer="Paris",
            metadata={"category": "geography"}
        )
        
        # Verify ID was returned
        assert example_id is not None
        
        # Retrieve the example to verify it was added
        example = memory.get(example_id)
        assert example is not None
        assert example.question == "What is the capital of France?"
        assert example.answer == "Paris"
        assert example.metadata["category"] == "geography"
        
    def test_get_examples(self, memory):
        """Test retrieving examples by similarity"""
        # Search for geography related examples
        examples = memory.get_examples(
            "What is the capital of Germany?",
            limit=5,
            threshold=1.0  # Set high threshold to get all examples
        )
        
        # Verify results include geography examples
        geography_found = False
        literature_found = False
        
        for example in examples:
            if "capital" in example.question:
                geography_found = True
            if "Hamlet" in example.question:
                literature_found = True
                
        assert geography_found, "Should find geography examples"
        assert len(examples) > 0, "Should return examples"
    
    def test_get_examples_with_filters(self, memory, mock_vector_store):
        """Test retrieving examples with filtering"""
        # Create filters for geography category
        filters = {"metadata.category": "geography"}
        
        # Search with filters
        examples = memory.get_examples(
            "What is a famous play?", 
            limit=10,
            filters=filters,
            threshold=1.0  # Set high threshold to get all examples
        )
        
        # Check that the search was called with the right filters
        # This verifies that filtering logic is attempted correctly, even if the mock doesn't implement filtering
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        
        # Check that filters were passed
        assert call_args is not None, "search method was not called with arguments"
        assert 'filters' in call_args[1], "filters parameter was not passed to search method"
        
        # Check that the filters contain the metadata.category field
        passed_filters = call_args[1]['filters']
        assert passed_filters is not None
        assert "category" in passed_filters, "category filter was not properly processed"
        assert passed_filters["category"] == "geography", "wrong category value passed to filter"
    
    def test_delete_example(self, memory):
        """Test deleting a specific example"""
        # Delete an example
        result = memory.delete_example("test-id")
        
        # Verify deletion
        assert result is True
        
    def test_update_example(self, memory):
        """Test updating an example"""
        # Update the example
        result = memory.update(
            example_id="test-id",
            answer="Updated answer",
            metadata={"category": "test", "updated": True}
        )
        
        # Verify update
        assert result is True
        
    def test_get_all(self, memory):
        """Test retrieving all examples"""
        # Get all examples
        examples = memory.get_all()
        
        # Verify
        assert len(examples) >= 5
        
        # Check some examples
        test_questions = [ex.question for ex in examples]
        assert any("Test question" in q for q in test_questions)
    
    def test_delete_all(self, memory, mock_vector_store):
        """Test deleting all examples"""
        # Set up the mock to return examples for get_all
        mock_list_results = []
        for i in range(2):
            mock_result = MagicMock()
            mock_result.id = f"to-delete-{i}"
            mock_result.payload = {
                "question": f"Delete me {i}",
                "answer": f"Answer {i}",
                "metadata": {"category": "test"}
            }
            mock_list_results.append(mock_result)
        
        mock_vector_store.list.return_value = [mock_list_results]
        
        # Delete all
        result = memory.delete_all()
        
        # Verify
        assert result is True
        assert mock_vector_store.delete.call_count >= 1
    
    def test_reset(self, memory):
        """Test resetting the memory"""
        # Reset should call delete_all
        with patch.object(memory, 'delete_all', return_value=True) as mock_delete_all:
            result = memory.reset()
            
            # Verify
            assert result is True
            mock_delete_all.assert_called_once()

    def test_history(self, memory, mock_history_manager):
        """Test history tracking"""
        # Get history
        history = memory.history("test-id")
        
        # Verify history has entries
        assert len(history) > 0
        
        # Verify we have both create and update events
        events = [entry.get("event") for entry in history]
        assert "create" in events
        assert "update" in events
