import os
import pytest
from unittest.mock import MagicMock, patch

from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.memory.few_shot_memory import FewShotExample, FewShotMemory


# Simple in-memory mock vector store for testing
class MockVectorStore:
    def __init__(self):
        self.examples = {}
        self.embedding_model_dims = 384

    def insert(self, vectors, ids, payloads):
        for i, _id in enumerate(ids):
            self.examples[_id] = {"vector": vectors[i], "payload": payloads[i]}
        return ids

    def search(self, query, vectors, limit=5, filters=None):
        # In a real scenario, we'd compute cosine similarity here
        # For testing, just return all examples with mock scores
        results = []
        for _id, example in self.examples.items():
            if filters:
                match = True
                for key, value in filters.items():
                    # Special handling for category filter to mimic actual implementation
                    if key == "category" and "category" in example["payload"].get("metadata", {}):
                        if example["payload"]["metadata"]["category"] != value:
                            match = False
                            break
                    elif key not in example["payload"] or example["payload"][key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(
                MagicMock(
                    id=_id,
                    score=0.1,  # Mock similarity score
                    payload=example["payload"]
                )
            )
            if len(results) >= limit:
                break
        return results

    def get(self, vector_id):
        if vector_id in self.examples:
            return MagicMock(
                id=vector_id,
                payload=self.examples[vector_id]["payload"]
            )
        return None

    def update(self, vector_id, vector=None, payload=None):
        if vector_id not in self.examples:
            return False
        
        if vector is not None:
            self.examples[vector_id]["vector"] = vector
        
        if payload is not None:
            self.examples[vector_id]["payload"] = payload
        
        return True

    def delete(self, vector_id):
        if vector_id in self.examples:
            del self.examples[vector_id]
            return True
        return False

    def list(self, filters=None, limit=100):
        results = []
        for _id, example in self.examples.items():
            if filters:
                match = True
                for key, value in filters.items():
                    if key not in example["payload"] or example["payload"][key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(
                MagicMock(
                    id=_id,
                    score=None,
                    payload=example["payload"]
                )
            )
            if len(results) >= limit:
                break
        
        return [results]


# Mock embedding model for testing
class MockEmbedder:
    def __init__(self, *args, **kwargs):
        pass
    
    def embed(self, text, memory_action=None):
        # Create a simple deterministic embedding for testing
        return [0.1] * 384


@pytest.fixture
def mock_factories():
    """Mock the factory classes to return our test implementations"""
    with patch("mem0.utils.factory.VectorStoreFactory.create") as mock_vector_factory:
        with patch("mem0.utils.factory.EmbedderFactory.create") as mock_embedder_factory:
            # Set up our mocks
            mock_vector_store = MockVectorStore()
            mock_vector_factory.return_value = mock_vector_store
            
            mock_embedder = MockEmbedder()
            mock_embedder_factory.return_value = mock_embedder
            
            yield {
                "vector_store": mock_vector_store,
                "embedder": mock_embedder
            }


@pytest.fixture
def memory(mock_factories):
    """Create a FewShotMemory instance with mocked dependencies"""
    config = MemoryConfig(
        vector_store=VectorStoreConfig(provider="faiss", config={"embedding_model_dims": 384}),
        embedder=EmbedderConfig(provider="openai", config={"embedding_dims": 384})
    )
    return FewShotMemory(config)


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
    
    def test_initialization(self, memory, mock_factories):
        """Test basic initialization of FewShotMemory"""
        assert memory is not None
        assert memory.vector_store == mock_factories["vector_store"]
        assert memory.embedding_model == mock_factories["embedder"]
        
    def test_add_example(self, memory):
        """Test adding examples to memory"""
        example_id = memory.add_example(
            question="What is the capital of France?",
            answer="Paris",
            metadata={"category": "geography"}
        )
        
        assert example_id is not None
        
        # Verify the example was added correctly
        example = memory.get(example_id)
        assert example is not None
        assert example.question == "What is the capital of France?"
        assert example.answer == "Paris"
        assert example.metadata["category"] == "geography"
        
    def test_get_examples(self, memory):
        """Test retrieving examples by similarity"""
        # Add some test examples
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris",
            metadata={"category": "geography"}
        )
        memory.add_example(
            question="What is the capital of Germany?",
            answer="Berlin",
            metadata={"category": "geography"}
        )
        memory.add_example(
            question="What is the capital of Italy?",
            answer="Rome",
            metadata={"category": "geography"}
        )
        memory.add_example(
            question="Who wrote Hamlet?",
            answer="William Shakespeare",
            metadata={"category": "literature"}
        )
        
        # Search for geography questions
        examples = memory.get_examples("What is the capital of Spain?", limit=3)
        
        # In our mock implementation, we're not doing real similarity search
        # But we should get the examples we added
        assert len(examples) > 0
        for example in examples:
            assert isinstance(example, FewShotExample)
            assert example.question is not None
            assert example.answer is not None
    
    def test_get_examples_with_filters(self, memory):
        """Test retrieving examples with filtering"""
        # Add examples with different categories
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris",
            metadata={"category": "geography"}
        )
        memory.add_example(
            question="What is the capital of Germany?",
            answer="Berlin",
            metadata={"category": "geography"}
        )
        memory.add_example(
            question="Who wrote Hamlet?",
            answer="William Shakespeare",
            metadata={"category": "literature"}
        )
        
        # Create custom filters
        filters = {"metadata.category": "geography"}
        
        # This won't actually filter in our mock implementation
        # but tests the API works correctly
        examples = memory.get_examples(
            "What is a famous play?", 
            limit=10,
            filters=filters
        )
        
        assert len(examples) > 0
    
    def test_get_formatted_examples(self, memory):
        """Test getting formatted examples"""
        # Add example
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        # Get formatted examples
        formatted = memory.get_formatted_examples(
            "What is the capital of Spain?",
            template="Q: {question}\nA: {answer}"
        )
        
        assert len(formatted) > 0
        for example in formatted:
            assert example.startswith("Q: ")
            assert "A: " in example
    
    def test_get_examples_as_context(self, memory):
        """Test getting examples as a single context string"""
        # Add examples
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        memory.add_example(
            question="What is the capital of Germany?",
            answer="Berlin"
        )
        
        # Get as context
        context = memory.get_examples_as_context(
            "What is the capital of Spain?",
            preamble="Here are some examples:",
            template="Q: {question}\nA: {answer}"
        )
        
        assert context.startswith("Here are some examples:")
        assert "Q: " in context
        assert "A: " in context
    
    def test_delete_example(self, memory):
        """Test deleting a specific example"""
        # Add example and then delete it
        example_id = memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        assert memory.get(example_id) is not None
        assert memory.delete_example(example_id) is True
        assert memory.get(example_id) is None
    
    def test_update_example(self, memory):
        """Test updating an example"""
        # Add example
        example_id = memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        # Update it
        result = memory.update(
            example_id=example_id,
            answer="Paris, the city of light",
            metadata={"category": "geography", "difficulty": "easy"}
        )
        
        assert result is True
        
        # Verify update
        updated = memory.get(example_id)
        assert updated is not None
        assert updated.answer == "Paris, the city of light"
        assert updated.metadata["category"] == "geography"
        assert updated.metadata["difficulty"] == "easy"
    
    def test_get_all(self, memory):
        """Test retrieving all examples"""
        # Add multiple examples
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        memory.add_example(
            question="What is the capital of Germany?",
            answer="Berlin"
        )
        
        # Get all examples
        examples = memory.get_all()
        
        assert len(examples) >= 2
        for example in examples:
            assert isinstance(example, FewShotExample)
            assert example.question is not None
            assert example.answer is not None
    
    def test_delete_all(self, memory):
        """Test deleting all examples"""
        # Add examples
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        memory.add_example(
            question="What is the capital of Germany?",
            answer="Berlin"
        )
        
        # Verify we have examples
        assert len(memory.get_all()) >= 2
        
        # Delete all
        result = memory.delete_all()
        assert result is True
        
        # Verify they're gone
        assert len(memory.get_all()) == 0
    
    def test_search_alias(self, memory):
        """Test that search is an alias for get_examples"""
        # Add example
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        # Search should return the same as get_examples
        search_results = memory.search("What is the capital?")
        examples = memory.get_examples("What is the capital?")
        
        assert len(search_results) == len(examples)
        
    def test_reset(self, memory):
        """Test resetting the memory"""
        # Add examples
        memory.add_example(
            question="What is the capital of France?",
            answer="Paris"
        )
        
        # Verify we have examples
        assert len(memory.get_all()) >= 1
        
        # Reset the memory
        result = memory.reset()
        assert result is True
        
        # Verify they're gone
        assert len(memory.get_all()) == 0 