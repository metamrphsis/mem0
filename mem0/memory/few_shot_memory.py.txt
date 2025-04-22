import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pytz

from mem0.configs.base import MemoryConfig
from mem0.configs.enums import MemoryType
from mem0.memory.base import MemoryBase
from mem0.memory.setup import setup_config, get_user_id
from mem0.utils.factory import EmbedderFactory, VectorStoreFactory

logger = logging.getLogger(__name__)

# Setup user config
setup_config()

class FewShotExample:
    """Represents a few-shot example with question, answer, and metadata."""
    
    def __init__(self, 
                 question: str, 
                 answer: str, 
                 example_id: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 score: Optional[float] = None):
        self.question = question
        self.answer = answer
        self.id = example_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.score = score
    
    def format(self, template: Optional[str] = None) -> str:
        """Format the example using the provided template or default format."""
        if template:
            return template.format(question=self.question, answer=self.answer)
        return f"Question: {self.question}\nAnswer: {self.answer}"
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FewShotExample':
        """Create a FewShotExample from a dictionary."""
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            example_id=data.get("id"),
            metadata=data.get("metadata", {}),
            score=data.get("score")
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata,
            "score": self.score
        }


class FewShotMemory(MemoryBase):
    """
    A memory system specialized for few-shot learning that stores and retrieves
    examples based on semantic similarity to optimize in-context learning.
    """
    
    def __init__(self, config: MemoryConfig = None):
        """
        Initialize the FewShotMemory with the given configuration.
        
        Args:
            config: Configuration for memory, vector store, embedding model, etc.
        """
        self.config = config or MemoryConfig()
        
        # Setup embedding model
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        
        # Setup vector store
        # Use a dedicated collection for few-shot examples
        self.config.vector_store.config.collection_name = "mem0_few_shot"
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, 
            self.config.vector_store.config
        )
        
        # Set default user ID if not specified
        self.user_id = get_user_id()
        
    def add_example(self, 
                   question: str, 
                   answer: str, 
                   user_id: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> str:
        """
        Add a new few-shot example to memory.
        
        Args:
            question: The example question/prompt
            answer: The corresponding answer/response
            user_id: User ID to associate with the example
            metadata: Additional metadata for the example
            
        Returns:
            The ID of the new example
        """
        # Prepare metadata
        metadata = metadata or {}
        user_id = user_id or self.user_id
        
        # Create payload
        payload = {
            "user_id": user_id,
            "memory_type": MemoryType.PROCEDURAL.value,
            "question": question,
            "answer": answer,
            "created_at": datetime.now(pytz.timezone("UTC")).isoformat(),
            # Store the user-provided metadata in a separate field
            "metadata": metadata
        }
        
        # Generate embedding for the question (we search based on question similarity)
        question_embedding = self.embedding_model.embed(question, "add")
        
        # Create a unique ID
        example_id = str(uuid.uuid4())
        
        # Insert into vector store
        self.vector_store.insert(
            vectors=[question_embedding],
            ids=[example_id],
            payloads=[payload],
        )
        
        logger.info(f"Added few-shot example with ID {example_id}")
        return example_id
    
    def get_examples(self, 
                    query: str, 
                    limit: int = 5, 
                    threshold: float = 0.7,
                    user_id: Optional[str] = None,
                    filters: Optional[Dict] = None) -> List[FewShotExample]:
        """
        Retrieve few-shot examples most relevant to the query.
        
        Args:
            query: The query to find examples for
            limit: Maximum number of examples to return
            threshold: Similarity threshold (0-1)
            user_id: Filter examples by user ID
            filters: Additional filters to apply
            
        Returns:
            List of FewShotExample objects sorted by relevance
        """
        # Prepare filters
        search_filters = filters or {}
        if user_id:
            search_filters["user_id"] = user_id
            
        # Process metadata filters - convert metadata.field to field for proper filtering
        processed_filters = {}
        for key, value in search_filters.items():
            if key.startswith("metadata."):
                # Extract the field name after "metadata."
                field_name = key.split(".", 1)[1]
                processed_filters[field_name] = value
            else:
                processed_filters[key] = value
        
        # Generate embedding for query
        query_embedding = self.embedding_model.embed(query, "search")
        
        # Search vector store for similar questions
        results = self.vector_store.search(
            query=query,
            vectors=query_embedding,
            limit=limit,
            filters=processed_filters
        )
        
        # Convert results to FewShotExample objects
        examples = []
        for item in results:
            # Skip examples below threshold
            if hasattr(item, "score") and item.score is not None and item.score > threshold:
                continue
                
            payload = item.payload
            example = FewShotExample(
                question=payload.get("question", ""),
                answer=payload.get("answer", ""),
                example_id=item.id,
                metadata={k: v for k, v in payload.items() 
                         if k not in ["question", "answer", "user_id", "memory_type", "created_at"]},
                score=item.score
            )
            examples.append(example)
            
        return examples
    
    def get_formatted_examples(self, 
                              query: str, 
                              limit: int = 5,
                              template: Optional[str] = None,
                              **kwargs) -> List[str]:
        """
        Get formatted few-shot examples for the query.
        
        Args:
            query: The query to find examples for
            limit: Maximum number of examples
            template: Optional format template with {question} and {answer} placeholders
            **kwargs: Additional arguments for get_examples
            
        Returns:
            List of formatted example strings
        """
        examples = self.get_examples(query, limit, **kwargs)
        return [example.format(template) for example in examples]
    
    def get_examples_as_context(self, 
                               query: str,
                               limit: int = 5,
                               preamble: str = "Here are some examples that might help:",
                               template: Optional[str] = None,
                               **kwargs) -> str:
        """
        Get formatted few-shot examples as a single context string.
        
        Args:
            query: The query to find examples for
            limit: Maximum number of examples
            preamble: Text to introduce the examples
            template: Optional format template
            **kwargs: Additional arguments for get_examples
            
        Returns:
            Formatted context string with examples
        """
        examples = self.get_formatted_examples(query, limit, template, **kwargs)
        if not examples:
            return ""
            
        return f"{preamble}\n\n" + "\n\n".join(examples)
    
    def delete_example(self, example_id: str) -> bool:
        """
        Delete a specific example by ID.
        
        Args:
            example_id: ID of the example to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete(vector_id=example_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete example {example_id}: {e}")
            return False
    
    def get(self, example_id: str) -> Optional[FewShotExample]:
        """
        Retrieve a specific example by ID.
        
        Args:
            example_id: ID of the example to retrieve
            
        Returns:
            FewShotExample if found, None otherwise
        """
        try:
            result = self.vector_store.get(vector_id=example_id)
            if result and hasattr(result, "payload"):
                payload = result.payload
                return FewShotExample(
                    question=payload.get("question", ""),
                    answer=payload.get("answer", ""),
                    example_id=result.id,
                    metadata=payload.get("metadata", {}),
                )
        except Exception as e:
            logger.error(f"Failed to retrieve example {example_id}: {e}")
        return None
    
    def update(self, example_id: str, question: Optional[str] = None, 
               answer: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing example.
        
        Args:
            example_id: ID of the example to update
            question: New question text (if updating)
            answer: New answer text (if updating)
            metadata: New metadata (if updating)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing example
            existing = self.vector_store.get(vector_id=example_id)
            if not existing:
                logger.error(f"Example {example_id} not found")
                return False
                
            payload = existing.payload.copy()
            
            # Update fields if provided
            if question is not None:
                payload["question"] = question
                # We need to update the vector for the new question
                question_embedding = self.embedding_model.embed(question, "update")
            else:
                question_embedding = None
                
            if answer is not None:
                payload["answer"] = answer
                
            if metadata is not None:
                # Initialize metadata field if it doesn't exist
                if "metadata" not in payload:
                    payload["metadata"] = {}
                
                # Update metadata values
                for k, v in metadata.items():
                    payload["metadata"][k] = v
            
            # Update timestamp
            payload["updated_at"] = datetime.now(pytz.timezone("UTC")).isoformat()
            
            # Update in vector store
            self.vector_store.update(
                vector_id=example_id,
                vector=question_embedding,
                payload=payload
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to update example {example_id}: {e}")
            return False
    
    def get_all(self, user_id: Optional[str] = None, limit: int = 100) -> List[FewShotExample]:
        """
        Retrieve all few-shot examples, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of examples to return
            
        Returns:
            List of FewShotExample objects
        """
        filters = {"memory_type": MemoryType.PROCEDURAL.value}
        if user_id:
            filters["user_id"] = user_id
            
        result = self.vector_store.list(filters=filters, limit=limit)[0]
        
        examples = []
        for item in result:
            payload = item.payload
            example = FewShotExample(
                question=payload.get("question", ""),
                answer=payload.get("answer", ""),
                example_id=item.id,
                metadata=payload.get("metadata", {})
            )
            examples.append(example)
            
        return examples
    
    def delete_all(self, user_id: Optional[str] = None) -> bool:
        """
        Delete all examples, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filters = {"memory_type": MemoryType.PROCEDURAL.value}
            if user_id:
                filters["user_id"] = user_id
                
            # Get all matching examples
            examples = self.get_all(user_id)
            for example in examples:
                self.delete_example(example.id)
                
            return True
        except Exception as e:
            logger.error(f"Failed to delete all examples: {e}")
            return False
    
    def history(self, example_id: str) -> List[Dict]:
        """
        Get the history of changes for an example.
        
        Args:
            example_id: ID of the example
            
        Returns:
            List of history entries (empty for now as history isn't tracked for examples)
        """
        # History tracking not implemented for few-shot examples
        return []
    
    def search(self, query: str, limit: int = 5, user_id: Optional[str] = None, **kwargs) -> List[FewShotExample]:
        """
        Search for examples matching the query (alias for get_examples).
        
        Args:
            query: The search query
            limit: Maximum number of results
            user_id: Optional user ID to filter by
            **kwargs: Additional arguments for get_examples
            
        Returns:
            List of matching FewShotExample objects
        """
        return self.get_examples(query, limit, user_id=user_id, **kwargs)
    
    def reset(self) -> bool:
        """
        Reset the memory by deleting all examples.
        
        Returns:
            True if successful, False otherwise
        """
        return self.delete_all()
        
    def delete(self, memory_id):
        """
        Delete a memory by ID. Required implementation for MemoryBase abstract method.
        
        Args:
            memory_id (str): ID of the memory to delete.
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.delete_example(memory_id)