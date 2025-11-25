"""
FAISS Vector Database for Face Recognition
==========================================

Manages face embeddings using FAISS (Facebook AI Similarity Search) for fast 
nearest neighbor search. Supports user enrollment, similarity search, and 
persistence.

Key Features:
- IndexFlatL2: Exact L2 distance search (converted to cosine similarity)
- User metadata storage (name, enrollment date, photo count)
- Database persistence (save/load from disk)
- Thread-safe operations
- Efficient batch operations

Mathematical Foundation:
-----------------------
For L2-normalized embeddings (||x|| = ||y|| = 1):
    L2 distance: d = ||x - y||²
    Cosine similarity: s = x·y = (2 - d²)/2
    
Therefore: similarity = 1 - (L2_distance² / 2)

Author: Face Recognition System
Date: 2024
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import threading


class VectorDatabase:
    """
    FAISS-based vector database for storing and searching face embeddings.
    
    Uses IndexFlatL2 for exact nearest neighbor search. All embeddings are
    L2-normalized, allowing L2 distance to be converted to cosine similarity.
    
    Attributes:
        embedding_dim (int): Dimension of face embeddings (default: 512)
        index (faiss.IndexFlatL2): FAISS index for similarity search
        user_ids (List[str]): List of user IDs corresponding to embeddings
        user_metadata (Dict): Additional user information (enrollment date, etc.)
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize FAISS vector database.
        
        Args:
            embedding_dim: Dimension of face embeddings (default: 512 for ArcFace)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
        self.user_ids: List[str] = []  # Maps index position to user ID
        self.user_metadata: Dict[str, Dict] = {}  # Additional user info
        self.lock = threading.Lock()  # Thread safety
        
        print(f"✓ Vector database initialized (dimension: {embedding_dim})")
    
    def add_user(self, user_id: str, embedding: np.ndarray, 
                 metadata: Optional[Dict] = None) -> bool:
        """
        Add a new user to the database with their face embedding.
        
        Args:
            user_id: Unique identifier for the user
            embedding: Face embedding vector (must be L2-normalized)
            metadata: Optional metadata (enrollment_date, photo_count, etc.)
        
        Returns:
            True if added successfully, False if user already exists
        """
        with self.lock:
            # Check if user already exists
            if user_id in self.user_ids:
                print(f"[WARNING] User '{user_id}' already exists in database")
                return False
            
            # Validate embedding
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {embedding.shape[0]}"
                )
            
            # Ensure embedding is L2-normalized
            norm = np.linalg.norm(embedding)
            if not np.isclose(norm, 1.0, atol=1e-5):
                print(f"[WARNING] Embedding not L2-normalized (norm: {norm:.6f}), normalizing...")
                embedding = embedding / norm
            
            # Add to FAISS index
            embedding_2d = embedding.reshape(1, -1).astype('float32')
            self.index.add(embedding_2d)
            
            # Store user ID and metadata
            self.user_ids.append(user_id)
            self.user_metadata[user_id] = metadata or {
                'enrollment_date': datetime.now().isoformat(),
                'embedding_norm': float(norm)
            }
            
            print(f"✓ Added user '{user_id}' to database (total users: {len(self.user_ids)})")
            return True
    
    def search(self, query_embedding: np.ndarray, k: int = 1, 
               threshold: float = 0.4) -> List[Tuple[str, float]]:
        """
        Search for the k nearest neighbors in the database.
        
        Args:
            query_embedding: Face embedding to search for (must be L2-normalized)
            k: Number of nearest neighbors to return
            threshold: Minimum similarity threshold (0-1)
        
        Returns:
            List of (user_id, similarity) tuples, sorted by similarity (descending)
        """
        with self.lock:
            # Handle empty database
            if self.index.ntotal == 0:
                return []
            
            # Validate embedding
            if query_embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {query_embedding.shape[0]}"
                )
            
            # Ensure L2-normalized
            norm = np.linalg.norm(query_embedding)
            if not np.isclose(norm, 1.0, atol=1e-5):
                query_embedding = query_embedding / norm
            
            # Search FAISS index
            query_2d = query_embedding.reshape(1, -1).astype('float32')
            k_actual = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_2d, k_actual)
            
            # Convert L2 distances to cosine similarities
            # For L2-normalized vectors: similarity = 1 - (L2_distance² / 2)
            similarities = 1 - (distances[0] ** 2) / 2
            
            # Filter by threshold and prepare results
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if similarity >= threshold:
                    user_id = self.user_ids[idx]
                    results.append((user_id, float(similarity)))
            
            return results
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from the database.
        
        Note: FAISS IndexFlatL2 doesn't support direct deletion, so we rebuild
        the index without the deleted user. This is efficient for small-medium
        databases but may be slow for very large databases.
        
        Args:
            user_id: User ID to delete
        
        Returns:
            True if deleted successfully, False if user not found
        """
        with self.lock:
            # Find user index
            if user_id not in self.user_ids:
                print(f"[WARNING] User '{user_id}' not found in database")
                return False
            
            user_idx = self.user_ids.index(user_id)
            
            # Extract all embeddings except the one to delete
            embeddings = []
            for i in range(self.index.ntotal):
                if i != user_idx:
                    embedding = self.index.reconstruct(i)
                    embeddings.append(embedding)
            
            # Rebuild index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if embeddings:
                embeddings_array = np.vstack(embeddings).astype('float32')
                self.index.add(embeddings_array)
            
            # Remove from user lists
            self.user_ids.pop(user_idx)
            del self.user_metadata[user_id]
            
            print(f"✓ Deleted user '{user_id}' from database (remaining users: {len(self.user_ids)})")
            return True
    
    def update_user(self, user_id: str, new_embedding: np.ndarray, 
                    metadata: Optional[Dict] = None) -> bool:
        """
        Update a user's embedding and metadata.
        
        Args:
            user_id: User ID to update
            new_embedding: New face embedding
            metadata: Optional new metadata
        
        Returns:
            True if updated successfully, False if user not found
        """
        # Delete and re-add
        if user_id not in self.user_ids:
            return False
        
        old_metadata = self.user_metadata[user_id].copy()
        self.delete_user(user_id)
        
        # Merge metadata
        if metadata:
            old_metadata.update(metadata)
        old_metadata['last_updated'] = datetime.now().isoformat()
        
        return self.add_user(user_id, new_embedding, old_metadata)
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific user.
        
        Args:
            user_id: User ID to query
        
        Returns:
            User metadata dictionary, or None if not found
        """
        return self.user_metadata.get(user_id)
    
    def get_all_users(self) -> List[Dict[str, any]]:
        """
        Get information about all users in the database.
        
        Returns:
            List of dictionaries containing user_id and metadata
        """
        with self.lock:
            users = []
            for user_id in self.user_ids:
                users.append({
                    'user_id': user_id,
                    **self.user_metadata[user_id]
                })
            return users
    
    def get_user_count(self) -> int:
        """Get total number of users in database."""
        return len(self.user_ids)
    
    def clear(self):
        """Clear all users from the database."""
        with self.lock:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.user_ids = []
            self.user_metadata = {}
            print("✓ Database cleared")
    
    def save(self, filepath: str):
        """
        Save database to disk.
        
        Args:
            filepath: Path to save the database (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.index')))
        
        # Save metadata
        metadata = {
            'user_ids': self.user_ids,
            'user_metadata': self.user_metadata,
            'embedding_dim': self.embedding_dim
        }
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Database saved to {filepath}")
    
    def load(self, filepath: str) -> bool:
        """
        Load database from disk.
        
        Args:
            filepath: Path to load the database from (without extension)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = Path(filepath)
        index_file = filepath.with_suffix('.index')
        metadata_file = filepath.with_suffix('.pkl')
        
        if not index_file.exists() or not metadata_file.exists():
            print(f"[ERROR] Database files not found: {filepath}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.user_ids = metadata['user_ids']
            self.user_metadata = metadata['user_metadata']
            self.embedding_dim = metadata['embedding_dim']
            
            print(f"✓ Database loaded from {filepath} ({len(self.user_ids)} users)")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to load database: {e}")
            return False
    
    def __len__(self) -> int:
        """Return number of users in database."""
        return len(self.user_ids)
    
    def __repr__(self) -> str:
        """String representation of database."""
        return (f"VectorDatabase(users={len(self.user_ids)}, "
                f"dimension={self.embedding_dim})")


# Example usage
if __name__ == "__main__":
    # Create database
    db = VectorDatabase(embedding_dim=512)
    
    # Add users
    user1_embedding = np.random.randn(512).astype('float32')
    user1_embedding /= np.linalg.norm(user1_embedding)  # L2 normalize
    db.add_user("Alice", user1_embedding, {'department': 'Engineering'})
    
    user2_embedding = np.random.randn(512).astype('float32')
    user2_embedding /= np.linalg.norm(user2_embedding)
    db.add_user("Bob", user2_embedding, {'department': 'Sales'})
    
    # Search
    query = user1_embedding + np.random.randn(512) * 0.01  # Slightly perturbed
    query /= np.linalg.norm(query)
    results = db.search(query, k=2, threshold=0.3)
    print(f"\nSearch results: {results}")
    
    # Save and load
    db.save("data/test_db")
    
    db2 = VectorDatabase()
    db2.load("data/test_db")
    print(f"\nLoaded database: {db2}")
    print(f"All users: {db2.get_all_users()}")
