"""
Embedding Service - Convert text and images to vector embeddings
Supports both text queries and photo captions for semantic search
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union
import time
from functools import lru_cache

class EmbeddingService:
    """
    Handles text and image embeddings for semantic search
    Uses sentence-transformers for text encoding
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service
        
        Args:
            model_name: Name of sentence-transformers model
                      - all-MiniLM-L6-v2: Fast, good quality (384 dims)
                      - all-mpnet-base-v2: Slower, best quality (768 dims)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.cache = {}  # Simple in-memory cache for repeated queries
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text query or caption
            use_cache: Whether to use cached embeddings
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Check cache first
        if use_cache and text in self.cache:
            return self.cache[text]
        
        # Generate embedding
        start_time = time.time()
        embedding = self.model.encode(text, convert_to_numpy=True)
        elapsed = time.time() - start_time
        
        # Cache result
        if use_cache:
            self.cache[text] = embedding
        
        print(f"Encoded text in {elapsed*1000:.2f}ms: '{text[:50]}...'")
        return embedding
    
    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts efficiently in batches
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        print(f"Encoding {len(texts)} texts in batches of {batch_size}")
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        
        elapsed = time.time() - start_time
        print(f"Batch encoding completed in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)")
        
        return embeddings
    
    def generate_photo_caption(self, photo_metadata: Dict) -> str:
        """
        Generate natural language caption from photo metadata
        This caption will be embedded for semantic search
        
        Args:
            photo_metadata: Dict with keys like:
                - people: List[str] - names of people
                - objects: List[str] - detected objects
                - scene: str - scene type (beach, office, etc.)
                - emotion: str - dominant emotion
                - colors: List[str] - dominant colors
                - time_of_day: str - morning, evening, etc.
                - season: str - spring, summer, etc.
                
        Returns:
            Natural language caption string
        """
        caption_parts = []
        
        # People
        people = photo_metadata.get('people', [])
        if people:
            if len(people) == 1:
                caption_parts.append(f"Photo of {people[0]}")
            elif len(people) == 2:
                caption_parts.append(f"Photo of {people[0]} and {people[1]}")
            else:
                caption_parts.append(f"Photo of {', '.join(people[:-1])}, and {people[-1]}")
        else:
            caption_parts.append("Photo")
        
        # Scene and location
        scene = photo_metadata.get('scene')
        if scene:
            caption_parts.append(f"at {scene}")
        
        # Emotion/mood
        emotion = photo_metadata.get('emotion')
        if emotion and emotion != 'neutral':
            caption_parts.append(f"feeling {emotion}")
        
        # Activity
        activity = photo_metadata.get('activity')
        if activity:
            caption_parts.append(f"during {activity}")
        
        # Objects
        objects = photo_metadata.get('objects', [])
        if objects:
            obj_str = ', '.join(objects[:3])  # Limit to top 3 objects
            caption_parts.append(f"with {obj_str}")
        
        # Colors
        colors = photo_metadata.get('colors', [])
        if colors:
            color_str = ', '.join(colors[:2])
            caption_parts.append(f"featuring {color_str} colors")
        
        # Time context
        time_of_day = photo_metadata.get('time_of_day')
        season = photo_metadata.get('season')
        if time_of_day and season:
            caption_parts.append(f"in the {time_of_day} during {season}")
        elif time_of_day:
            caption_parts.append(f"in the {time_of_day}")
        elif season:
            caption_parts.append(f"during {season}")
        
        # Join all parts
        caption = ' '.join(caption_parts)
        
        return caption
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            candidate_embeddings: Array of shape (n_candidates, embedding_dim)
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'index' and 'similarity' keys, sorted by similarity
        """
        if len(candidate_embeddings) == 0:
            return []
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize candidates
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Calculate similarities (dot product of normalized vectors = cosine similarity)
        similarities = np.dot(candidate_norms, query_norm)
        
        # Get top K indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = [
            {
                'index': int(idx),
                'similarity': float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        print("Embedding cache cleared")


# Global instance (lazy loaded)
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# Example usage
if __name__ == "__main__":
    # Test the embedding service
    service = EmbeddingService()
    
    # Test text encoding
    query = "beach sunset with family"
    query_emb = service.encode_text(query)
    print(f"Query embedding shape: {query_emb.shape}")
    
    # Test caption generation
    metadata = {
        'people': ['Mom', 'Dad', 'Sister'],
        'scene': 'beach',
        'emotion': 'happy',
        'objects': ['umbrella', 'sand', 'water'],
        'colors': ['blue', 'yellow'],
        'time_of_day': 'evening',
        'season': 'summer'
    }
    caption = service.generate_photo_caption(metadata)
    print(f"Generated caption: {caption}")
    
    # Test batch encoding
    captions = [
        "Photo of family at beach",
        "Indoor party with friends",
        "Mountain hiking adventure"
    ]
    embeddings = service.encode_texts_batch(captions)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = service.similarity(query_emb, embeddings[0])
    print(f"Similarity between query and first caption: {sim:.3f}")