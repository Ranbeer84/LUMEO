"""
Query Service - Text-to-Vector Query Engine
Phase 3.1: Build Text-to-Vector Query Engine

Converts natural language queries into embeddings and generates
photo captions for semantic search.
"""

from .clip_service import get_clip_service
import numpy as np
import logging
from typing import Dict, List, Optional
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:
    """
    Handles query processing and caption generation for semantic search
    """
    
    def __init__(self):
        self.clip_service = get_clip_service()
        self._query_cache = {}  # Simple cache for repeated queries
    
    def encode_query(self, query_text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Convert text query to CLIP embedding
        
        Args:
            query_text: Natural language query (e.g., "beach sunset with family")
            use_cache: Whether to cache the embedding
        
        Returns:
            512-dimensional embedding vector or None
        """
        # Normalize query
        normalized_query = query_text.lower().strip()
        
        # Check cache
        if use_cache and normalized_query in self._query_cache:
            logger.info(f"✓ Query cache hit: '{normalized_query}'")
            return self._query_cache[normalized_query]
        
        # Generate embedding
        logger.info(f"Encoding query: '{query_text}'")
        embedding = self.clip_service.encode_text(query_text)
        
        if embedding is None:
            logger.error("Failed to encode query")
            return None
        
        # Cache it
        if use_cache:
            self._query_cache[normalized_query] = embedding
        
        logger.info(f"✓ Query encoded: {embedding.shape}")
        return embedding
    
    def generate_photo_caption(self, photo_data: Dict) -> str:
        """
        Generate natural language caption from photo metadata
        
        This caption describes what's in the photo and will be used
        for semantic search and LLM context.
        
        Args:
            photo_data: Dict containing:
                - people: List[str] - cluster names
                - emotions: List[str] - detected emotions
                - objects: List[str] - detected object labels
                - scene_type: str - indoor/outdoor
                - location_type: str - beach, office, etc.
                - activity: str - party, sports, etc.
                - dominant_colors: List[str] - color names
                - time_of_day: str - morning, evening, etc.
                - season: str - winter, summer, etc.
                - date_taken: datetime - when photo was taken
        
        Returns:
            Natural language caption string
        """
        parts = []
        
        # Start with people
        people = photo_data.get('people', [])
        if people:
            if len(people) == 1:
                parts.append(f"Photo of {people[0]}")
            elif len(people) == 2:
                parts.append(f"Photo of {people[0]} and {people[1]}")
            elif len(people) == 3:
                parts.append(f"Photo of {people[0]}, {people[1]}, and {people[2]}")
            else:
                parts.append(f"Photo of {', '.join(people[:2])}, and {len(people)-2} others")
        else:
            parts.append("Photo")
        
        # Add emotions
        emotions = photo_data.get('emotions', [])
        if emotions:
            emotion_str = self._format_emotions(emotions)
            if emotion_str:
                parts.append(emotion_str)
        
        # Add location and scene
        location = photo_data.get('location_type')
        scene = photo_data.get('scene_type')
        
        if location and location != 'unknown':
            parts.append(f"at {location}")
        elif scene:
            parts.append(f"in {scene} setting")
        
        # Add activity
        activity = photo_data.get('activity')
        if activity and activity != 'unknown':
            parts.append(f"during {activity}")
        
        # Add objects (select most interesting ones)
        objects = photo_data.get('objects', [])
        if objects:
            interesting_objects = self._filter_interesting_objects(objects)
            if interesting_objects:
                obj_str = ', '.join(interesting_objects[:3])
                parts.append(f"with {obj_str}")
        
        # Add colors
        colors = photo_data.get('dominant_colors', [])
        if colors and len(colors) > 0:
            color_str = ', '.join(colors[:2])
            parts.append(f"featuring {color_str} tones")
        
        # Add temporal context
        temporal = self._format_temporal(
            photo_data.get('time_of_day'),
            photo_data.get('season'),
            photo_data.get('date_taken')
        )
        if temporal:
            parts.append(temporal)
        
        caption = ' '.join(parts) + '.'
        
        return caption
    
    def _format_emotions(self, emotions: List[str]) -> str:
        """Format emotion list into natural language"""
        if not emotions:
            return ""
        
        # Map emotions to descriptive phrases
        emotion_phrases = {
            'happy': 'looking happy',
            'sad': 'looking sad',
            'angry': 'looking angry',
            'surprise': 'looking surprised',
            'fear': 'looking worried',
            'disgust': 'looking displeased',
            'neutral': 'with neutral expressions'
        }
        
        unique_emotions = list(set(emotions))
        
        if len(unique_emotions) == 1:
            return emotion_phrases.get(unique_emotions[0], '')
        elif 'happy' in unique_emotions:
            return 'smiling and joyful'
        else:
            return 'with mixed emotions'
    
    def _filter_interesting_objects(self, objects: List[str]) -> List[str]:
        """
        Filter to most interesting/meaningful objects
        
        Skip generic objects like 'person', focus on contextual items
        """
        # Skip these common/generic objects
        skip_objects = {'person', 'man', 'woman', 'child', 'people'}
        
        # Prioritize these interesting categories
        priority_objects = {
            'cake', 'pizza', 'wine', 'food', 'dining',  # Food/dining
            'car', 'motorcycle', 'bicycle', 'bus', 'train',  # Vehicles
            'laptop', 'phone', 'tv', 'keyboard',  # Electronics
            'umbrella', 'backpack', 'suitcase',  # Items
            'dog', 'cat', 'bird', 'horse',  # Animals
            'couch', 'chair', 'table', 'bed'  # Furniture
        }
        
        interesting = []
        for obj in objects:
            obj_lower = obj.lower()
            if obj_lower not in skip_objects:
                interesting.append(obj)
        
        # Sort by priority
        interesting.sort(key=lambda x: 0 if x.lower() in priority_objects else 1)
        
        return interesting
    
    def _format_temporal(self, time_of_day: Optional[str], season: Optional[str], date_taken) -> str:
        """Format temporal context"""
        parts = []
        
        if time_of_day and time_of_day != 'unknown':
            parts.append(f"in the {time_of_day}")
        
        if season and season != 'unknown':
            parts.append(f"during {season}")
        
        if parts:
            return ' '.join(parts)
        
        # Fallback to date if available
        if date_taken:
            try:
                if hasattr(date_taken, 'strftime'):
                    return f"taken on {date_taken.strftime('%B %d, %Y')}"
            except:
                pass
        
        return ""
    
    def generate_search_keywords(self, query_text: str) -> List[str]:
        """
        Extract searchable keywords from query
        
        Args:
            query_text: Natural language query
        
        Returns:
            List of keyword strings
        """
        # Simple keyword extraction (can be enhanced with NLP)
        words = query_text.lower().split()
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'of', 'by', 'from', 'show', 'me', 'find', 'get'}
        
        keywords = [w.strip('.,!?;:') for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def expand_query(self, query_text: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query_text: Original query
        
        Returns:
            List of expanded query variations
        """
        # Simple synonym expansion (can be enhanced with WordNet or embeddings)
        synonyms = {
            'beach': ['beach', 'ocean', 'sea', 'coast', 'shore', 'sand'],
            'happy': ['happy', 'joyful', 'smiling', 'cheerful', 'delighted'],
            'sad': ['sad', 'unhappy', 'crying', 'upset', 'down'],
            'party': ['party', 'celebration', 'gathering', 'event', 'festive'],
            'outdoor': ['outdoor', 'outside', 'nature', 'open air'],
            'indoor': ['indoor', 'inside', 'interior'],
            'family': ['family', 'relatives', 'loved ones', 'kin'],
            'friend': ['friend', 'buddy', 'pal', 'companion'],
            'food': ['food', 'meal', 'dining', 'eating', 'restaurant'],
            'work': ['work', 'office', 'professional', 'business', 'job']
        }
        
        expanded = [query_text]
        
        # Check for synonym matches
        for key, variants in synonyms.items():
            if key in query_text.lower():
                for variant in variants[:3]:  # Limit to 3 variants
                    expanded_query = query_text.lower().replace(key, variant)
                    if expanded_query != query_text.lower():
                        expanded.append(expanded_query)
        
        return expanded[:5]  # Return up to 5 variations
    
    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
        logger.info("✓ Query cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_queries': len(self._query_cache),
            'cache_size_mb': sum(
                q.nbytes for q in self._query_cache.values()
            ) / (1024 * 1024)
        }


# Singleton instance
_query_service = None

def get_query_service():
    """Get or create query service singleton"""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


# Example usage
if __name__ == "__main__":
    service = QueryService()
    
    # Test query encoding
    query = "beach sunset with family"
    embedding = service.encode_query(query)
    print(f"Query embedding shape: {embedding.shape}")
    
    # Test caption generation
    photo_data = {
        'people': ['Mom', 'Dad', 'Sister'],
        'emotions': ['happy', 'happy', 'happy'],
        'objects': ['umbrella', 'sand', 'water', 'person'],
        'scene_type': 'outdoor',
        'location_type': 'beach',
        'dominant_colors': ['blue', 'yellow'],
        'time_of_day': 'evening',
        'season': 'summer'
    }
    
    caption = service.generate_photo_caption(photo_data)
    print(f"Caption: {caption}")
    
    # Test keyword extraction
    keywords = service.generate_search_keywords(query)
    print(f"Keywords: {keywords}")
    
    # Test query expansion
    expanded = service.expand_query(query)
    print(f"Expanded queries: {expanded}")