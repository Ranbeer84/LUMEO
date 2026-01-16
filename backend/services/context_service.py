"""
Context Assembly Service - Format Retrieved Photos for LLM
Phase 3.3: Build Context Assembly Service

Takes retrieved photos and formats them into natural language
context for the LLM to generate responses.
"""

from typing import List, Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextService:
    """
    Assembles retrieved photos into structured context for LLM
    """
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize context service
        
        Args:
            max_tokens: Maximum context size in tokens (rough estimate)
        """
        self.max_tokens = max_tokens
        self.chars_per_token = 4  # Rough estimate: 1 token ≈ 4 characters
    
    def build_context(
        self,
        retrieved_photos: List[Dict],
        query: str,
        include_system_prompt: bool = True
    ) -> str:
        """
        Build complete context for LLM
        
        Args:
            retrieved_photos: List of photo dicts from retrieval service
            query: Original user query
            include_system_prompt: Whether to include system instructions
        
        Returns:
            Formatted context string ready for LLM
        """
        context_parts = []
        
        # System prompt (if requested)
        if include_system_prompt:
            system_prompt = self._generate_system_prompt()
            context_parts.append(system_prompt)
        
        # User query
        context_parts.append(f"USER QUERY: {query}\n")
        
        # Retrieved photos
        if not retrieved_photos:
            context_parts.append("No relevant photos found in the library.")
            return '\n'.join(context_parts)
        
        context_parts.append(f"RETRIEVED PHOTOS ({len(retrieved_photos)} results):\n")
        
        # Format each photo
        max_chars = self.max_tokens * self.chars_per_token
        current_chars = sum(len(p) for p in context_parts)
        
        for idx, photo in enumerate(retrieved_photos, 1):
            photo_desc = self._format_photo(idx, photo)
            photo_chars = len(photo_desc)
            
            # Check if adding this photo would exceed token limit
            if current_chars + photo_chars > max_chars:
                remaining = len(retrieved_photos) - idx + 1
                context_parts.append(f"\n[{remaining} more photos available but omitted due to context length]")
                logger.warning(f"Context truncated: included {idx-1}/{len(retrieved_photos)} photos")
                break
            
            context_parts.append(photo_desc)
            current_chars += photo_chars
        
        context = '\n'.join(context_parts)
        
        # Log stats
        estimated_tokens = len(context) // self.chars_per_token
        logger.info(f"✓ Context built: {len(retrieved_photos)} photos, ~{estimated_tokens} tokens")
        
        return context
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt with instructions for LLM"""
        return """You are Lumeo, an AI assistant for a photo memory system. You help users explore and understand their photo collection through natural conversation.

INSTRUCTIONS:
1. ONLY use information from the provided photos below
2. If you're unsure or the photos don't contain the answer, say "I don't have enough information"
3. Always reference specific photos by number (e.g., "Photo 1 shows...")
4. Be conversational and warm, like talking to a friend about memories
5. Point out interesting patterns, emotions, or moments in the photos
6. NEVER make up information that isn't in the provided context

---

"""
    
    def _format_photo(self, index: int, photo: Dict) -> str:
        """
        Format a single photo into natural language description
        
        Args:
            index: Photo number (for reference)
            photo: Photo dict with metadata
        
        Returns:
            Formatted photo description
        """
        parts = []
        
        # Header
        parts.append(f"PHOTO {index}:")
        
        # Caption (if available)
        if photo.get('caption'):
            parts.append(f"  Caption: {photo['caption']}")
        
        # People
        people = photo.get('people', [])
        if people:
            people_str = ', '.join(people)
            parts.append(f"  People: {people_str}")
        
        # Emotions
        emotion = photo.get('dominant_emotion')
        mood_score = photo.get('mood_score')
        if emotion:
            mood_desc = self._describe_mood(emotion, mood_score)
            parts.append(f"  Emotion: {mood_desc}")
        
        # Scene and location
        scene = photo.get('scene_type')
        location = photo.get('location')
        activity = photo.get('activity')
        
        scene_parts = []
        if scene:
            scene_parts.append(f"{scene} scene")
        if location and location != 'unknown':
            scene_parts.append(f"at {location}")
        if activity and activity != 'unknown':
            scene_parts.append(f"during {activity}")
        
        if scene_parts:
            parts.append(f"  Scene: {', '.join(scene_parts)}")
        
        # Objects
        objects = photo.get('objects', [])
        if objects:
            # Show top 5 most confident objects
            sorted_objects = sorted(objects, key=lambda x: x.get('confidence', 0), reverse=True)
            obj_list = []
            for obj in sorted_objects[:5]:
                obj_str = obj['label']
                if obj.get('color'):
                    obj_str += f" ({obj['color']})"
                obj_list.append(obj_str)
            
            parts.append(f"  Objects: {', '.join(obj_list)}")
        
        # Temporal context
        temporal = []
        if photo.get('date_taken'):
            try:
                # Parse ISO format date
                date_str = photo['date_taken']
                if isinstance(date_str, str):
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    temporal.append(f"taken on {date.strftime('%B %d, %Y')}")
            except:
                pass
        
        if photo.get('season'):
            temporal.append(f"in {photo['season']}")
        
        if photo.get('time_of_day'):
            temporal.append(f"during {photo['time_of_day']}")
        
        if temporal:
            parts.append(f"  When: {', '.join(temporal)}")
        
        # Match explanation (why this photo was retrieved)
        match_reasons = photo.get('match_reasons', [])
        if match_reasons:
            parts.append(f"  Relevance: {'; '.join(match_reasons)}")
        
        # Similarity score
        if 'similarity' in photo:
            parts.append(f"  Similarity: {photo['similarity']:.3f}")
        
        parts.append("")  # Blank line between photos
        
        return '\n'.join(parts)
    
    def _describe_mood(self, emotion: str, mood_score: Optional[float]) -> str:
        """Create natural language mood description"""
        if mood_score is not None:
            if mood_score > 0.5:
                intensity = "very"
            elif mood_score > 0.3:
                intensity = "moderately"
            elif mood_score > 0:
                intensity = "somewhat"
            elif mood_score > -0.3:
                intensity = "slightly"
            elif mood_score > -0.5:
                intensity = "moderately"
            else:
                intensity = "very"
            
            return f"{intensity} {emotion} (mood: {mood_score:.2f})"
        else:
            return emotion
    
    def build_summary_context(
        self,
        retrieved_photos: List[Dict],
        summary_type: str = "general"
    ) -> str:
        """
        Build aggregated summary context for insights/analysis
        
        Args:
            retrieved_photos: Photos to summarize
            summary_type: Type of summary ("general", "emotional", "temporal", "people")
        
        Returns:
            Summary context string
        """
        if not retrieved_photos:
            return "No photos to summarize."
        
        context_parts = []
        context_parts.append(f"PHOTO COLLECTION SUMMARY ({len(retrieved_photos)} photos):\n")
        
        # Aggregate statistics
        emotions = {}
        people = {}
        locations = {}
        activities = {}
        seasons = {}
        
        for photo in retrieved_photos:
            # Count emotions
            emotion = photo.get('dominant_emotion')
            if emotion:
                emotions[emotion] = emotions.get(emotion, 0) + 1
            
            # Count people
            for person in photo.get('people', []):
                people[person] = people.get(person, 0) + 1
            
            # Count locations
            location = photo.get('location')
            if location and location != 'unknown':
                locations[location] = locations.get(location, 0) + 1
            
            # Count activities
            activity = photo.get('activity')
            if activity and activity != 'unknown':
                activities[activity] = activities.get(activity, 0) + 1
            
            # Count seasons
            season = photo.get('season')
            if season and season != 'unknown':
                seasons[season] = seasons.get(season, 0) + 1
        
        # Format based on summary type
        if summary_type == "people" or summary_type == "general":
            if people:
                sorted_people = sorted(people.items(), key=lambda x: x[1], reverse=True)
                context_parts.append("PEOPLE:")
                for person, count in sorted_people[:10]:
                    context_parts.append(f"  - {person}: appears in {count} photos")
                context_parts.append("")
        
        if summary_type == "emotional" or summary_type == "general":
            if emotions:
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                context_parts.append("EMOTIONS:")
                for emotion, count in sorted_emotions:
                    percentage = (count / len(retrieved_photos)) * 100
                    context_parts.append(f"  - {emotion}: {count} photos ({percentage:.1f}%)")
                context_parts.append("")
        
        if summary_type == "temporal" or summary_type == "general":
            if seasons:
                sorted_seasons = sorted(seasons.items(), key=lambda x: x[1], reverse=True)
                context_parts.append("SEASONS:")
                for season, count in sorted_seasons:
                    context_parts.append(f"  - {season}: {count} photos")
                context_parts.append("")
        
        if summary_type == "general":
            if locations:
                sorted_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)
                context_parts.append("TOP LOCATIONS:")
                for location, count in sorted_locations[:5]:
                    context_parts.append(f"  - {location}: {count} photos")
                context_parts.append("")
            
            if activities:
                sorted_activities = sorted(activities.items(), key=lambda x: x[1], reverse=True)
                context_parts.append("TOP ACTIVITIES:")
                for activity, count in sorted_activities[:5]:
                    context_parts.append(f"  - {activity}: {count} photos")
                context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count"""
        return len(text) // self.chars_per_token
    
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit token limit"""
        max_chars = max_tokens * self.chars_per_token
        
        if len(context) <= max_chars:
            return context
        
        truncated = context[:max_chars]
        truncated += "\n\n[Context truncated due to length limit]"
        
        logger.warning(f"Context truncated from {len(context)} to {len(truncated)} chars")
        
        return truncated


# Singleton instance
_context_service = None

def get_context_service():
    """Get or create context service singleton"""
    global _context_service
    if _context_service is None:
        _context_service = ContextService()
    return _context_service


# Example usage
if __name__ == "__main__":
    service = ContextService()
    
    # Example photos
    photos = [
        {
            'photo_id': 'photo_1',
            'filename': 'beach.jpg',
            'caption': 'Photo of Mom and Dad at beach looking happy during summer',
            'people': ['Mom', 'Dad'],
            'dominant_emotion': 'happy',
            'mood_score': 0.8,
            'scene_type': 'outdoor',
            'location': 'beach',
            'objects': [
                {'label': 'umbrella', 'confidence': 0.9, 'color': 'blue'},
                {'label': 'sand', 'confidence': 0.95, 'color': 'beige'}
            ],
            'season': 'summer',
            'time_of_day': 'afternoon',
            'date_taken': '2023-07-15T14:30:00',
            'similarity': 0.87,
            'match_reasons': ['High semantic match (0.87)', 'Emotion: happy', 'Scene: outdoor']
        }
    ]
    
    # Build context
    context = service.build_context(photos, "Show me happy beach photos")
    print(context)
    print(f"\nEstimated tokens: {service.estimate_tokens(context)}")
    
    # Build summary
    summary = service.build_summary_context(photos, summary_type="general")
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(summary)