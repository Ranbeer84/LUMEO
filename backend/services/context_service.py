"""
Context Assembly Service - Format Retrieved Photos for LLM
Phase 3.3: Build Context Assembly Service

FIX APPLIED:
- build_context no longer injects "USER QUERY: {query}" into the photo block
  when include_system_prompt=False (the mode used by the chat endpoint).

  The chat endpoint calls:
      llm_service.generate_response(context=full_context, query=message)

  and generate_response already appends:
      "USER QUESTION: {query}\n\nASSISTANT:"

  Having the query appear twice confused local LLMs (especially smaller
  models) and was one of the root causes of the "wrong photo" responses —
  the model spent attention re-reading an already-answered question instead
  of grounding itself in the photo descriptions.

  When include_system_prompt=True (used by /api/search/context and other
  standalone calls) we still embed the query for context, since in that
  mode generate_response is not always the consumer.
"""

from typing import List, Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextService:
    """Assembles retrieved photos into structured context for LLM."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.chars_per_token = 4

    def build_context(
        self,
        retrieved_photos: List[Dict],
        query: str,
        include_system_prompt: bool = True
    ) -> str:
        """
        Build complete context for LLM.

        Args:
            retrieved_photos:     Photos from retrieval service
            query:                Original user query
            include_system_prompt: When True, prepends Lumeo system instructions
                                  AND embeds the query in the photo block.
                                  When False (chat-endpoint mode), omits both
                                  so the caller's LLM service handles them.
        """
        context_parts = []

        if include_system_prompt:
            context_parts.append(self._generate_system_prompt())
            # Only add USER QUERY here when we own the full prompt (standalone mode)
            context_parts.append(f"USER QUERY: {query}\n")

        # Retrieved photos section
        if not retrieved_photos:
            context_parts.append("No relevant photos found in the library.")
            return '\n'.join(context_parts)

        context_parts.append(f"RETRIEVED PHOTOS ({len(retrieved_photos)} results):\n")

        max_chars = self.max_tokens * self.chars_per_token
        current_chars = sum(len(p) for p in context_parts)

        for idx, photo in enumerate(retrieved_photos, 1):
            photo_desc = self._format_photo(idx, photo)
            photo_chars = len(photo_desc)

            if current_chars + photo_chars > max_chars:
                remaining = len(retrieved_photos) - idx + 1
                context_parts.append(
                    f"\n[{remaining} more photos omitted due to context length]"
                )
                logger.warning(f"Context truncated: included {idx - 1}/{len(retrieved_photos)} photos")
                break

            context_parts.append(photo_desc)
            current_chars += photo_chars

        context = '\n'.join(context_parts)

        estimated_tokens = len(context) // self.chars_per_token
        logger.info(f"✓ Context built: {len(retrieved_photos)} photos, ~{estimated_tokens} tokens")

        return context

    def _generate_system_prompt(self) -> str:
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
        parts = [f"PHOTO {index}:"]

        if photo.get('caption'):
            parts.append(f"  Caption: {photo['caption']}")

        people = photo.get('people', [])
        if people:
            parts.append(f"  People: {', '.join(people)}")

        emotion = photo.get('dominant_emotion')
        mood_score = photo.get('mood_score')
        if emotion:
            parts.append(f"  Emotion: {self._describe_mood(emotion, mood_score)}")

        scene_parts = []
        if photo.get('scene_type'):
            scene_parts.append(f"{photo['scene_type']} scene")
        if photo.get('location') and photo['location'] != 'unknown':
            scene_parts.append(f"at {photo['location']}")
        if photo.get('activity') and photo['activity'] != 'unknown':
            scene_parts.append(f"during {photo['activity']}")
        if scene_parts:
            parts.append(f"  Scene: {', '.join(scene_parts)}")

        objects = photo.get('objects', [])
        if objects:
            sorted_objects = sorted(objects, key=lambda x: x.get('confidence', 0), reverse=True)
            obj_list = []
            for obj in sorted_objects[:5]:
                obj_str = obj['label']
                if obj.get('color'):
                    obj_str += f" ({obj['color']})"
                obj_list.append(obj_str)
            parts.append(f"  Objects: {', '.join(obj_list)}")

        temporal = []
        if photo.get('date_taken'):
            try:
                date_str = photo['date_taken']
                if isinstance(date_str, str):
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    temporal.append(f"taken on {date.strftime('%B %d, %Y')}")
            except Exception:
                pass
        if photo.get('season'):
            temporal.append(f"in {photo['season']}")
        if photo.get('time_of_day'):
            temporal.append(f"during {photo['time_of_day']}")
        if temporal:
            parts.append(f"  When: {', '.join(temporal)}")

        if photo.get('match_reasons'):
            parts.append(f"  Relevance: {'; '.join(photo['match_reasons'])}")

        if 'similarity' in photo:
            parts.append(f"  Similarity: {photo['similarity']:.3f}")

        parts.append("")
        return '\n'.join(parts)

    def _describe_mood(self, emotion: str, mood_score: Optional[float]) -> str:
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
        return emotion

    def build_summary_context(
        self,
        retrieved_photos: List[Dict],
        summary_type: str = "general"
    ) -> str:
        if not retrieved_photos:
            return "No photos to summarize."

        context_parts = [f"PHOTO COLLECTION SUMMARY ({len(retrieved_photos)} photos):\n"]

        emotions   = {}
        people     = {}
        locations  = {}
        activities = {}
        seasons    = {}

        for photo in retrieved_photos:
            emotion = photo.get('dominant_emotion')
            if emotion:
                emotions[emotion] = emotions.get(emotion, 0) + 1

            for person in photo.get('people', []):
                people[person] = people.get(person, 0) + 1

            location = photo.get('location')
            if location and location != 'unknown':
                locations[location] = locations.get(location, 0) + 1

            activity = photo.get('activity')
            if activity and activity != 'unknown':
                activities[activity] = activities.get(activity, 0) + 1

            season = photo.get('season')
            if season and season != 'unknown':
                seasons[season] = seasons.get(season, 0) + 1

        if summary_type in ('people', 'general') and people:
            context_parts.append("PEOPLE:")
            for person, count in sorted(people.items(), key=lambda x: x[1], reverse=True)[:10]:
                context_parts.append(f"  - {person}: appears in {count} photos")
            context_parts.append("")

        if summary_type in ('emotional', 'general') and emotions:
            context_parts.append("EMOTIONS:")
            for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(retrieved_photos)) * 100
                context_parts.append(f"  - {emotion}: {count} photos ({pct:.1f}%)")
            context_parts.append("")

        if summary_type in ('temporal', 'general') and seasons:
            context_parts.append("SEASONS:")
            for season, count in sorted(seasons.items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"  - {season}: {count} photos")
            context_parts.append("")

        if summary_type == 'general':
            if locations:
                context_parts.append("TOP LOCATIONS:")
                for location, count in sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"  - {location}: {count} photos")
                context_parts.append("")

            if activities:
                context_parts.append("TOP ACTIVITIES:")
                for activity, count in sorted(activities.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"  - {activity}: {count} photos")
                context_parts.append("")

        return '\n'.join(context_parts)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // self.chars_per_token

    def truncate_context(self, context: str, max_tokens: int) -> str:
        max_chars = max_tokens * self.chars_per_token
        if len(context) <= max_chars:
            return context
        truncated = context[:max_chars] + "\n\n[Context truncated due to length limit]"
        logger.warning(f"Context truncated from {len(context)} to {len(truncated)} chars")
        return truncated


# Singleton
_context_service = None

def get_context_service():
    global _context_service
    if _context_service is None:
        _context_service = ContextService()
    return _context_service