"""
Query Parser - Natural Language to Structured Filters
Phase 3.4: Create Natural Language Query Parser

Extracts entities and intent from user queries and converts
them into database filters for hybrid search.
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import relativedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryParser:
    """
    Parse natural language queries into structured filters
    
    Handles:
    - People names ("with Mom", "showing Dad and Sister")
    - Emotions ("happy", "smiling", "sad")
    - Colors ("red dress", "blue shirt", "wearing black")
    - Clothing ("t-shirt", "dress", "jacket")
    - Temporal ("last summer", "yesterday", "2023")
    - Locations ("at beach", "in office", "park")
    - Objects ("with cake", "holding phone")
    """
    
    def __init__(self, known_people: Optional[List[str]] = None):
        """
        Initialize query parser
        
        Args:
            known_people: List of known person names (cluster names)
        """
        self.known_people = known_people or []
        
        # Emotion synonyms
        self.emotion_map = {
            'happy': ['happy', 'smiling', 'joyful', 'cheerful', 'delighted', 'grinning', 'laughing'],
            'sad': ['sad', 'crying', 'unhappy', 'upset', 'down', 'depressed', 'tearful'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'fear': ['scared', 'afraid', 'fearful', 'worried', 'anxious'],
            'neutral': ['neutral', 'calm', 'expressionless']
        }
        
        # Color patterns
        self.colors = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown', 'beige', 'navy',
            'teal', 'maroon', 'gold', 'silver'
        ]
        
        # Clothing items
        self.clothing = [
            'shirt', 't-shirt', 'tshirt', 'dress', 'jacket', 'coat',
            'pants', 'jeans', 'shorts', 'skirt', 'suit', 'tie',
            'hat', 'cap', 'shoes', 'boots', 'sweater', 'hoodie'
        ]
        
        # Location keywords
        self.locations = {
            'beach': ['beach', 'ocean', 'sea', 'coast', 'shore', 'sand'],
            'mountain': ['mountain', 'hiking', 'trail', 'peak', 'climb'],
            'park': ['park', 'playground', 'garden'],
            'home': ['home', 'house', 'living room', 'bedroom', 'kitchen'],
            'office': ['office', 'work', 'desk', 'workplace'],
            'restaurant': ['restaurant', 'cafe', 'dining', 'bar'],
            'city': ['city', 'urban', 'downtown', 'street'],
            'indoor': ['indoor', 'inside', 'interior'],
            'outdoor': ['outdoor', 'outside', 'exterior']
        }
        
        # Common objects
        self.objects = [
            'cake', 'pizza', 'food', 'wine', 'beer', 'coffee',
            'car', 'bike', 'bicycle', 'motorcycle',
            'laptop', 'phone', 'computer', 'camera',
            'dog', 'cat', 'pet', 'animal',
            'book', 'ball', 'umbrella', 'bag', 'backpack'
        ]
        
        # Time of day
        self.times_of_day = {
            'morning': ['morning', 'dawn', 'sunrise', 'am'],
            'afternoon': ['afternoon', 'noon', 'midday', 'pm'],
            'evening': ['evening', 'sunset', 'dusk'],
            'night': ['night', 'nighttime', 'dark']
        }
        
        # Seasons
        self.seasons = ['spring', 'summer', 'autumn', 'fall', 'winter']
    
    def parse(self, query: str) -> Dict:
        """
        Parse natural language query into structured filters
        
        Args:
            query: User's natural language query
        
        Returns:
            Dict with extracted filters:
                - people: List[str]
                - emotions: List[str]
                - objects: List[str]
                - colors: List[str]
                - clothing: List[str]
                - scene_type: str (indoor/outdoor)
                - location: str
                - time_of_day: str
                - season: str
                - date_range: Tuple[datetime, datetime]
                - raw_query: str (original query for semantic search)
        """
        logger.info(f"Parsing query: '{query}'")
        
        query_lower = query.lower()
        
        filters = {
            'people': self._extract_people(query_lower),
            'emotions': self._extract_emotions(query_lower),
            'objects': self._extract_objects(query_lower),
            'colors': self._extract_colors(query_lower),
            'clothing': self._extract_clothing(query_lower),
            'scene_type': self._extract_scene_type(query_lower),
            'location': self._extract_location(query_lower),
            'time_of_day': self._extract_time_of_day(query_lower),
            'season': self._extract_season(query_lower),
            'date_range': self._extract_date_range(query_lower),
            'raw_query': query
        }
        
        # Remove None and empty values
        filters = {k: v for k, v in filters.items() if v}
        
        # Log extracted filters
        if len(filters) > 1:  # More than just raw_query
            logger.info(f"✓ Extracted filters: {filters}")
        else:
            logger.info("✓ No specific filters extracted, using semantic search only")
        
        return filters
    
    def _extract_people(self, query: str) -> Optional[List[str]]:
        """Extract people names from query"""
        if not self.known_people:
            return None
        
        found_people = []
        
        for person in self.known_people:
            # Case-insensitive match
            if person.lower() in query:
                found_people.append(person)
        
        # Also check for common patterns
        # "with X", "showing X", "of X"
        patterns = [
            r'with\s+(\w+)',
            r'showing\s+(\w+)',
            r'of\s+(\w+)',
            r'featuring\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Check if match is in known people
                for person in self.known_people:
                    if match.lower() == person.lower():
                        if person not in found_people:
                            found_people.append(person)
        
        return found_people if found_people else None
    
    def _extract_emotions(self, query: str) -> Optional[List[str]]:
        """Extract emotions from query"""
        found_emotions = []
        
        for emotion, synonyms in self.emotion_map.items():
            for synonym in synonyms:
                if synonym in query:
                    if emotion not in found_emotions:
                        found_emotions.append(emotion)
                    break
        
        return found_emotions if found_emotions else None
    
    def _extract_colors(self, query: str) -> Optional[List[str]]:
        """Extract colors from query"""
        found_colors = []
        
        for color in self.colors:
            # Match whole word only
            if re.search(r'\b' + color + r'\b', query):
                found_colors.append(color)
        
        return found_colors if found_colors else None
    
    def _extract_clothing(self, query: str) -> Optional[List[str]]:
        """Extract clothing items from query"""
        found_clothing = []
        
        for item in self.clothing:
            if item in query:
                found_clothing.append(item)
        
        return found_clothing if found_clothing else None
    
    def _extract_objects(self, query: str) -> Optional[List[str]]:
        """Extract objects from query"""
        found_objects = []
        
        for obj in self.objects:
            if re.search(r'\b' + obj + r'\b', query):
                found_objects.append(obj)
        
        return found_objects if found_objects else None
    
    def _extract_scene_type(self, query: str) -> Optional[str]:
        """Extract indoor/outdoor from query"""
        if 'outdoor' in query or 'outside' in query:
            return 'outdoor'
        elif 'indoor' in query or 'inside' in query:
            return 'indoor'
        return None
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query"""
        for location, keywords in self.locations.items():
            for keyword in keywords:
                if keyword in query:
                    return location
        return None
    
    def _extract_time_of_day(self, query: str) -> Optional[str]:
        """Extract time of day from query"""
        for time, keywords in self.times_of_day.items():
            for keyword in keywords:
                if keyword in query:
                    return time
        return None
    
    def _extract_season(self, query: str) -> Optional[str]:
        """Extract season from query"""
        for season in self.seasons:
            if season in query:
                return season
        return None
    
    def _extract_date_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Extract date range from temporal expressions
        
        Handles:
        - "yesterday", "today"
        - "last week", "last month", "last year"
        - "2023", "December 2023"
        - "this summer", "last winter"
        """
        today = datetime.now()
        
        # Yesterday
        if 'yesterday' in query:
            start = today - timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
            return (start, end)
        
        # Today
        if 'today' in query:
            start = today.replace(hour=0, minute=0, second=0, microsecond=0)
            end = today.replace(hour=23, minute=59, second=59)
            return (start, end)
        
        # Last week
        if 'last week' in query or 'past week' in query:
            start = today - timedelta(days=7)
            return (start, today)
        
        # Last month
        if 'last month' in query or 'past month' in query:
            start = today - relativedelta.relativedelta(months=1)
            return (start, today)
        
        # Last year
        if 'last year' in query or 'past year' in query:
            start = today - relativedelta.relativedelta(years=1)
            return (start, today)
        
        # This year
        if 'this year' in query:
            start = datetime(today.year, 1, 1)
            return (start, today)
        
        # Specific year (e.g., "2023")
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31, 23, 59, 59)
            return (start, end)
        
        # Month + Year (e.g., "December 2023")
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in query:
                year_match = re.search(r'\b(20\d{2})\b', query)
                if year_match:
                    year = int(year_match.group(1))
                else:
                    year = today.year
                
                start = datetime(year, month_num, 1)
                # Last day of month
                if month_num == 12:
                    end = datetime(year, 12, 31, 23, 59, 59)
                else:
                    end = datetime(year, month_num + 1, 1) - timedelta(seconds=1)
                
                return (start, end)
        
        # Seasonal with "last" (e.g., "last summer")
        if 'last summer' in query:
            year = today.year if today.month >= 9 else today.year - 1
            start = datetime(year, 6, 1)
            end = datetime(year, 8, 31, 23, 59, 59)
            return (start, end)
        
        if 'last winter' in query:
            year = today.year if today.month >= 3 else today.year - 1
            start = datetime(year, 12, 1)
            end = datetime(year + 1, 2, 28, 23, 59, 59)
            return (start, end)
        
        return None
    
    def update_known_people(self, people: List[str]):
        """Update list of known people for name extraction"""
        self.known_people = people
        logger.info(f"Updated known people: {len(people)} names")
    
    def get_semantic_query(self, filters: Dict) -> str:
        """
        Convert structured filters back to semantic query
        
        Useful for combining with CLIP embeddings
        """
        parts = []
        
        if filters.get('people'):
            parts.append(f"with {', '.join(filters['people'])}")
        
        if filters.get('emotions'):
            parts.append(', '.join(filters['emotions']))
        
        if filters.get('location'):
            parts.append(f"at {filters['location']}")
        
        if filters.get('activity'):
            parts.append(f"during {filters['activity']}")
        
        if filters.get('objects'):
            parts.append(f"showing {', '.join(filters['objects'])}")
        
        if filters.get('colors'):
            parts.append(f"with {', '.join(filters['colors'])} colors")
        
        if filters.get('time_of_day'):
            parts.append(f"in the {filters['time_of_day']}")
        
        if filters.get('season'):
            parts.append(f"during {filters['season']}")
        
        return ' '.join(parts) if parts else filters.get('raw_query', '')


# Singleton instance
_query_parser = None

def get_query_parser(known_people: Optional[List[str]] = None):
    """Get or create query parser singleton"""
    global _query_parser
    if _query_parser is None:
        _query_parser = QueryParser(known_people)
    return _query_parser


# Example usage
if __name__ == "__main__":
    # Initialize with known people
    parser = QueryParser(known_people=['Mom', 'Dad', 'Sister', 'Abhigyan', 'Sara'])
    
    # Test queries
    test_queries = [
        "Show me photos with Mom at the beach",
        "Find happy photos from last summer",
        "Photos where I'm wearing a black t-shirt with Abhigyan",
        "Beach sunset photos from 2023",
        "Indoor party photos with cake",
        "Sad photos from last winter",
        "Morning photos at home",
        "Photos with my dog outdoors",
        "December 2023 photos with Sara",
        "Yesterday's photos"
    ]
    
    print("="*60)
    print("QUERY PARSER TEST")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        filters = parser.parse(query)
        
        # Show extracted filters (excluding raw_query)
        display_filters = {k: v for k, v in filters.items() if k != 'raw_query'}
        if display_filters:
            for key, value in display_filters.items():
                print(f"  {key}: {value}")
        else:
            print("  (no structured filters, semantic search only)")