"""
Memory Service - Advanced Memory Management for Phase 5
Handles long-term memory, context optimization, and smart recall
"""

from typing import List, Dict, Optional, Set
import logging
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryService:
    """
    Advanced memory management for conversations
    
    Features:
    - Long-term memory tracking
    - Topic extraction and clustering
    - Smart context recall
    - Memory compression
    """
    
    def __init__(self):
        self.topic_cache = {}  # Cache for topic extraction
    
    def extract_conversation_topics(
        self,
        session,
        conversation_id: str
    ) -> List[str]:
        """
        Extract main topics from conversation
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
        
        Returns:
            List of topic strings
        """
        from models import Message
        
        messages = session.query(Message).filter_by(
            conversation_id=conversation_id
        ).all()
        
        if not messages:
            return []
        
        # Extract topics from user queries and photo metadata
        topics = set()
        
        for msg in messages:
            if msg.role == 'user':
                # Simple keyword extraction (can be enhanced with NLP)
                keywords = self._extract_keywords(msg.content)
                topics.update(keywords)
            
            # Extract from retrieved photo metadata
            if msg.retrieved_photo_ids:
                metadata = msg.metadata or {}
                if 'filters' in metadata:
                    filters = metadata['filters']
                    
                    # Add filter values as topics
                    for key, value in filters.items():
                        if key != 'raw_query' and value:
                            if isinstance(value, list):
                                topics.update(str(v).lower() for v in value)
                            else:
                                topics.add(str(value).lower())
        
        return list(topics)
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Simple keyword extraction"""
        
        # Remove common stop words
        stop_words = {
            'show', 'me', 'find', 'get', 'the', 'a', 'an', 'in', 'on', 'at',
            'to', 'for', 'with', 'from', 'by', 'about', 'as', 'of', 'my', 'i'
        }
        
        # Extract words
        words = text.lower().split()
        keywords = {
            w.strip('.,!?;:') 
            for w in words 
            if len(w) > 3 and w not in stop_words
        }
        
        return keywords
    
    def get_relevant_memories(
        self,
        session,
        current_query: str,
        user_id: str = "default_user",
        limit: int = 3
    ) -> List[Dict]:
        """
        Get relevant memories from past conversations
        
        Finds past conversations that might be relevant to current query
        
        Args:
            session: SQLAlchemy session
            current_query: Current user query
            user_id: User identifier
            limit: Max memories to return
        
        Returns:
            List of relevant conversation snippets
        """
        from models import Conversation, Message
        
        # Extract keywords from current query
        query_keywords = self._extract_keywords(current_query)
        
        if not query_keywords:
            return []
        
        # Find conversations with matching topics
        conversations = session.query(Conversation).filter_by(
            user_id=user_id
        ).order_by(Conversation.updated_at.desc()).limit(20).all()
        
        relevant = []
        
        for conv in conversations:
            # Skip if no summary
            if not conv.summary:
                continue
            
            # Check summary for keyword matches
            summary_lower = conv.summary.lower()
            matches = sum(1 for kw in query_keywords if kw in summary_lower)
            
            if matches > 0:
                # Get a relevant message from this conversation
                messages = session.query(Message).filter_by(
                    conversation_id=conv.conversation_id,
                    role='assistant'
                ).order_by(Message.created_at.desc()).limit(3).all()
                
                for msg in messages:
                    msg_lower = msg.content.lower()
                    msg_matches = sum(1 for kw in query_keywords if kw in msg_lower)
                    
                    if msg_matches > 0:
                        relevant.append({
                            'conversation_id': conv.conversation_id,
                            'summary': conv.summary,
                            'relevance_score': matches + msg_matches,
                            'snippet': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
                            'date': datetime.fromtimestamp(conv.updated_at).strftime('%Y-%m-%d')
                        })
                        break
        
        # Sort by relevance and return top results
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant[:limit]
    
    def build_memory_context(
        self,
        relevant_memories: List[Dict]
    ) -> str:
        """
        Format relevant memories for LLM context
        
        Args:
            relevant_memories: List of memory dicts
        
        Returns:
            Formatted memory context string
        """
        if not relevant_memories:
            return ""
        
        parts = ["RELEVANT PAST CONVERSATIONS:\n"]
        
        for i, memory in enumerate(relevant_memories, 1):
            parts.append(
                f"{i}. From {memory['date']}: {memory['summary']}\n"
                f"   Note: \"{memory['snippet']}\"\n"
            )
        
        parts.append("---\n")
        
        return '\n'.join(parts)
    
    def get_photo_interaction_history(
        self,
        session,
        photo_id: str,
        user_id: str = "default_user"
    ) -> List[Dict]:
        """
        Get history of interactions with a specific photo
        
        Args:
            session: SQLAlchemy session
            photo_id: Photo ID
            user_id: User ID
        
        Returns:
            List of interaction records
        """
        from models import Message, Conversation
        
        interactions = []
        
        # Find all messages that retrieved this photo
        messages = session.query(Message).all()
        
        for msg in messages:
            if msg.retrieved_photo_ids:
                try:
                    photo_ids = json.loads(msg.retrieved_photo_ids)
                    if photo_id in photo_ids:
                        # Get conversation info
                        conv = session.query(Conversation).filter_by(
                            conversation_id=msg.conversation_id,
                            user_id=user_id
                        ).first()
                        
                        if conv:
                            interactions.append({
                                'conversation_id': msg.conversation_id,
                                'message_id': msg.message_id,
                                'query': msg.content if msg.role == 'user' else None,
                                'date': datetime.fromtimestamp(msg.created_at).strftime('%Y-%m-%d %H:%M'),
                                'conversation_summary': conv.summary
                            })
                except:
                    continue
        
        # Sort by date
        interactions.sort(key=lambda x: x['date'], reverse=True)
        
        return interactions
    
    def get_frequently_discussed_topics(
        self,
        session,
        user_id: str = "default_user",
        limit: int = 10
    ) -> List[Dict]:
        """
        Get most frequently discussed topics across all conversations
        
        Args:
            session: SQLAlchemy session
            user_id: User ID
            limit: Max topics to return
        
        Returns:
            List of topic dicts with counts
        """
        from models import Conversation
        
        conversations = session.query(Conversation).filter_by(
            user_id=user_id
        ).all()
        
        # Count topic occurrences
        topic_counts = {}
        
        for conv in conversations:
            topics = self.extract_conversation_topics(session, conv.conversation_id)
            
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by frequency
        sorted_topics = sorted(
            topic_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'topic': topic, 'count': count}
            for topic, count in sorted_topics[:limit]
        ]
    
    def get_conversation_timeline(
        self,
        session,
        user_id: str = "default_user",
        days: int = 30
    ) -> List[Dict]:
        """
        Get timeline of conversations over time
        
        Args:
            session: SQLAlchemy session
            user_id: User ID
            days: Number of days to look back
        
        Returns:
            Timeline data
        """
        from models import Conversation
        
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        conversations = session.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.created_at >= cutoff
        ).order_by(Conversation.created_at).all()
        
        timeline = []
        for conv in conversations:
            timeline.append({
                'conversation_id': conv.conversation_id,
                'date': datetime.fromtimestamp(conv.created_at).strftime('%Y-%m-%d'),
                'message_count': conv.message_count or 0,
                'summary': conv.summary[:100] if conv.summary else 'No summary'
            })
        
        return timeline
    
    def optimize_long_conversation(
        self,
        session,
        conversation_id: str,
        llm_service,
        target_messages: int = 20
    ) -> Dict:
        """
        Optimize a long conversation by compressing old messages
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            llm_service: LLM service for summarization
            target_messages: Keep this many recent messages
        
        Returns:
            Optimization results
        """
        from models import Message, Conversation
        
        messages = session.query(Message).filter_by(
            conversation_id=conversation_id
        ).order_by(Message.created_at).all()
        
        total_messages = len(messages)
        
        if total_messages <= target_messages:
            return {
                'optimized': False,
                'reason': f'Only {total_messages} messages, no optimization needed'
            }
        
        # Split messages: old (to compress) vs recent (to keep)
        old_messages = messages[:-target_messages]
        recent_messages = messages[-target_messages:]
        
        # Summarize old messages
        from services.conversation_service import get_conversation_service
        conv_service = get_conversation_service()
        
        old_summary = conv_service._generate_summary(
            [
                {'role': m.role, 'content': m.content}
                for m in old_messages
            ],
            llm_service
        )
        
        # Update conversation summary
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        if conversation:
            conversation.summary = old_summary
            session.commit()
        
        logger.info(
            f"✓ Optimized conversation {conversation_id}: "
            f"{total_messages} messages → {target_messages} + summary"
        )
        
        return {
            'optimized': True,
            'original_messages': total_messages,
            'compressed_messages': len(old_messages),
            'kept_messages': len(recent_messages),
            'summary_length': len(old_summary)
        }


# Singleton
_memory_service = None

def get_memory_service():
    """Get or create memory service singleton"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


# Example usage
if __name__ == "__main__":
    from models import Session
    
    session = Session()
    memory_service = MemoryService()
    
    # Get frequently discussed topics
    topics = memory_service.get_frequently_discussed_topics(session)
    print("Top topics:", topics)
    
    # Get relevant memories for a query
    memories = memory_service.get_relevant_memories(
        session,
        "beach photos with family"
    )
    print(f"Found {len(memories)} relevant memories")
    
    session.close()