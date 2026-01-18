"""
Enhanced Conversation Service - Phase 5: Conversational Memory
Includes automatic summarization, context optimization, and long-term memory
"""

from typing import List, Dict, Optional, Tuple
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedConversationService:
    """
    Advanced conversation management with:
    - Automatic summarization
    - Context window optimization
    - Long-term memory
    - Recursive summarization
    """
    
    def __init__(
        self,
        max_context_messages: int = 10,
        max_tokens: int = 4000,
        summarize_threshold: int = 8,  # Summarize after N exchanges
        summary_every_n: int = 10  # Re-summarize every N new messages
    ):
        """
        Initialize enhanced conversation service
        
        Args:
            max_context_messages: Maximum messages in context
            max_tokens: Maximum context tokens
            summarize_threshold: When to create first summary
            summary_every_n: How often to update summary
        """
        self.max_context_messages = max_context_messages
        self.max_tokens = max_tokens
        self.chars_per_token = 4
        self.summarize_threshold = summarize_threshold
        self.summary_every_n = summary_every_n
    
    # ========================================================================
    # STEP 5.1: CONVERSATION STORAGE (Enhanced)
    # ========================================================================
    
    def create_conversation(
        self,
        session,
        user_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Create a new conversation with optional title
        
        Args:
            session: SQLAlchemy session
            user_id: Optional user identifier
            title: Optional conversation title
        
        Returns:
            conversation_id
        """
        from models import Conversation
        
        conversation = Conversation(
            user_id=user_id or "default_user",
            created_at=time.time(),
            updated_at=time.time(),
            summary=title or None  # Use title as initial summary
        )
        
        session.add(conversation)
        session.commit()
        
        conv_id = conversation.conversation_id
        logger.info(f"✓ Created conversation: {conv_id}")
        
        return conv_id
    
    def add_message(
        self,
        session,
        conversation_id: str,
        role: str,
        content: str,
        retrieved_photo_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add message and check if summarization is needed
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            role: 'user' or 'assistant'
            content: Message text
            retrieved_photo_ids: Photos retrieved for this message
            metadata: Additional metadata
        
        Returns:
            message_id
        """
        from models import Message, Conversation
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            retrieved_photo_ids=json.dumps(retrieved_photo_ids) if retrieved_photo_ids else None,
            metadata=metadata or {},
            created_at=time.time()
        )
        
        session.add(message)
        
        # Update conversation
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        if conversation:
            conversation.updated_at = time.time()
            old_count = conversation.message_count or 0
            conversation.message_count = old_count + 1
            
            # Check if we should summarize
            new_count = conversation.message_count
            
            # First summarization after threshold
            if new_count >= self.summarize_threshold and not conversation.summary:
                logger.info(f"📝 Triggering first summarization at {new_count} messages")
                conversation.needs_summary = True
            
            # Re-summarization
            elif conversation.summary and new_count % self.summary_every_n == 0:
                logger.info(f"📝 Triggering re-summarization at {new_count} messages")
                conversation.needs_summary = True
        
        session.commit()
        
        logger.info(f"✓ Added {role} message to {conversation_id}")
        
        return message.message_id
    
    def get_conversation_history(
        self,
        session,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get conversation history with pagination
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            limit: Number of messages to return
            offset: Number of messages to skip
        
        Returns:
            List of message dicts
        """
        from models import Message
        
        query = session.query(Message).filter_by(
            conversation_id=conversation_id
        ).order_by(Message.created_at)
        
        if offset:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        messages = query.all()
        
        history = []
        for msg in messages:
            history.append({
                'message_id': msg.message_id,
                'role': msg.role,
                'content': msg.content,
                'retrieved_photo_ids': json.loads(msg.retrieved_photo_ids) if msg.retrieved_photo_ids else [],
                'metadata': msg.metadata or {},
                'created_at': msg.created_at
            })
        
        return history
    
    # ========================================================================
    # STEP 5.2: CONTEXT CARRY-OVER (Enhanced)
    # ========================================================================
    
    def build_context_with_history(
        self,
        session,
        conversation_id: str,
        current_query: str,
        current_context: str,
        use_summary: bool = True
    ) -> str:
        """
        Build context with intelligent history management
        
        Uses sliding window + summary for optimal context
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            current_query: Current user query
            current_context: Context from current retrieval
            use_summary: Whether to use conversation summary
        
        Returns:
            Combined context string
        """
        from models import Conversation
        
        # Get conversation
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        context_parts = []
        
        # Add summary if available and requested
        if use_summary and conversation and conversation.summary:
            context_parts.append("CONVERSATION SUMMARY:")
            context_parts.append(conversation.summary)
            context_parts.append("\n---\n")
        
        # Get recent messages (sliding window)
        history = self.get_conversation_history(
            session,
            conversation_id,
            limit=self.max_context_messages
        )
        
        # Add recent conversation history
        if history:
            # Determine how many messages to include
            if conversation and conversation.summary:
                # If we have summary, only include last 4-6 messages
                recent_history = history[-6:]
            else:
                # No summary, include more history
                recent_history = history[-10:]
            
            if recent_history:
                context_parts.append("RECENT CONVERSATION:")
                
                for msg in recent_history:
                    if msg['role'] == 'user':
                        context_parts.append(f"\nUser: {msg['content']}")
                    else:
                        # Truncate long assistant responses
                        response = msg['content']
                        if len(response) > 300:
                            response = response[:300] + "..."
                        context_parts.append(f"Assistant: {response}")
                
                context_parts.append("\n---\n")
        
        # Add current retrieved photos
        context_parts.append("CURRENT QUERY RESULTS:")
        context_parts.append(current_context)
        
        combined_context = '\n'.join(context_parts)
        
        # Check token limit
        estimated_tokens = len(combined_context) // self.chars_per_token
        
        if estimated_tokens > self.max_tokens:
            logger.warning(f"Context too long ({estimated_tokens} tokens), optimizing...")
            
            # Strategy 1: Remove older history, keep only last 4 messages
            if len(history) > 4:
                return self._build_minimal_context(
                    conversation.summary if conversation else None,
                    history[-4:],
                    current_context
                )
            
            # Strategy 2: Use only summary + current context
            if conversation and conversation.summary:
                return self._build_minimal_context(
                    conversation.summary,
                    [],
                    current_context
                )
            
            # Strategy 3: Just current context
            return current_context
        
        return combined_context
    
    def _build_minimal_context(
        self,
        summary: Optional[str],
        recent_messages: List[Dict],
        current_context: str
    ) -> str:
        """Build minimal context to fit token limits"""
        parts = []
        
        if summary:
            parts.append(f"PREVIOUS CONTEXT: {summary}\n")
        
        if recent_messages:
            parts.append("RECENT EXCHANGE:")
            for msg in recent_messages:
                content = msg['content'][:150]  # Truncate heavily
                parts.append(f"{msg['role'].title()}: {content}...")
            parts.append("")
        
        parts.append(current_context)
        
        return '\n'.join(parts)
    
    # ========================================================================
    # STEP 5.3: CONVERSATION SUMMARIZATION (New)
    # ========================================================================
    
    def auto_summarize_if_needed(
        self,
        session,
        conversation_id: str,
        llm_service
    ) -> Optional[str]:
        """
        Automatically summarize conversation if needed
        
        Checks if conversation needs summarization and performs it
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            llm_service: LLM service for generation
        
        Returns:
            Summary text if generated, None otherwise
        """
        from models import Conversation
        
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        if not conversation:
            return None
        
        # Check if summarization is needed
        if not getattr(conversation, 'needs_summary', False):
            return None
        
        logger.info(f"📝 Auto-summarizing conversation {conversation_id}")
        
        # Generate summary
        summary = self.summarize_conversation(
            session,
            conversation_id,
            llm_service,
            recursive=True
        )
        
        # Clear flag
        conversation.needs_summary = False
        session.commit()
        
        return summary
    
    def summarize_conversation(
        self,
        session,
        conversation_id: str,
        llm_service,
        recursive: bool = False,
        max_messages: int = 50
    ) -> str:
        """
        Generate or update conversation summary
        
        Phase 5.3: Advanced summarization with recursion
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            llm_service: LLM service
            recursive: Use recursive summarization for long conversations
            max_messages: Max messages to summarize at once
        
        Returns:
            Summary text
        """
        from models import Conversation
        
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        if not conversation:
            return "Conversation not found"
        
        # Get all messages
        history = self.get_conversation_history(session, conversation_id)
        
        if not history:
            return "Empty conversation"
        
        total_messages = len(history)
        
        # Recursive summarization for long conversations
        if recursive and total_messages > max_messages:
            logger.info(f"📚 Using recursive summarization for {total_messages} messages")
            summary = self._recursive_summarize(
                history,
                llm_service,
                chunk_size=max_messages
            )
        else:
            # Standard summarization
            summary = self._generate_summary(history, llm_service)
        
        # Update conversation
        old_summary = conversation.summary
        
        if old_summary:
            # Combine old and new
            summary = self._merge_summaries(old_summary, summary, llm_service)
        
        conversation.summary = summary
        conversation.last_summarized_at = time.time()
        session.commit()
        
        logger.info(f"✓ Generated summary for {conversation_id} ({total_messages} messages)")
        
        return summary
    
    def _generate_summary(
        self,
        messages: List[Dict],
        llm_service
    ) -> str:
        """Generate summary from message list"""
        
        # Build conversation text
        conversation_text = []
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            conversation_text.append(f"{role}: {content}")
        
        context = '\n'.join(conversation_text)
        
        # Summarization prompt
        system_prompt = """You are summarizing a conversation about photos. Create a concise summary (3-5 sentences) that captures:

1. What the user was looking for
2. Key topics or photos discussed  
3. Important findings or patterns mentioned
4. Any specific people, places, or events referenced

Be factual and specific. Focus on photo-related information."""
        
        summary = llm_service.generate_response(
            context=context,
            query="Summarize this conversation concisely",
            system_prompt=system_prompt
        )
        
        return summary.strip()
    
    def _recursive_summarize(
        self,
        messages: List[Dict],
        llm_service,
        chunk_size: int = 50
    ) -> str:
        """
        Recursively summarize long conversations
        
        Breaks conversation into chunks, summarizes each, then summarizes summaries
        """
        
        if len(messages) <= chunk_size:
            return self._generate_summary(messages, llm_service)
        
        logger.info(f"📚 Recursive summarization: {len(messages)} messages in chunks of {chunk_size}")
        
        # Split into chunks
        chunks = []
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i + chunk_size]
            chunks.append(chunk)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Summarizing chunk {i+1}/{len(chunks)}...")
            summary = self._generate_summary(chunk, llm_service)
            chunk_summaries.append(summary)
        
        # If we have many chunk summaries, recursively summarize them
        if len(chunk_summaries) > 5:
            # Convert summaries to message format
            summary_messages = [
                {'role': 'assistant', 'content': s}
                for s in chunk_summaries
            ]
            return self._recursive_summarize(
                summary_messages,
                llm_service,
                chunk_size=10
            )
        
        # Combine chunk summaries into final summary
        combined = '\n\n'.join([
            f"Part {i+1}: {s}"
            for i, s in enumerate(chunk_summaries)
        ])
        
        # Generate meta-summary
        system_prompt = """Combine these partial summaries into one coherent summary. Keep it concise (3-5 sentences) while preserving key information about photos, people, and topics discussed."""
        
        final_summary = llm_service.generate_response(
            context=combined,
            query="Create a unified summary from these parts",
            system_prompt=system_prompt
        )
        
        return final_summary.strip()
    
    def _merge_summaries(
        self,
        old_summary: str,
        new_summary: str,
        llm_service
    ) -> str:
        """Merge old and new summaries"""
        
        context = f"PREVIOUS SUMMARY:\n{old_summary}\n\nNEW ACTIVITY:\n{new_summary}"
        
        system_prompt = """Merge these two summaries into one updated summary. The previous summary covers earlier conversation, the new activity is more recent. Create a coherent summary that incorporates both, keeping it concise (3-5 sentences)."""
        
        merged = llm_service.generate_response(
            context=context,
            query="Merge these summaries",
            system_prompt=system_prompt
        )
        
        return merged.strip()
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_all_conversations(
        self,
        session,
        user_id: Optional[str] = None,
        limit: int = 20,
        include_stats: bool = True
    ) -> List[Dict]:
        """Get all conversations with stats"""
        from models import Conversation, Message
        
        query = session.query(Conversation)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        query = query.order_by(Conversation.updated_at.desc()).limit(limit)
        conversations = query.all()
        
        result = []
        for conv in conversations:
            # Get preview (first user message)
            first_message = session.query(Message).filter_by(
                conversation_id=conv.conversation_id,
                role='user'
            ).order_by(Message.created_at).first()
            
            preview = first_message.content[:100] if first_message else conv.summary[:100] if conv.summary else "New conversation"
            
            conv_dict = {
                'conversation_id': conv.conversation_id,
                'created_at': conv.created_at,
                'updated_at': conv.updated_at,
                'message_count': conv.message_count or 0,
                'preview': preview,
                'summary': conv.summary
            }
            
            # Add stats if requested
            if include_stats:
                stats = self.get_conversation_stats(session, conv.conversation_id)
                conv_dict['stats'] = stats
            
            result.append(conv_dict)
        
        return result
    
    def get_conversation_stats(self, session, conversation_id: str) -> Dict:
        """Get detailed statistics about a conversation"""
        from models import Message
        
        messages = session.query(Message).filter_by(
            conversation_id=conversation_id
        ).all()
        
        if not messages:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'assistant_messages': 0,
                'total_photos_discussed': 0,
                'avg_response_length': 0,
                'conversation_duration': 0
            }
        
        user_msgs = [m for m in messages if m.role == 'user']
        assistant_msgs = [m for m in messages if m.role == 'assistant']
        
        # Count unique photos
        all_photo_ids = set()
        for msg in messages:
            if msg.retrieved_photo_ids:
                try:
                    photo_ids = json.loads(msg.retrieved_photo_ids)
                    all_photo_ids.update(photo_ids)
                except:
                    pass
        
        # Calculate duration
        if len(messages) > 1:
            duration = messages[-1].created_at - messages[0].created_at
        else:
            duration = 0
        
        # Average response length
        if assistant_msgs:
            avg_length = sum(len(m.content) for m in assistant_msgs) / len(assistant_msgs)
        else:
            avg_length = 0
        
        return {
            'total_messages': len(messages),
            'user_messages': len(user_msgs),
            'assistant_messages': len(assistant_msgs),
            'total_photos_discussed': len(all_photo_ids),
            'avg_response_length': int(avg_length),
            'conversation_duration': int(duration),
            'exchanges': len(user_msgs)
        }
    
    def delete_conversation(self, session, conversation_id: str) -> bool:
        """Delete conversation and all messages"""
        from models import Conversation, Message
        
        try:
            session.query(Message).filter_by(
                conversation_id=conversation_id
            ).delete()
            
            session.query(Conversation).filter_by(
                conversation_id=conversation_id
            ).delete()
            
            session.commit()
            logger.info(f"✓ Deleted conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            session.rollback()
            return False
    
    def search_conversations(
        self,
        session,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search conversations by content or summary
        
        Args:
            session: SQLAlchemy session
            query: Search query
            user_id: Optional user filter
            limit: Max results
        
        Returns:
            List of matching conversations
        """
        from models import Conversation, Message
        
        # Search in summaries
        conv_query = session.query(Conversation).filter(
            Conversation.summary.ilike(f'%{query}%')
        )
        
        if user_id:
            conv_query = conv_query.filter_by(user_id=user_id)
        
        conversations = conv_query.limit(limit).all()
        
        # Also search in message content
        message_query = session.query(Message).filter(
            Message.content.ilike(f'%{query}%')
        ).limit(limit * 2)
        
        messages = message_query.all()
        
        # Get unique conversation IDs from messages
        conv_ids_from_messages = list(set(m.conversation_id for m in messages))
        
        # Combine results
        all_conv_ids = list(set(
            [c.conversation_id for c in conversations] + 
            conv_ids_from_messages
        ))
        
        # Get full conversation objects
        results = session.query(Conversation).filter(
            Conversation.conversation_id.in_(all_conv_ids)
        ).order_by(Conversation.updated_at.desc()).limit(limit).all()
        
        return [
            {
                'conversation_id': c.conversation_id,
                'summary': c.summary,
                'message_count': c.message_count,
                'updated_at': c.updated_at
            }
            for c in results
        ]


# Singleton instance
_enhanced_conversation_service = None

def get_enhanced_conversation_service():
    """Get or create enhanced conversation service singleton"""
    global _enhanced_conversation_service
    if _enhanced_conversation_service is None:
        _enhanced_conversation_service = EnhancedConversationService()
    return _enhanced_conversation_service


# Backward compatibility
get_conversation_service = get_enhanced_conversation_service