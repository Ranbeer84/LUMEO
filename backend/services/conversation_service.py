"""
Conversation Service - Manage Chat History and Context
Phase 4: Multi-turn conversation support

Handles:
- Creating and managing conversations
- Storing messages with retrieved photo context
- Context window management (token limits)
- Conversation summarization
"""

from typing import List, Dict, Optional, Tuple
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationService:
    """
    Manages conversation history and context for multi-turn dialogues
    """
    
    def __init__(self, max_context_messages: int = 10, max_tokens: int = 4000):
        """
        Initialize conversation service
        
        Args:
            max_context_messages: Maximum messages to include in context
            max_tokens: Maximum total tokens for context
        """
        self.max_context_messages = max_context_messages
        self.max_tokens = max_tokens
        self.chars_per_token = 4  # Rough estimate
    
    def create_conversation(self, session, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation
        
        Args:
            session: SQLAlchemy session
            user_id: Optional user identifier
        
        Returns:
            conversation_id
        """
        from models import Conversation
        
        conversation = Conversation(
            user_id=user_id or "default_user",
            created_at=time.time(),
            updated_at=time.time()
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
        Add a message to conversation
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            role: 'user' or 'assistant'
            content: Message text
            retrieved_photo_ids: Photos retrieved for this message
            metadata: Additional metadata (similarity scores, filters, etc.)
        
        Returns:
            message_id
        """
        from models import Message, Conversation
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            retrieved_photo_ids=json.dumps(retrieved_photo_ids) if retrieved_photo_ids else None,
            meta_data=metadata or {},
            created_at=time.time()
        )
        
        session.add(message)
        
        # Update conversation timestamp
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        if conversation:
            conversation.updated_at = time.time()
            conversation.message_count = (conversation.message_count or 0) + 1
        
        session.commit()
        
        logger.info(f"✓ Added {role} message to conversation {conversation_id}")
        
        return message.message_id
    
    def get_conversation_history(
        self,
        session,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation message history
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            limit: Optional limit on number of messages
        
        Returns:
            List of message dicts
        """
        from models import Message
        
        query = session.query(Message).filter_by(
            conversation_id=conversation_id
        ).order_by(Message.created_at)
        
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
                'metadata': msg.meta_data or {},
                'created_at': msg.created_at
            })
        
        return history
    
    def build_context_with_history(
        self,
        session,
        conversation_id: str,
        current_query: str,
        current_context: str
    ) -> str:
        """
        Build context including conversation history
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            current_query: Current user query
            current_context: Context from current retrieval
        
        Returns:
            Combined context string with history
        """
        # Get recent messages (sliding window)
        history = self.get_conversation_history(
            session,
            conversation_id,
            limit=self.max_context_messages
        )
        
        # Build context parts
        context_parts = []
        
        # Add conversation history
        if history:
            context_parts.append("CONVERSATION HISTORY:")
            
            for msg in history[-6:]:  # Last 6 messages (3 exchanges)
                if msg['role'] == 'user':
                    context_parts.append(f"\nUser: {msg['content']}")
                else:
                    # Summarize assistant response if too long
                    response = msg['content']
                    if len(response) > 200:
                        response = response[:200] + "..."
                    context_parts.append(f"Assistant: {response}")
            
            context_parts.append("\n---\n")
        
        # Add current retrieved photos
        context_parts.append(current_context)
        
        combined_context = '\n'.join(context_parts)
        
        # Check token limit
        estimated_tokens = len(combined_context) // self.chars_per_token
        
        if estimated_tokens > self.max_tokens:
            logger.warning(f"Context too long ({estimated_tokens} tokens), truncating history")
            # Use only current context without full history
            combined_context = f"PREVIOUS CONTEXT: [User has been exploring their photos]\n\n{current_context}"
        
        return combined_context
    
    def get_all_conversations(
        self,
        session,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get all conversations for a user
        
        Args:
            session: SQLAlchemy session
            user_id: Optional user filter
            limit: Maximum conversations to return
        
        Returns:
            List of conversation dicts
        """
        from models import Conversation, Message
        
        query = session.query(Conversation)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        query = query.order_by(Conversation.updated_at.desc()).limit(limit)
        conversations = query.all()
        
        result = []
        for conv in conversations:
            # Get first user message for preview
            first_message = session.query(Message).filter_by(
                conversation_id=conv.conversation_id,
                role='user'
            ).order_by(Message.created_at).first()
            
            preview = first_message.content[:100] if first_message else "New conversation"
            
            result.append({
                'conversation_id': conv.conversation_id,
                'created_at': conv.created_at,
                'updated_at': conv.updated_at,
                'message_count': conv.message_count or 0,
                'preview': preview,
                'summary': conv.summary
            })
        
        return result
    
    def delete_conversation(self, session, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        from models import Conversation, Message
        
        try:
            # Delete messages first
            session.query(Message).filter_by(
                conversation_id=conversation_id
            ).delete()
            
            # Delete conversation
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
    
    def summarize_conversation(
        self,
        session,
        conversation_id: str,
        llm_service
    ) -> str:
        """
        Generate a summary of the conversation
        
        Args:
            session: SQLAlchemy session
            conversation_id: Conversation ID
            llm_service: LLM service instance
        
        Returns:
            Summary text
        """
        from models import Conversation
        
        # Get full history
        history = self.get_conversation_history(session, conversation_id)
        
        if not history:
            return "Empty conversation"
        
        # Build summary prompt
        conversation_text = []
        for msg in history:
            conversation_text.append(f"{msg['role'].upper()}: {msg['content']}")
        
        context = '\n'.join(conversation_text)
        
        system_prompt = """Summarize this conversation in 2-3 sentences. Focus on:
- What the user was looking for
- Key topics or photos discussed
- Main findings or insights

Keep it concise and factual."""
        
        summary = llm_service.generate_response(
            context=context,
            query="Summarize this conversation",
            system_prompt=system_prompt
        )
        
        # Store summary
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()
        
        if conversation:
            conversation.summary = summary
            session.commit()
            logger.info(f"✓ Generated summary for conversation {conversation_id}")
        
        return summary
    
    def get_conversation_stats(self, session, conversation_id: str) -> Dict:
        """Get statistics about a conversation"""
        from models import Message
        
        messages = session.query(Message).filter_by(
            conversation_id=conversation_id
        ).all()
        
        if not messages:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'assistant_messages': 0,
                'total_photos_discussed': 0
            }
        
        user_msgs = sum(1 for m in messages if m.role == 'user')
        assistant_msgs = sum(1 for m in messages if m.role == 'assistant')
        
        # Count unique photos discussed
        all_photo_ids = set()
        for msg in messages:
            if msg.retrieved_photo_ids:
                try:
                    photo_ids = json.loads(msg.retrieved_photo_ids)
                    all_photo_ids.update(photo_ids)
                except:
                    pass
        
        return {
            'total_messages': len(messages),
            'user_messages': user_msgs,
            'assistant_messages': assistant_msgs,
            'total_photos_discussed': len(all_photo_ids),
            'exchanges': user_msgs  # Number of user queries
        }


# Singleton instance
_conversation_service = None

def get_conversation_service():
    """Get or create conversation service singleton"""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service


# Example usage
if __name__ == "__main__":
    from models import Session
    
    session = Session()
    service = ConversationService()
    
    # Create conversation
    conv_id = service.create_conversation(session)
    print(f"Created conversation: {conv_id}")
    
    # Add messages
    service.add_message(
        session,
        conv_id,
        role='user',
        content='Show me beach photos',
        retrieved_photo_ids=['photo_1', 'photo_2']
    )
    
    service.add_message(
        session,
        conv_id,
        role='assistant',
        content='I found 2 beach photos showing family at the coast.'
    )
    
    # Get history
    history = service.get_conversation_history(session, conv_id)
    print(f"\nConversation history: {len(history)} messages")
    
    # Get stats
    stats = service.get_conversation_stats(session, conv_id)
    print(f"Stats: {stats}")
    
    session.close()