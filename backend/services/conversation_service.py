"""
Enhanced Conversation Service - Phase 5: Conversational Memory
Includes automatic summarization, context optimization, and long-term memory

FIXES APPLIED:
- All `message.metadata` / `msg.metadata` references changed to `message.meta_data` / `msg.meta_data`
  to match the Message model column name (which avoids the SQLAlchemy Base.metadata collision).
- `Message(metadata=...)` constructor calls changed to `Message(meta_data=...)`.
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
        summarize_threshold: int = 8,
        summary_every_n: int = 10
    ):
        self.max_context_messages = max_context_messages
        self.max_tokens = max_tokens
        self.chars_per_token = 4
        self.summarize_threshold = summarize_threshold
        self.summary_every_n = summary_every_n

    # =========================================================================
    # STEP 5.1: CONVERSATION STORAGE
    # =========================================================================

    def create_conversation(
        self,
        session,
        user_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        from models import Conversation

        conversation = Conversation(
            user_id=user_id or "default_user",
            created_at=time.time(),
            updated_at=time.time(),
            summary=title or None
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
        from models import Message, Conversation

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            retrieved_photo_ids=json.dumps(retrieved_photo_ids) if retrieved_photo_ids else None,
            # FIX: was `metadata=metadata or {}` — must match the column name `meta_data`
            meta_data=metadata or {},
            created_at=time.time()
        )

        session.add(message)

        # Update conversation counters & summary flag
        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        if conversation:
            conversation.updated_at = time.time()
            old_count = conversation.message_count or 0
            conversation.message_count = old_count + 1
            new_count = conversation.message_count

            if new_count >= self.summarize_threshold and not conversation.summary:
                logger.info(f"📝 Triggering first summarization at {new_count} messages")
                conversation.needs_summary = True
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
                # FIX: was `msg.metadata` — correct attribute is `msg.meta_data`
                'metadata': msg.meta_data or {},
                'created_at': msg.created_at
            })

        return history

    # =========================================================================
    # STEP 5.2: CONTEXT CARRY-OVER
    # =========================================================================

    def build_context_with_history(
        self,
        session,
        conversation_id: str,
        current_query: str,
        current_context: str,
        use_summary: bool = True
    ) -> str:
        from models import Conversation

        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        context_parts = []

        if use_summary and conversation and conversation.summary:
            context_parts.append("CONVERSATION SUMMARY:")
            context_parts.append(conversation.summary)
            context_parts.append("\n---\n")

        history = self.get_conversation_history(
            session, conversation_id, limit=self.max_context_messages
        )

        if history:
            if conversation and conversation.summary:
                recent_history = history[-6:]
            else:
                recent_history = history[-10:]

            if recent_history:
                context_parts.append("RECENT CONVERSATION:")
                for msg in recent_history:
                    if msg['role'] == 'user':
                        context_parts.append(f"\nUser: {msg['content']}")
                    else:
                        response = msg['content']
                        if len(response) > 300:
                            response = response[:300] + "..."
                        context_parts.append(f"Assistant: {response}")
                context_parts.append("\n---\n")

        context_parts.append("CURRENT QUERY RESULTS:")
        context_parts.append(current_context)

        combined_context = '\n'.join(context_parts)

        estimated_tokens = len(combined_context) // self.chars_per_token
        if estimated_tokens > self.max_tokens:
            logger.warning(f"Context too long ({estimated_tokens} tokens), optimizing...")
            if len(history) > 4:
                return self._build_minimal_context(
                    conversation.summary if conversation else None,
                    history[-4:],
                    current_context
                )
            if conversation and conversation.summary:
                return self._build_minimal_context(conversation.summary, [], current_context)
            return current_context

        return combined_context

    def _build_minimal_context(
        self,
        summary: Optional[str],
        recent_messages: List[Dict],
        current_context: str
    ) -> str:
        parts = []
        if summary:
            parts.append(f"PREVIOUS CONTEXT: {summary}\n")
        if recent_messages:
            parts.append("RECENT EXCHANGE:")
            for msg in recent_messages:
                content = msg['content'][:150]
                parts.append(f"{msg['role'].title()}: {content}...")
            parts.append("")
        parts.append(current_context)
        return '\n'.join(parts)

    # =========================================================================
    # STEP 5.3: CONVERSATION SUMMARIZATION
    # =========================================================================

    def auto_summarize_if_needed(self, session, conversation_id: str, llm_service) -> Optional[str]:
        from models import Conversation

        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        if not conversation:
            return None
        if not getattr(conversation, 'needs_summary', False):
            return None

        logger.info(f"📝 Auto-summarizing conversation {conversation_id}")
        summary = self.summarize_conversation(session, conversation_id, llm_service, recursive=True)
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
        from models import Conversation

        conversation = session.query(Conversation).filter_by(
            conversation_id=conversation_id
        ).first()

        if not conversation:
            return "Conversation not found"

        history = self.get_conversation_history(session, conversation_id)
        if not history:
            return "Empty conversation"

        total_messages = len(history)

        if recursive and total_messages > max_messages:
            logger.info(f"📚 Using recursive summarization for {total_messages} messages")
            summary = self._recursive_summarize(history, llm_service, chunk_size=max_messages)
        else:
            summary = self._generate_summary(history, llm_service)

        old_summary = conversation.summary
        if old_summary:
            summary = self._merge_summaries(old_summary, summary, llm_service)

        conversation.summary = summary
        conversation.last_summarized_at = time.time()
        session.commit()

        logger.info(f"✓ Generated summary for {conversation_id} ({total_messages} messages)")
        return summary

    def _generate_summary(self, messages: List[Dict], llm_service) -> str:
        conversation_text = []
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            conversation_text.append(f"{role}: {content}")

        context = '\n'.join(conversation_text)

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

    def _recursive_summarize(self, messages: List[Dict], llm_service, chunk_size: int = 50) -> str:
        if len(messages) <= chunk_size:
            return self._generate_summary(messages, llm_service)

        logger.info(f"📚 Recursive summarization: {len(messages)} messages in chunks of {chunk_size}")
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
            chunk_summaries.append(self._generate_summary(chunk, llm_service))

        if len(chunk_summaries) > 5:
            summary_messages = [{'role': 'assistant', 'content': s} for s in chunk_summaries]
            return self._recursive_summarize(summary_messages, llm_service, chunk_size=10)

        combined = '\n\n'.join([f"Part {i + 1}: {s}" for i, s in enumerate(chunk_summaries)])
        system_prompt = """Combine these partial summaries into one coherent summary. Keep it concise (3-5 sentences) while preserving key information about photos, people, and topics discussed."""
        final_summary = llm_service.generate_response(
            context=combined,
            query="Create a unified summary from these parts",
            system_prompt=system_prompt
        )
        return final_summary.strip()

    def _merge_summaries(self, old_summary: str, new_summary: str, llm_service) -> str:
        context = f"PREVIOUS SUMMARY:\n{old_summary}\n\nNEW ACTIVITY:\n{new_summary}"
        system_prompt = """Merge these two summaries into one updated summary. The previous summary covers earlier conversation, the new activity is more recent. Create a coherent summary that incorporates both, keeping it concise (3-5 sentences)."""
        merged = llm_service.generate_response(
            context=context,
            query="Merge these summaries",
            system_prompt=system_prompt
        )
        return merged.strip()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_all_conversations(
        self,
        session,
        user_id: Optional[str] = None,
        limit: int = 20,
        include_stats: bool = True
    ) -> List[Dict]:
        from models import Conversation, Message

        query = session.query(Conversation)
        if user_id:
            query = query.filter_by(user_id=user_id)
        query = query.order_by(Conversation.updated_at.desc()).limit(limit)
        conversations = query.all()

        result = []
        for conv in conversations:
            first_message = session.query(Message).filter_by(
                conversation_id=conv.conversation_id,
                role='user'
            ).order_by(Message.created_at).first()

            preview = (
                first_message.content[:100] if first_message
                else conv.summary[:100] if conv.summary
                else "New conversation"
            )

            conv_dict = {
                'conversation_id': conv.conversation_id,
                'created_at': conv.created_at,
                'updated_at': conv.updated_at,
                'message_count': conv.message_count or 0,
                'preview': preview,
                'summary': conv.summary
            }

            if include_stats:
                conv_dict['stats'] = self.get_conversation_stats(session, conv.conversation_id)

            result.append(conv_dict)

        return result

    def get_conversation_stats(self, session, conversation_id: str) -> Dict:
        from models import Message

        messages = session.query(Message).filter_by(conversation_id=conversation_id).all()

        if not messages:
            return {
                'total_messages': 0, 'user_messages': 0, 'assistant_messages': 0,
                'total_photos_discussed': 0, 'avg_response_length': 0, 'conversation_duration': 0
            }

        user_msgs = [m for m in messages if m.role == 'user']
        assistant_msgs = [m for m in messages if m.role == 'assistant']

        all_photo_ids = set()
        for msg in messages:
            if msg.retrieved_photo_ids:
                try:
                    all_photo_ids.update(json.loads(msg.retrieved_photo_ids))
                except Exception:
                    pass

        duration = messages[-1].created_at - messages[0].created_at if len(messages) > 1 else 0
        avg_length = (
            sum(len(m.content) for m in assistant_msgs) / len(assistant_msgs)
            if assistant_msgs else 0
        )

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
        from models import Conversation, Message

        try:
            session.query(Message).filter_by(conversation_id=conversation_id).delete()
            session.query(Conversation).filter_by(conversation_id=conversation_id).delete()
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
        from models import Conversation, Message

        conv_query = session.query(Conversation).filter(
            Conversation.summary.ilike(f'%{query}%')
        )
        if user_id:
            conv_query = conv_query.filter_by(user_id=user_id)
        conversations = conv_query.limit(limit).all()

        messages = session.query(Message).filter(
            Message.content.ilike(f'%{query}%')
        ).limit(limit * 2).all()

        conv_ids_from_messages = list(set(m.conversation_id for m in messages))
        all_conv_ids = list(set(
            [c.conversation_id for c in conversations] + conv_ids_from_messages
        ))

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
    global _enhanced_conversation_service
    if _enhanced_conversation_service is None:
        _enhanced_conversation_service = EnhancedConversationService()
    return _enhanced_conversation_service


# Backward compatibility alias used by app.py
get_conversation_service = get_enhanced_conversation_service