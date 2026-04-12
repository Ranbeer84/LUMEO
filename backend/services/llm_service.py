"""
LLM Service - Ollama Integration for Photo Memory Assistant
Phase 4.2: Integrate LLM into Flask Backend
Phase 4.3: Grounded Response Enforcement
Phase 4.4: Multi-Photo Reasoning

Handles:
- Connection to local Ollama instance
- Streaming and non-streaming responses
- Grounded prompting (no hallucinations)
- Multi-photo analysis and reasoning
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Generator, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles LLM generation using local Ollama
    """
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize LLM service
        
        Args:
            model: Ollama model name (llama3.2:3b, llama3.3:70b, etc.)
            base_url: Ollama API base URL
            temperature: Response randomness (0-1, higher = more creative)
            max_tokens: Maximum response length
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Verify Ollama is running
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model in model_names:
                    logger.info(f"✓ Ollama connected. Using model: {self.model}")
                else:
                    logger.warning(f"⚠️  Model {self.model} not found. Available: {model_names}")
                    logger.warning(f"   Run: ollama pull {self.model}")
            else:
                logger.error(f"✗ Ollama API returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to Ollama at {self.base_url}")
            logger.error("  Make sure Ollama is running: ollama serve")
        except Exception as e:
            logger.error(f"✗ Ollama connection check failed: {str(e)}")
    
    def generate_response(
        self,
        context: str,
        query: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate LLM response (non-streaming)
        
        Args:
            context: Retrieved photo context
            query: User's question
            system_prompt: Optional custom system prompt
            stream: If True, returns generator instead
        
        Returns:
            Generated response string
        """
        if stream:
            return self.generate_streaming_response(context, query, system_prompt)
        
        # Build prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        full_prompt = f"{system_prompt}\n\n{context}\n\nUSER QUESTION: {query}\n\nASSISTANT:"
        
        logger.info(f"Generating response for query: {query}")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens
                    }
                },
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('response', '')
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Response generated in {elapsed:.2f}s ({len(generated_text)} chars)")
                
                return generated_text
            else:
                logger.error(f"LLM generation failed: {response.status_code}")
                return "I apologize, but I'm having trouble generating a response right now."
                
        except requests.exceptions.Timeout:
            logger.error("LLM generation timed out")
            return "I apologize, but the response is taking too long. Please try a simpler query."
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return f"I encountered an error: {str(e)}"
    
    def generate_streaming_response(
        self,
        context: str,
        query: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate LLM response with streaming (word-by-word)
        
        Args:
            context: Retrieved photo context
            query: User's question
            system_prompt: Optional custom system prompt
        
        Yields:
            Response chunks as they're generated
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        full_prompt = f"{system_prompt}\n\n{context}\n\nUSER QUESTION: {query}\n\nASSISTANT:"
        
        logger.info(f"Starting streaming response for: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "stream": True,
                    "options": {
                        "num_predict": self.max_tokens
                    }
                },
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get('response', '')
                            if chunk:
                                yield chunk
                            
                            # Check if done
                            if data.get('done', False):
                                logger.info("✓ Streaming complete")
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"Streaming failed: {response.status_code}")
                yield "I apologize, but I'm having trouble generating a response."
                
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _get_default_system_prompt(self) -> str:
        """
        Get default grounded system prompt
        
        Phase 4.3: Grounded Response Enforcement
        """
        return """You are Lumeo, an AI assistant for a personal photo memory system. You help users explore and understand their photo collection through natural conversation.

CRITICAL RULES - YOU MUST FOLLOW THESE:

1. GROUNDING: ONLY use information from the photos provided in the context below. DO NOT make up or infer information that isn't explicitly in the context.

2. UNCERTAINTY: If you don't have enough information to answer accurately, say "I don't have enough information in these photos to answer that" or similar. It's better to say you don't know than to guess.

3. CITATION: Always reference specific photos by number when discussing them (e.g., "In Photo 1, I can see..." or "Photos 2 and 3 show...").

4. ACCURACY: Do not hallucinate dates, people, locations, or events. Only state what is clearly visible or stated in the photo metadata.

5. TONE: Be conversational, warm, and helpful - like talking to a friend about their memories. But stay factual.

6. NO SPECULATION: Don't speculate about emotions, relationships, or events beyond what's explicitly detected. If emotion is "happy" in metadata, you can say that. But don't invent "they seem to be celebrating" unless objects/scene support it.

7. PATTERNS: You CAN identify patterns across multiple photos (e.g., "You appear most often with [Person]" or "Most of your beach photos are from summer") as long as this is based on the provided data.

Remember: Your users trust you with their memories. Be accurate, honest, and helpful."""
    
    def generate_insight(
        self,
        summary_context: str,
        insight_type: str = "general"
    ) -> str:
        """
        Generate insights from photo collection summary
        
        Phase 4.4: Multi-Photo Reasoning
        
        Args:
            summary_context: Aggregated statistics from photos
            insight_type: Type of insight (emotional, social, temporal, etc.)
        
        Returns:
            Natural language insights
        """
        prompts = {
            "emotional": "Analyze the emotional patterns in these photos. What moments seem happiest? Are there any emotional trends over time?",
            "social": "Analyze the social patterns. Who appears most often? What are the main social groups or relationships?",
            "temporal": "Analyze the temporal patterns. Which seasons or times have the most photos? Any patterns in when photos are taken?",
            "activity": "Analyze the activities and locations. What does this person enjoy doing? Where do they spend time?",
            "general": "Provide 3-5 interesting insights about this photo collection. Look for patterns, trends, and notable moments."
        }
        
        query = prompts.get(insight_type, prompts["general"])
        
        system_prompt = """You are analyzing a photo collection to find meaningful patterns and insights. 

Focus on:
- Observable patterns in the data
- Meaningful trends over time
- Notable moments or clusters of activity
- Social connections and relationships

Be specific and cite the numbers you're seeing. For example:
- "You appear happiest in beach photos (78% happy vs 45% overall)"
- "Summer has 3x more photos than winter"
- "You and [Person] appear together in 42 photos, often at [Location]"

Keep insights concise, interesting, and actionable."""
        
        return self.generate_response(
            context=summary_context,
            query=query,
            system_prompt=system_prompt
        )
    
    def analyze_photos_comparison(
        self,
        photo_contexts: List[str],
        comparison_query: str
    ) -> str:
        """
        Compare multiple photos or sets of photos
        
        Phase 4.4: Multi-Photo Reasoning
        
        Args:
            photo_contexts: List of photo context strings
            comparison_query: What to compare (e.g., "summer vs winter photos")
        
        Returns:
            Comparison analysis
        """
        combined_context = "\n\n---\n\n".join([
            f"SET {i+1}:\n{ctx}" 
            for i, ctx in enumerate(photo_contexts)
        ])
        
        system_prompt = """You are comparing different sets of photos. 

Focus on:
- Differences in emotions, settings, activities
- Unique characteristics of each set
- Similarities and patterns
- Notable contrasts

Be specific about what makes each set distinct."""
        
        return self.generate_response(
            context=combined_context,
            query=comparison_query,
            system_prompt=system_prompt
        )
    
    def validate_response(self, response: str, context: str) -> Dict[str, Any]:
        """
        Validate that LLM response is grounded in context
        
        Phase 4.3: Response validation
        
        Returns:
            Dict with validation results:
                - is_grounded: bool
                - confidence: float
                - warnings: List[str]
        """
        warnings = []
        
        # Check for common hallucination patterns
        hallucination_phrases = [
            "i think", "probably", "might be", "seems like",
            "i imagine", "i assume", "likely", "perhaps"
        ]
        
        response_lower = response.lower()
        for phrase in hallucination_phrases:
            if phrase in response_lower:
                warnings.append(f"Contains uncertain phrase: '{phrase}'")
        
        # Check for photo references
        has_photo_references = any([
            "photo " in response_lower,
            "image " in response_lower,
            "picture " in response_lower
        ])
        
        if not has_photo_references and len(context) > 100:
            warnings.append("Response doesn't reference specific photos")
        
        # Check for "I don't know" type responses (good!)
        acknowledges_uncertainty = any([
            "don't have enough" in response_lower,
            "not enough information" in response_lower,
            "can't determine" in response_lower,
            "unclear from" in response_lower
        ])
        
        # Calculate confidence
        confidence = 1.0
        if warnings:
            confidence -= 0.1 * len(warnings)
        if acknowledges_uncertainty:
            confidence = max(confidence, 0.8)  # Uncertainty is honest, so good
        
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "is_grounded": len(warnings) < 3,
            "confidence": confidence,
            "warnings": warnings,
            "acknowledges_uncertainty": acknowledges_uncertainty
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_llm_service = None

def get_llm_service(model: str = "llama3.2:3b"):
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(model=model)
    return _llm_service


# Example usage
if __name__ == "__main__":
    # Test LLM service
    service = LLMService()
    
    # Test basic response
    test_context = """
    PHOTO 1:
      Caption: Photo of Mom and Dad at beach looking happy during summer
      People: Mom, Dad
      Emotion: happy
      Scene: outdoor at beach
      Objects: umbrella, sand
    """
    
    query = "What are they doing?"
    
    print("Testing non-streaming response...")
    response = service.generate_response(test_context, query)
    print(f"Response: {response}\n")
    
    # Test streaming
    print("Testing streaming response...")
    print("Response: ", end="", flush=True)
    for chunk in service.generate_streaming_response(test_context, query):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Test validation
    validation = service.validate_response(response, test_context)
    print(f"Validation: {validation}")