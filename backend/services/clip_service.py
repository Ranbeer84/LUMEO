"""
CLIP Service - Multi-Modal Image Embeddings for Semantic Search
Phase 2.6
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPService:
    """Handle CLIP embeddings for images and text"""
    
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        """
        Initialize CLIP model
        
        Args:
            model_name: HuggingFace model name for CLIP
        """
        try:
            logger.info(f"Loading CLIP model: {model_name}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            self.model = None
            self.processor = None
    
    def encode_image(self, image_path):
        """
        Generate CLIP embedding for an image
        
        Args:
            image_path: Path to image file
        
        Returns:
            numpy.array: 512-dimensional embedding vector, or None on error
        """
        if self.model is None or self.processor is None:
            logger.error("CLIP model not loaded")
            return None
        
        try:
            logger.info(f"Encoding image: {image_path}")
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array
            embedding = image_features.cpu().numpy().flatten()
            
            logger.info(f"Generated embedding with shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None
    
    def encode_text(self, text):
        """
        Generate CLIP embedding for text query
        
        Args:
            text: Text string to encode
        
        Returns:
            numpy.array: 512-dimensional embedding vector, or None on error
        """
        if self.model is None or self.processor is None:
            logger.error("CLIP model not loaded")
            return None
        
        try:
            logger.info(f"Encoding text: {text}")
            
            # Process text
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array
            embedding = text_features.cpu().numpy().flatten()
            
            logger.info(f"Generated text embedding with shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        try:
            # Normalize if not already
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def search_similar_images(self, query_embedding, image_embeddings, top_k=10):
        """
        Find top-k most similar images to a query
        
        Args:
            query_embedding: Query embedding vector
            image_embeddings: List of (image_id, embedding) tuples
            top_k: Number of results to return
        
        Returns:
            list: List of (image_id, similarity_score) tuples, sorted by similarity
        """
        try:
            similarities = []
            
            for image_id, image_embedding in image_embeddings:
                similarity = self.calculate_similarity(query_embedding, image_embedding)
                similarities.append((image_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def generate_caption_embedding(self, caption_text):
        """
        Generate embedding for a photo caption/description
        
        This is useful for hybrid search where you combine
        the image embedding with a text description embedding
        
        Args:
            caption_text: Description of the photo
        
        Returns:
            numpy.array: Text embedding
        """
        return self.encode_text(caption_text)


# Singleton instance
_clip_service = None

def get_clip_service():
    """Get or create CLIP service singleton"""
    global _clip_service
    if _clip_service is None:
        _clip_service = CLIPService()
    return _clip_service