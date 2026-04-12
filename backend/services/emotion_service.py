"""
Emotion Service - Emotion Detection and Mood Analysis
Phase 2.3
"""

from deepface import DeepFace
import numpy as np
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionService:
    """Handle emotion detection and mood analysis"""
    
    # Emotion to valence mapping (-1 negative to +1 positive)
    EMOTION_VALENCE = {
        'happy': 1.0,
        'neutral': 0.0,
        'surprise': 0.3,
        'sad': -0.8,
        'angry': -0.9,
        'fear': -0.7,
        'disgust': -0.6
    }
    
    def __init__(self):
        self.model_name = 'Emotion'  # DeepFace emotion model
    
    def detect_emotion(self, image_path, face_location=None):
        """
        Detect emotion in a face
        
        Args:
            image_path: Path to image file
            face_location: Optional tuple (top, right, bottom, left) to analyze specific region
        
        Returns:
            dict: {
                'dominant_emotion': str,
                'confidence': float,
                'all_emotions': dict,
                'valence': float
            }
        """
        try:
            logger.info(f"Detecting emotion in: {image_path}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            # If face location provided, crop to that region
            if face_location:
                top, right, bottom, left = face_location
                image = image[top:bottom, left:right]
            
            # Analyze emotion using DeepFace
            try:
                result = DeepFace.analyze(
                    img_path=image,
                    actions=['emotion'],
                    enforce_detection=False,  # Don't fail if no face detected
                    detector_backend='opencv'
                )
                
                # Handle both single face and multiple faces
                if isinstance(result, list):
                    result = result[0] if result else {}
                
                emotion_scores = result.get('emotion', {})
                dominant_emotion = result.get('dominant_emotion', 'neutral')
                
                # Calculate confidence (probability of dominant emotion)
                confidence = emotion_scores.get(dominant_emotion, 0) / 100.0
                
                # Calculate valence score
                valence = self._calculate_valence(emotion_scores)
                
                emotion_data = {
                    'dominant_emotion': dominant_emotion,
                    'confidence': round(confidence, 3),
                    'all_emotions': {k: round(v/100, 3) for k, v in emotion_scores.items()},
                    'valence': round(valence, 3)
                }
                
                logger.info(f"Detected emotion: {dominant_emotion} (confidence: {confidence:.2f})")
                
                return emotion_data
                
            except Exception as e:
                logger.warning(f"DeepFace analysis failed: {str(e)}")
                # Return neutral default
                return {
                    'dominant_emotion': 'neutral',
                    'confidence': 0.5,
                    'all_emotions': {'neutral': 1.0},
                    'valence': 0.0
                }
        
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return None
    
    def _calculate_valence(self, emotion_scores):
        """
        Calculate overall emotional valence from emotion scores
        
        Args:
            emotion_scores: Dict of emotion -> probability (0-100)
        
        Returns:
            float: Valence score from -1 (negative) to +1 (positive)
        """
        total_valence = 0
        total_weight = 0
        
        for emotion, score in emotion_scores.items():
            emotion_lower = emotion.lower()
            if emotion_lower in self.EMOTION_VALENCE:
                weight = score / 100.0
                valence = self.EMOTION_VALENCE[emotion_lower]
                total_valence += valence * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_valence / total_weight
    
    def aggregate_photo_emotions(self, face_emotions):
        """
        Aggregate emotions from multiple faces in a photo
        
        Args:
            face_emotions: List of emotion dicts from detect_emotion()
        
        Returns:
            dict: {
                'dominant_emotion': str,
                'emotion_counts': dict,
                'average_valence': float,
                'mood_score': float,
                'face_count': int
            }
        """
        if not face_emotions:
            return {
                'dominant_emotion': 'neutral',
                'emotion_counts': {'neutral': 0},
                'average_valence': 0.0,
                'mood_score': 0.0,
                'face_count': 0
            }
        
        # Count emotions
        emotion_counts = {}
        valences = []
        
        for face_emotion in face_emotions:
            if face_emotion:
                emotion = face_emotion.get('dominant_emotion', 'neutral')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                valences.append(face_emotion.get('valence', 0))
        
        # Determine dominant emotion for photo
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Calculate average valence
        average_valence = np.mean(valences) if valences else 0.0
        
        # Calculate mood score (-1 to +1, where +1 = all positive, -1 = all negative)
        # This considers both valence and consistency
        mood_score = average_valence
        
        aggregated = {
            'dominant_emotion': dominant_emotion,
            'emotion_counts': emotion_counts,
            'average_valence': round(average_valence, 3),
            'mood_score': round(mood_score, 3),
            'face_count': len(face_emotions)
        }
        
        logger.info(f"Photo emotions aggregated: {dominant_emotion}, mood: {mood_score:.2f}")
        
        return aggregated
    
    def classify_mood(self, mood_score):
        """
        Classify mood score into categories
        
        Args:
            mood_score: Float from -1 to +1
        
        Returns:
            str: 'very_positive', 'positive', 'neutral', 'negative', 'very_negative'
        """
        if mood_score >= 0.6:
            return 'very_positive'
        elif mood_score >= 0.2:
            return 'positive'
        elif mood_score >= -0.2:
            return 'neutral'
        elif mood_score >= -0.6:
            return 'negative'
        else:
            return 'very_negative'


# Singleton instance
_emotion_service = None

def get_emotion_service():
    """Get or create emotion service singleton"""
    global _emotion_service
    if _emotion_service is None:
        _emotion_service = EmotionService()
    return _emotion_service