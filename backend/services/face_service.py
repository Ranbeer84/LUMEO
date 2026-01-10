"""
Face Service - Face Detection, Quality Assessment, and Clustering
Phase 2.1 & 2.2
"""

import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceService:
    """Handle all face-related operations with quality assessment"""
    
    def __init__(self):
        self.face_encodings = []
        self.face_locations = []
        self.quality_scores = []
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image and return encodings with locations
        
        Returns:
            tuple: (face_encodings, face_locations, quality_scores)
        """
        try:
            logger.info(f"Detecting faces in: {image_path}")
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image, model="hog")
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Calculate quality scores for each face
            quality_scores = []
            for location in face_locations:
                quality = self._calculate_face_quality(image, location)
                quality_scores.append(quality)
            
            logger.info(f"Found {len(face_encodings)} faces")
            
            return face_encodings, face_locations, quality_scores
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return [], [], []
    
    def _calculate_face_quality(self, image, face_location):
        """
        Calculate quality score for a detected face (0-1 scale)
        
        Factors:
        - Sharpness (blur detection)
        - Brightness
        - Face size
        - Face angle (frontal vs profile)
        
        Args:
            image: numpy array of full image
            face_location: tuple (top, right, bottom, left)
        
        Returns:
            float: Quality score 0-1
        """
        try:
            top, right, bottom, left = face_location
            
            # Extract face region
            face_image = image[top:bottom, left:right]
            
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # 1. Sharpness Score (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize
            
            # 2. Brightness Score
            brightness = np.mean(gray_face)
            # Optimal brightness is around 127 (middle gray)
            brightness_score = 1.0 - abs(brightness - 127) / 127
            
            # 3. Size Score (larger faces = better)
            face_height = bottom - top
            face_width = right - left
            face_area = face_height * face_width
            # Normalize assuming 200x200 is ideal
            size_score = min(face_area / 40000, 1.0)
            
            # 4. Aspect Ratio Score (closer to 1:1 = more frontal)
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            # Ideal ratio is around 0.8-1.2
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)
            aspect_score = max(0, aspect_score)
            
            # Weighted combination
            quality_score = (
                sharpness_score * 0.35 +
                brightness_score * 0.25 +
                size_score * 0.25 +
                aspect_score * 0.15
            )
            
            return round(quality_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating face quality: {str(e)}")
            return 0.5  # Default medium quality
    
    def cluster_faces(self, encodings, quality_scores=None, min_samples=1, eps=0.6):
        """
        Cluster face encodings using DBSCAN with quality-weighted selection
        
        Args:
            encodings: List of face encodings
            quality_scores: Optional quality scores for each face
            min_samples: Minimum samples for DBSCAN
            eps: DBSCAN distance threshold
        
        Returns:
            numpy.array: Cluster labels for each face
        """
        if len(encodings) == 0:
            return np.array([])
        
        logger.info(f"Clustering {len(encodings)} faces with eps={eps}")
        
        # Convert to numpy array
        encodings_array = np.array(encodings)
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(encodings_array)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")
        
        return labels
    
    def select_best_face(self, faces_data):
        """
        Select the best representative face from a cluster based on quality
        
        Args:
            faces_data: List of dicts with 'encoding', 'location', 'quality_score'
        
        Returns:
            dict: Best face data
        """
        if not faces_data:
            return None
        
        # Sort by quality score
        best_face = max(faces_data, key=lambda x: x.get('quality_score', 0))
        
        logger.info(f"Selected best face with quality score: {best_face.get('quality_score', 0)}")
        
        return best_face
    
    def extract_face_thumbnail(self, image_path, face_location, output_path, padding=40):
        """
        Extract and save face thumbnail with padding
        
        Args:
            image_path: Path to original image
            face_location: tuple (top, right, bottom, left)
            output_path: Where to save thumbnail
            padding: Pixels to add around face
        
        Returns:
            bool: Success status
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return False
            
            top, right, bottom, left = face_location
            
            # Add padding
            height, width = image.shape[:2]
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(height, bottom + padding)
            right = min(width, right + padding)
            
            # Extract face region
            face_image = image[top:bottom, left:right]
            
            # Resize to consistent thumbnail size
            face_image = cv2.resize(face_image, (200, 200))
            
            # Save thumbnail
            cv2.imwrite(str(output_path), face_image)
            
            logger.info(f"Thumbnail saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting face thumbnail: {str(e)}")
            return False
    
    def compare_faces(self, known_encoding, unknown_encoding, tolerance=0.6):
        """
        Compare two face encodings
        
        Args:
            known_encoding: Reference face encoding
            unknown_encoding: Face to compare
            tolerance: Match threshold (lower = stricter)
        
        Returns:
            tuple: (is_match, distance)
        """
        distance = np.linalg.norm(known_encoding - unknown_encoding)
        is_match = distance <= tolerance
        
        return is_match, round(distance, 3)


# Singleton instance
_face_service = None

def get_face_service():
    """Get or create face service singleton"""
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service