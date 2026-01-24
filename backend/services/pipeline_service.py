"""
Pipeline Service - Orchestrates All Analysis Services
Phase 2.8
"""

from .face_service import get_face_service
from .emotion_service import get_emotion_service
from .object_service import get_object_service
from .clip_service import get_clip_service
from .metadata_service import get_metadata_service
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrate all photo analysis services"""
    
    def __init__(self):
        self.face_service = get_face_service()
        self.emotion_service = get_emotion_service()
        self.object_service = get_object_service()
        self.clip_service = get_clip_service()
        self.metadata_service = get_metadata_service()
    
    def process_photo(self, photo_path, photo_id=None, progress_callback=None):
        """
        Complete photo analysis pipeline
        
        Args:
            photo_path: Path to photo file
            photo_id: Optional photo ID for tracking
            progress_callback: Optional function(step, message) for progress updates
        
        Returns:
            dict: Complete analysis results
        """
        start_time = time.time()
        
        logger.info(f"=== Starting pipeline for photo: {photo_path} ===")
        
        results = {
            'photo_id': photo_id,
            'photo_path': str(photo_path),
            'analysis_complete': False,
            'error': None
        }
        
        try:
            # # Step 1: Extract Metadata (2-3 seconds)
            # self._progress(progress_callback, 1, "Extracting metadata...")
            # metadata = self.metadata_service.extract_exif(photo_path)
            # results['metadata'] = metadata
            # logger.info(f"✓ Metadata extracted: {metadata.get('date_taken', 'No date')}")

            # Step 1: Extract Metadata (2-3 seconds)
            self._progress(progress_callback, 1, "Extracting metadata...")
            metadata = self.metadata_service.extract_exif(photo_path)
            
            # --- SAFETY NET ---
            if metadata is None:
                metadata = {}
                
            # Force 'season' to exist so Step 8 doesn't crash
            if 'season' not in metadata:
                metadata['season'] = 'unknown'

            results['metadata'] = metadata
            logger.info(f"✓ Metadata extracted: {metadata.get('date_taken', 'No date')}")
            
            # Step 2: Detect Faces (3-5 seconds)
            self._progress(progress_callback, 2, "Detecting faces...")
            face_encodings, face_locations, quality_scores = self.face_service.detect_faces(photo_path)
            
            results['faces'] = []
            results['face_count'] = len(face_encodings)
            
            logger.info(f"✓ Detected {len(face_encodings)} faces")
            
            # Step 3: Detect Emotions for Each Face (2-4 seconds per face)
            if face_encodings:
                self._progress(progress_callback, 3, f"Analyzing emotions for {len(face_encodings)} faces...")
                
                face_emotions = []
                for idx, (encoding, location, quality) in enumerate(zip(face_encodings, face_locations, quality_scores)):
                    emotion_data = self.emotion_service.detect_emotion(photo_path, location)
                    
                    face_emotions.append(emotion_data)
                    
                    results['faces'].append({
                        'face_index': idx,
                        'encoding': encoding.tolist(),  # Convert numpy to list for JSON
                        'location': location,
                        'quality_score': quality,
                        'emotion': emotion_data
                    })
                
                # Aggregate emotions for the photo
                photo_emotion = self.emotion_service.aggregate_photo_emotions(face_emotions)
                results['photo_emotion'] = photo_emotion
                
                logger.info(f"✓ Emotions analyzed: {photo_emotion.get('dominant_emotion', 'unknown')}")
            else:
                results['photo_emotion'] = {
                    'dominant_emotion': 'neutral',
                    'emotion_counts': {},
                    'average_valence': 0.0,
                    'mood_score': 0.0,
                    'face_count': 0
                }
            
            # Step 4: Detect Objects (3-5 seconds)
            self._progress(progress_callback, 4, "Detecting objects...")
            detected_objects = self.object_service.detect_objects(photo_path)
            results['objects'] = detected_objects
            results['object_count'] = len(detected_objects)
            
            logger.info(f"✓ Detected {len(detected_objects)} objects")
            
            # Step 5: Classify Scene (< 1 second)
            self._progress(progress_callback, 5, "Classifying scene...")
            scene_info = self.object_service.classify_scene(detected_objects)
            results['scene'] = scene_info
            
            logger.info(f"✓ Scene: {scene_info.get('scene_type', 'unknown')} - {scene_info.get('location', 'unknown')}")
            
            # Step 6: Extract Clothing Colors (< 1 second)
            clothing_colors = self.object_service.get_clothing_colors(detected_objects)
            results['clothing_colors'] = clothing_colors
            
            # Step 7: Generate CLIP Embedding (2-3 seconds)
            self._progress(progress_callback, 6, "Generating semantic embedding...")
            clip_embedding = self.clip_service.encode_image(photo_path)
            
            if clip_embedding is not None:
                results['clip_embedding'] = clip_embedding.tolist()  # Convert to list for JSON
                logger.info(f"✓ CLIP embedding generated")
            else:
                results['clip_embedding'] = None
                logger.warning("× CLIP embedding failed")

            # Step 8: Generate Caption
            self._progress(progress_callback, 7, "Generating caption...")
            caption = self.metadata_service.generate_caption(
                metadata=metadata,  # This is now safe because of Step 1
                detected_objects=detected_objects,
                detected_faces=len(face_encodings),
                emotion=results['photo_emotion'].get('dominant_emotion')
            )
            results['caption'] = caption
            
            logger.info(f"✓ Caption: {caption[:100]}...")
            
            # Mark as complete
            results['analysis_complete'] = True
            
            elapsed = time.time() - start_time
            results['processing_time'] = round(elapsed, 2)
            
            logger.info(f"=== Pipeline complete in {elapsed:.2f}s ===")
            
            self._progress(progress_callback, 8, "Complete!")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results['error'] = str(e)
            results['analysis_complete'] = False
        
        return results
    
    def _progress(self, callback, step, message):
        """Helper to call progress callback if provided"""
        if callback:
            callback(step, message)
        logger.info(f"Step {step}: {message}")
    
    def process_batch(self, photo_paths, progress_callback=None):
        """
        Process multiple photos
        
        Args:
            photo_paths: List of photo paths
            progress_callback: Optional function(photo_index, total, step, message)
        
        Returns:
            list: List of analysis results
        """
        results = []
        total = len(photo_paths)
        
        for idx, photo_path in enumerate(photo_paths):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing photo {idx + 1}/{total}")
            logger.info(f"{'='*60}\n")
            
            def batch_progress(step, message):
                if progress_callback:
                    progress_callback(idx + 1, total, step, message)
            
            result = self.process_photo(photo_path, progress_callback=batch_progress)
            results.append(result)
        
        return results
    
    def reprocess_faces_only(self, photo_path):
        """
        Quick reprocessing of just faces (useful for re-clustering)
        
        Args:
            photo_path: Path to photo
        
        Returns:
            dict: Face data only
        """
        face_encodings, face_locations, quality_scores = self.face_service.detect_faces(photo_path)
        
        faces = []
        for idx, (encoding, location, quality) in enumerate(zip(face_encodings, face_locations, quality_scores)):
            faces.append({
                'face_index': idx,
                'encoding': encoding.tolist(),
                'location': location,
                'quality_score': quality
            })
        
        return {
            'faces': faces,
            'face_count': len(faces)
        }
    
    def reprocess_emotions_only(self, photo_path, face_locations):
        """
        Reprocess emotions only (useful if emotion model updated)
        
        Args:
            photo_path: Path to photo
            face_locations: List of face locations
        
        Returns:
            dict: Emotion data
        """
        face_emotions = []
        for location in face_locations:
            emotion_data = self.emotion_service.detect_emotion(photo_path, location)
            face_emotions.append(emotion_data)
        
        photo_emotion = self.emotion_service.aggregate_photo_emotions(face_emotions)
        
        return {
            'face_emotions': face_emotions,
            'photo_emotion': photo_emotion
        }
    
    def get_processing_stats(self):
        """
        Get statistics about the pipeline
        
        Returns:
            dict: Stats about loaded models and readiness
        """
        return {
            'face_service_ready': self.face_service is not None,
            'emotion_service_ready': self.emotion_service is not None,
            'object_service_ready': self.object_service.model is not None if self.object_service else False,
            'clip_service_ready': self.clip_service.model is not None if self.clip_service else False,
            'metadata_service_ready': self.metadata_service is not None
        }


# Singleton instance
_pipeline = None

def get_pipeline():
    """Get or create pipeline singleton"""
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline()
    return _pipeline