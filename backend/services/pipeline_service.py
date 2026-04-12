"""
Pipeline Service - Orchestrates All Analysis Services
Phase 2.8 + Object Clustering

FIX APPLIED:
- The object cluster assignment previously opened its own SQLAlchemy Session
  (DBSession()) inside process_photo().  This inner session ran concurrently
  with the outer session in app.py and could produce FK-violation errors or
  silently roll back because the Photo row had not been committed yet in the
  outer session at that point.

  Object cluster assignment is now the RESPONSIBILITY OF THE CALLER (app.py's
  process_photos route), which already holds the committed session.
  process_photo() simply returns the object categories in results so that the
  caller can persist them using its own session after the main commit.

  If you call process_photo() from a context where you DO want the pipeline to
  handle DB writes itself, pass assign_object_clusters=True along with a
  `db_session` keyword argument.
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
    """Orchestrate all photo analysis services."""

    def __init__(self):
        self.face_service     = get_face_service()
        self.emotion_service  = get_emotion_service()
        self.object_service   = get_object_service()
        self.clip_service     = get_clip_service()
        self.metadata_service = get_metadata_service()

    def process_photo(self, photo_path, photo_id=None, progress_callback=None,
                      assign_object_clusters=False, db_session=None):
        """
        Complete photo analysis pipeline.

        Args:
            photo_path:              Path to photo file
            photo_id:                Optional photo ID for tracking
            progress_callback:       Optional function(step, message) for updates
            assign_object_clusters:  If True and db_session is provided, will
                                     write object cluster links to the database.
                                     Defaults to False so the caller controls
                                     when the commit happens.
            db_session:              SQLAlchemy session to use when
                                     assign_object_clusters=True.

        Returns:
            dict: Complete analysis results (always contains 'object_categories')
        """
        start_time = time.time()
        logger.info(f"=== Starting pipeline for photo: {photo_path} ===")

        results = {
            'photo_id':          photo_id,
            'photo_path':        str(photo_path),
            'analysis_complete': False,
            'error':             None,
            'object_categories': [],   # populated even on partial failure
        }

        try:
            # ── Step 1: Extract Metadata ──────────────────────────────
            self._progress(progress_callback, 1, "Extracting metadata...")
            metadata = self.metadata_service.extract_exif(photo_path) or {}

            if 'season' not in metadata:
                metadata['season'] = 'unknown'

            results['metadata'] = metadata
            logger.info(f"✓ Metadata extracted: {metadata.get('date_taken', 'No date')}")

            # ── Step 2: Detect Faces ──────────────────────────────────
            self._progress(progress_callback, 2, "Detecting faces...")
            face_encodings, face_locations, quality_scores = self.face_service.detect_faces(photo_path)

            results['faces']      = []
            results['face_count'] = len(face_encodings)
            logger.info(f"✓ Detected {len(face_encodings)} faces")

            # ── Step 3: Analyse Emotions Per Face ─────────────────────
            if face_encodings:
                self._progress(progress_callback, 3, f"Analyzing emotions for {len(face_encodings)} faces...")

                face_emotions = []
                for idx, (encoding, location, quality) in enumerate(
                    zip(face_encodings, face_locations, quality_scores)
                ):
                    emotion_data = self.emotion_service.detect_emotion(photo_path, location)
                    face_emotions.append(emotion_data)

                    results['faces'].append({
                        'face_index':    idx,
                        'encoding':      encoding.tolist(),
                        'location':      location,
                        'quality_score': quality,
                        'emotion':       emotion_data,
                    })

                photo_emotion = self.emotion_service.aggregate_photo_emotions(face_emotions)
                results['photo_emotion'] = photo_emotion
                logger.info(f"✓ Emotions analyzed: {photo_emotion.get('dominant_emotion', 'unknown')}")
            else:
                results['photo_emotion'] = {
                    'dominant_emotion': 'neutral',
                    'emotion_counts':   {},
                    'average_valence':  0.0,
                    'mood_score':       0.0,
                    'face_count':       0,
                }

            # ── Step 4: Detect Objects ────────────────────────────────
            self._progress(progress_callback, 4, "Detecting objects...")
            detected_objects = self.object_service.detect_objects(photo_path)
            results['objects']      = detected_objects
            results['object_count'] = len(detected_objects)
            logger.info(f"✓ Detected {len(detected_objects)} objects")

            scene_weather          = self.object_service.detect_scene_and_weather(photo_path, detected_objects)
            results['weather']     = scene_weather['weather']
            results['scene_label'] = scene_weather['scene_label']

            # ── Step 4b: Determine Object Categories ─────────────────
            # We always compute the categories list so the caller can use it.
            # The actual DB writes happen in the caller (app.py) after the main
            # commit, UNLESS assign_object_clusters=True is passed explicitly.
            self._progress(progress_callback, 5, "Computing object categories...")
            from services.object_cluster_service import get_object_cluster_service

            obj_cluster_service = get_object_cluster_service()
            categories = obj_cluster_service.get_categories_for_photo(
                detected_objects=detected_objects,
                scene_label=results.get('scene_label'),
                weather=results.get('weather'),
                face_count=results.get('face_count', 0),
            )
            results['object_categories'] = categories

            # Optionally write to DB immediately (e.g. standalone scripts)
            if assign_object_clusters and db_session and photo_id and categories:
                obj_cluster_service.assign_photo_to_clusters(db_session, photo_id, categories)
                logger.info(f"✓ Object clusters written to DB: {categories}")
            else:
                logger.info(f"✓ Object categories computed (not written): {categories}")

            # ── Step 5: Classify Scene ────────────────────────────────
            self._progress(progress_callback, 6, "Classifying scene...")
            scene_info      = self.object_service.classify_scene(detected_objects)
            results['scene'] = scene_info
            logger.info(f"✓ Scene: {scene_info.get('scene_type')} - {scene_info.get('location')}")

            # ── Step 6: Extract Clothing Colors ───────────────────────
            results['clothing_colors'] = self.object_service.get_clothing_colors(detected_objects)

            # ── Step 7: Generate CLIP Embedding ───────────────────────
            self._progress(progress_callback, 7, "Generating semantic embedding...")
            clip_embedding = self.clip_service.encode_image(photo_path)

            if clip_embedding is not None:
                results['clip_embedding'] = clip_embedding.tolist()
                logger.info("✓ CLIP embedding generated")
            else:
                results['clip_embedding'] = None
                logger.warning("× CLIP embedding failed")

            # ── Step 8: Generate Caption ──────────────────────────────
            self._progress(progress_callback, 8, "Generating caption...")
            caption = self.metadata_service.generate_caption(
                metadata=metadata,
                detected_objects=detected_objects,
                detected_faces=len(face_encodings),
                emotion=results['photo_emotion'].get('dominant_emotion'),
            )
            results['caption'] = caption
            logger.info(f"✓ Caption: {caption[:100]}...")

            # ── Done ──────────────────────────────────────────────────
            results['analysis_complete'] = True
            elapsed                      = time.time() - start_time
            results['processing_time']   = round(elapsed, 2)

            logger.info(f"=== Pipeline complete in {elapsed:.2f}s ===")
            self._progress(progress_callback, 9, "Complete!")

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results['error']             = str(e)
            results['analysis_complete'] = False

        return results

    # ── Helpers ───────────────────────────────────────────────

    def _progress(self, callback, step, message):
        if callback:
            callback(step, message)
        logger.info(f"Step {step}: {message}")

    # ── Batch Processing ──────────────────────────────────────

    def process_batch(self, photo_paths, progress_callback=None):
        results = []
        total   = len(photo_paths)

        for idx, photo_path in enumerate(photo_paths):
            logger.info(f"\n{'='*60}\nProcessing photo {idx + 1}/{total}\n{'='*60}\n")

            def batch_progress(step, message, _idx=idx):
                if progress_callback:
                    progress_callback(_idx + 1, total, step, message)

            result = self.process_photo(photo_path, progress_callback=batch_progress)
            results.append(result)

        return results

    # ── Partial Reprocessing ──────────────────────────────────

    def reprocess_faces_only(self, photo_path):
        face_encodings, face_locations, quality_scores = self.face_service.detect_faces(photo_path)
        return {
            'faces': [
                {
                    'face_index':    idx,
                    'encoding':      encoding.tolist(),
                    'location':      location,
                    'quality_score': quality,
                }
                for idx, (encoding, location, quality) in enumerate(
                    zip(face_encodings, face_locations, quality_scores)
                )
            ],
            'face_count': len(face_encodings),
        }

    def reprocess_emotions_only(self, photo_path, face_locations):
        face_emotions = [
            self.emotion_service.detect_emotion(photo_path, loc)
            for loc in face_locations
        ]
        return {
            'face_emotions': face_emotions,
            'photo_emotion': self.emotion_service.aggregate_photo_emotions(face_emotions),
        }

    # ── Stats ─────────────────────────────────────────────────

    def get_processing_stats(self):
        return {
            'face_service_ready':     self.face_service     is not None,
            'emotion_service_ready':  self.emotion_service  is not None,
            'object_service_ready':   (self.object_service.model  is not None
                                       if self.object_service else False),
            'clip_service_ready':     (self.clip_service.model    is not None
                                       if self.clip_service else False),
            'metadata_service_ready': self.metadata_service is not None,
        }


# ── Singleton ─────────────────────────────────────────────────

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline()
    return _pipeline