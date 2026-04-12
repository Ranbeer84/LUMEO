"""
Retrieval Service - Hybrid Vector + Filter Search
Phase 3.2: Implement Vector Similarity Search

FIXES APPLIED:
1. query_embedding is now always converted to a Python list before being
   handed to pgvector.  Passing a raw numpy array can produce silent type
   errors that make ALL distance values identical, so every photo looks
   equally (ir)relevant.

2. ORDER BY now uses the SQLAlchemy column expression directly instead of
   the string alias 'distance'.  The string alias is resolved by PostgreSQL,
   but using the expression avoids any ambiguity and is the recommended
   pgvector-python pattern.

3. hybrid_search now applies a default min_similarity of 0.20.  Previously
   the default was 0.0, meaning every photo in the filtered set was returned
   regardless of how unrelated it was.

4. _apply_filters: the objects subquery is now OR-based (a photo that
   contains ANY of the requested objects matches) which is the expected
   user-facing behaviour.  The old IN() on the subquery already achieved
   this; the comment clarifies the intent.
"""

from models import Session, Photo, Cluster, FaceEmbedding, DetectedObject, PhotoCluster
from sqlalchemy import and_, or_, func, asc
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_list(embedding) -> list:
    """
    Convert a query embedding to a plain Python list for pgvector.

    pgvector-python accepts numpy arrays in theory, but in practice certain
    numpy dtypes (e.g. float32 vs float64) or zero-copy views can produce
    silent errors where every distance comes out identical.  Explicitly
    converting to list(float) is the safest approach.
    """
    if isinstance(embedding, np.ndarray):
        return embedding.astype(float).tolist()
    if hasattr(embedding, 'tolist'):
        return embedding.tolist()
    return list(embedding)


class RetrievalService:
    """
    Hybrid search combining:
    1. Semantic similarity (pgvector CLIP embeddings)
    2. Structured filters (people, emotions, dates, objects, scene, weather …)
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # SEMANTIC SEARCH
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query_embedding,
        top_k: int = 20,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Pure semantic search using CLIP embeddings.

        Args:
            query_embedding: Query vector (512-dim numpy array or list)
            top_k:           Number of results to return
            min_similarity:  Minimum cosine similarity threshold (0-1)

        Returns:
            List of photo dicts with similarity scores, sorted best-first.
        """
        session = Session()

        try:
            logger.info(f"Semantic search: top_k={top_k}, min_similarity={min_similarity}")

            # FIX 1: convert to plain list so pgvector never misinterprets dtype
            emb_list = _to_list(query_embedding)

            # FIX 2: order by the column expression, not a string alias
            distance_col = Photo.clip_embedding.cosine_distance(emb_list).label('distance')

            rows = (
                session.query(Photo, distance_col)
                .filter(Photo.clip_embedding.isnot(None))
                .order_by(asc(distance_col))          # ascending → closest first
                .limit(top_k)
                .all()
            )

            photos = []
            for photo, distance in rows:
                # cosine distance (0=identical, 2=opposite) → similarity (1=identical, 0=opposite)
                similarity = max(0.0, 1.0 - (float(distance) / 2.0))

                if similarity < min_similarity:
                    continue

                photos.append({
                    'photo_id':         photo.photo_id,
                    'filename':         photo.filename,
                    'path':             photo.path,
                    'similarity':       round(similarity, 3),
                    'distance':         round(float(distance), 3),
                    'caption':          photo.caption,
                    'scene_type':       photo.scene_type,
                    'location':         photo.location_type,
                    'activity':         photo.activity,
                    'dominant_emotion': photo.dominant_emotion,
                    'mood_score':       photo.mood_score,
                    'date_taken':       photo.date_taken.isoformat() if photo.date_taken else None,
                    'season':           photo.season,
                    'time_of_day':      photo.time_of_day,
                })

            logger.info(f"✓ Semantic search: {len(photos)} photos (similarity >= {min_similarity})")
            return photos

        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------
    # HYBRID SEARCH
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_embedding,
        filters: Optional[Dict] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Hybrid search: semantic similarity + structured filters.

        Args:
            query_embedding: Query vector (512-dim numpy array or list)
            filters: Optional dict — see _apply_filters for supported keys.
            top_k:   Maximum number of results.

        Returns:
            List of photo dicts sorted by relevance.
        """
        session = Session()

        try:
            logger.info(f"Hybrid search with filters: {filters}")

            # FIX 1: plain list for pgvector
            emb_list = _to_list(query_embedding)

            # FIX 2: column expression for ORDER BY
            distance_col = Photo.clip_embedding.cosine_distance(emb_list).label('distance')

            query = (
                session.query(Photo, distance_col)
                .filter(Photo.clip_embedding.isnot(None))
            )

            if filters:
                query = self._apply_filters(query, filters, session)

            # Fetch a bigger pool, then trim after similarity filtering
            rows = query.order_by(asc(distance_col)).limit(top_k * 3).all()

            # FIX 3: apply a sensible default minimum similarity in hybrid mode
            min_sim = (filters or {}).get('min_similarity', 0.20)

            photos = []
            for photo, distance in rows:
                similarity = max(0.0, 1.0 - (float(distance) / 2.0))

                if similarity < min_sim:
                    continue

                people  = self._get_photo_people(photo.photo_id, session)
                objects = self._get_photo_objects(photo.photo_id, session)

                photos.append({
                    'photo_id':         photo.photo_id,
                    'filename':         photo.filename,
                    'path':             photo.path,
                    'similarity':       round(similarity, 3),
                    'caption':          photo.caption,
                    'people':           people,
                    'objects':          objects,
                    'scene_type':       photo.scene_type,
                    'location':         photo.location_type,
                    'activity':         photo.activity,
                    'dominant_emotion': photo.dominant_emotion,
                    'mood_score':       photo.mood_score,
                    'date_taken':       photo.date_taken.isoformat() if photo.date_taken else None,
                    'season':           photo.season,
                    'time_of_day':      photo.time_of_day,
                    'match_reasons':    self._get_match_reasons(photo, filters, similarity),
                })

                if len(photos) >= top_k:
                    break

            logger.info(f"✓ Hybrid search: {len(photos)} photos")
            return photos

        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------
    # FILTER APPLICATION
    # ------------------------------------------------------------------

    def _apply_filters(self, query, filters: Dict, session):
        """Apply structured filters to a SQLAlchemy query."""

        # ── People (photo must contain ALL requested people) ──────────
        if filters.get('people'):
            people_names = filters['people']
            logger.info(f"Filtering by people: {people_names}")

            clusters = session.query(Cluster).filter(
                Cluster.name.in_(people_names)
            ).all()

            if clusters:
                for cluster in clusters:
                    subq = (
                        session.query(PhotoCluster.photo_id)
                        .filter(PhotoCluster.cluster_id == cluster.cluster_id)
                        .subquery()
                    )
                    query = query.filter(Photo.photo_id.in_(subq))

        # ── Emotions ──────────────────────────────────────────────────
        if filters.get('emotions'):
            query = query.filter(Photo.dominant_emotion.in_(filters['emotions']))

        # ── Objects (photo must contain ANY of the requested objects) ─
        if filters.get('objects'):
            object_labels = filters['objects']
            logger.info(f"Filtering by objects: {object_labels}")

            object_photo_ids = (
                session.query(DetectedObject.photo_id)
                .filter(DetectedObject.label.in_(object_labels))
                .distinct()
                .subquery()
            )
            query = query.filter(Photo.photo_id.in_(object_photo_ids))

        # ── Scene type ────────────────────────────────────────────────
        if filters.get('scene_type'):
            query = query.filter(Photo.scene_type == filters['scene_type'])

        # ── Location ──────────────────────────────────────────────────
        if filters.get('location'):
            query = query.filter(Photo.location_type == filters['location'])

        # ── Date range ────────────────────────────────────────────────
        if filters.get('date_range'):
            start_date, end_date = filters['date_range']
            query = query.filter(
                and_(
                    Photo.date_taken >= start_date,
                    Photo.date_taken <= end_date,
                )
            )

        # ── Season ────────────────────────────────────────────────────
        if filters.get('season'):
            query = query.filter(Photo.season == filters['season'])

        # ── Weather ───────────────────────────────────────────────────
        if filters.get('weather'):
            query = query.filter(Photo.weather == filters['weather'])

        # ── Time of day ───────────────────────────────────────────────
        if filters.get('time_of_day'):
            query = query.filter(Photo.time_of_day == filters['time_of_day'])

        # ── Color (object colour) ─────────────────────────────────────
        if filters.get('color'):
            color = filters['color']
            color_photo_ids = (
                session.query(DetectedObject.photo_id)
                .filter(DetectedObject.color_name.ilike(f'%{color}%'))
                .distinct()
                .subquery()
            )
            query = query.filter(Photo.photo_id.in_(color_photo_ids))

        return query

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _get_photo_people(self, photo_id: str, session) -> List[str]:
        """Return cluster names (people) visible in this photo."""
        photo_clusters = session.query(PhotoCluster).filter_by(photo_id=photo_id).all()
        people = []
        for pc in photo_clusters:
            cluster = session.query(Cluster).filter_by(cluster_id=pc.cluster_id).first()
            if cluster:
                people.append(cluster.name)
        return people

    def _get_photo_objects(self, photo_id: str, session) -> List[Dict]:
        """Return detected objects for this photo."""
        objs = session.query(DetectedObject).filter_by(photo_id=photo_id).all()
        return [
            {
                'label':      obj.label,
                'confidence': round(obj.confidence, 2),
                'color':      obj.color_name,
            }
            for obj in objs
        ]

    def _get_match_reasons(self, photo, filters: Optional[Dict], similarity: float) -> List[str]:
        """Human-readable explanation of why this photo matched."""
        reasons = []

        if similarity >= 0.7:
            reasons.append(f"High semantic match ({similarity:.2f})")
        elif similarity >= 0.5:
            reasons.append(f"Moderate semantic match ({similarity:.2f})")
        else:
            reasons.append(f"Semantic similarity: {similarity:.2f}")

        if not filters:
            return reasons

        if filters.get('emotions') and photo.dominant_emotion in filters['emotions']:
            reasons.append(f"Emotion: {photo.dominant_emotion}")

        if filters.get('scene_type') and photo.scene_type == filters['scene_type']:
            reasons.append(f"Scene: {photo.scene_type}")

        if filters.get('location') and photo.location_type == filters['location']:
            reasons.append(f"Location: {photo.location_type}")

        if filters.get('season') and photo.season == filters['season']:
            reasons.append(f"Season: {photo.season}")

        if filters.get('time_of_day') and photo.time_of_day == filters['time_of_day']:
            reasons.append(f"Time: {photo.time_of_day}")

        if filters.get('weather') and photo.weather == filters['weather']:
            reasons.append(f"Weather: {photo.weather}")

        return reasons

    # ------------------------------------------------------------------
    # SIMILAR-PHOTO SEARCH
    # ------------------------------------------------------------------

    def search_by_similar_photo(
        self,
        photo_id: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """Find photos visually similar to a given photo."""
        session = Session()
        try:
            photo = session.query(Photo).filter_by(photo_id=photo_id).first()

            if not photo or photo.clip_embedding is None:
                logger.warning(f"Photo {photo_id} not found or has no embedding")
                return []

            query_embedding = np.array(photo.clip_embedding)
            session.close()

            results = self.semantic_search(query_embedding, top_k=top_k + 1)

            if exclude_self:
                results = [r for r in results if r['photo_id'] != photo_id]

            return results[:top_k]

        except Exception as e:
            logger.error(f"Similar photo search error: {str(e)}")
            return []
        finally:
            try:
                session.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_retrieval_stats(self) -> Dict:
        """Statistics about the searchable photo collection."""
        session = Session()
        try:
            return {
                'total_photos':        session.query(Photo).count(),
                'searchable_photos':   session.query(Photo).filter(Photo.clip_embedding.isnot(None)).count(),
                'photos_with_people':  session.query(Photo).join(PhotoCluster).distinct().count(),
                'photos_with_emotions': session.query(Photo).filter(Photo.dominant_emotion.isnot(None)).count(),
                'photos_with_objects': session.query(Photo).join(DetectedObject).distinct().count(),
                'total_people':        session.query(Cluster).count(),
                'total_objects':       session.query(DetectedObject).count(),
            }
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            return {}
        finally:
            session.close()


# Singleton
_retrieval_service = None

def get_retrieval_service():
    """Get or create retrieval service singleton."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service