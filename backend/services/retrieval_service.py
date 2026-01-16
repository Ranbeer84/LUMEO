"""
Retrieval Service - Hybrid Vector + Filter Search
Phase 3.2: Implement Vector Similarity Search

Uses pgvector for semantic search combined with traditional filters
"""

from models import Session, Photo, Cluster, FaceEmbedding, DetectedObject, PhotoCluster
from sqlalchemy import and_, or_, func
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Hybrid search combining:
    1. Semantic similarity (pgvector)
    2. Structured filters (people, emotions, dates, objects)
    """
    
    def __init__(self):
        pass
    
    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Pure semantic search using CLIP embeddings
        
        Args:
            query_embedding: Query vector (512-dim)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of photo dicts with similarity scores
        """
        session = Session()
        
        try:
            logger.info(f"Semantic search: top_k={top_k}, min_similarity={min_similarity}")
            
            # pgvector cosine distance query
            # Note: <=> operator returns distance (0 = identical, 2 = opposite)
            # We convert to similarity: similarity = 1 - (distance / 2)
            
            results = session.query(
                Photo,
                Photo.clip_embedding.cosine_distance(query_embedding).label('distance')
            ).filter(
                Photo.clip_embedding.isnot(None)  # Only photos with embeddings
            ).order_by(
                'distance'  # Order by distance (ascending = most similar first)
            ).limit(top_k).all()
            
            photos = []
            for photo, distance in results:
                # Convert distance to similarity (0-1, higher = more similar)
                similarity = 1.0 - (float(distance) / 2.0)
                
                # Filter by minimum similarity
                if similarity < min_similarity:
                    continue
                
                photos.append({
                    'photo_id': photo.photo_id,
                    'filename': photo.filename,
                    'path': photo.path,
                    'similarity': round(similarity, 3),
                    'distance': round(float(distance), 3),
                    'caption': photo.caption,
                    'scene_type': photo.scene_type,
                    'location': photo.location_type,
                    'activity': photo.activity,
                    'dominant_emotion': photo.dominant_emotion,
                    'mood_score': photo.mood_score,
                    'date_taken': photo.date_taken.isoformat() if photo.date_taken else None,
                    'season': photo.season,
                    'time_of_day': photo.time_of_day
                })
            
            session.close()
            
            logger.info(f"✓ Found {len(photos)} photos (similarity >= {min_similarity})")
            
            return photos
            
        except Exception as e:
            session.close()
            logger.error(f"Semantic search error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Hybrid search: semantic similarity + structured filters
        
        Args:
            query_embedding: Query vector
            filters: Dict with optional filters:
                - people: List[str] - cluster names (e.g., ["Mom", "Dad"])
                - emotions: List[str] - emotion labels (e.g., ["happy"])
                - objects: List[str] - object labels (e.g., ["cake", "balloons"])
                - scene_type: str - "indoor" or "outdoor"
                - location: str - e.g., "beach", "office"
                - date_range: Tuple[datetime, datetime]
                - season: str - e.g., "summer"
                - time_of_day: str - e.g., "evening"
                - min_similarity: float - minimum cosine similarity
                - color: str - clothing/object color
            top_k: Number of results
        
        Returns:
            List of photo dicts with similarity and match info
        """
        session = Session()
        
        try:
            logger.info(f"Hybrid search with filters: {filters}")
            
            # Start with base query
            query = session.query(
                Photo,
                Photo.clip_embedding.cosine_distance(query_embedding).label('distance')
            ).filter(
                Photo.clip_embedding.isnot(None)
            )
            
            # Apply filters if provided
            if filters:
                query = self._apply_filters(query, filters, session)
            
            # Order by similarity and limit
            results = query.order_by('distance').limit(top_k * 2).all()  # Get more, then filter
            
            # Process results
            photos = []
            for photo, distance in results:
                similarity = 1.0 - (float(distance) / 2.0)
                
                # Check minimum similarity if specified
                min_sim = filters.get('min_similarity', 0.0) if filters else 0.0
                if similarity < min_sim:
                    continue
                
                # Get people in photo
                people = self._get_photo_people(photo.photo_id, session)
                
                # Get objects in photo
                objects = self._get_photo_objects(photo.photo_id, session)
                
                photo_data = {
                    'photo_id': photo.photo_id,
                    'filename': photo.filename,
                    'path': photo.path,
                    'similarity': round(similarity, 3),
                    'caption': photo.caption,
                    'people': people,
                    'objects': objects,
                    'scene_type': photo.scene_type,
                    'location': photo.location_type,
                    'activity': photo.activity,
                    'dominant_emotion': photo.dominant_emotion,
                    'mood_score': photo.mood_score,
                    'date_taken': photo.date_taken.isoformat() if photo.date_taken else None,
                    'season': photo.season,
                    'time_of_day': photo.time_of_day,
                    'match_reasons': self._get_match_reasons(photo, filters, similarity)
                }
                
                photos.append(photo_data)
                
                if len(photos) >= top_k:
                    break
            
            session.close()
            
            logger.info(f"✓ Hybrid search found {len(photos)} photos")
            
            return photos
            
        except Exception as e:
            session.close()
            logger.error(f"Hybrid search error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _apply_filters(self, query, filters: Dict, session):
        """Apply structured filters to query"""
        
        # Filter by people
        if 'people' in filters and filters['people']:
            people_names = filters['people']
            logger.info(f"Filtering by people: {people_names}")
            
            # Find cluster IDs for these names
            clusters = session.query(Cluster).filter(
                Cluster.name.in_(people_names)
            ).all()
            
            if clusters:
                cluster_ids = [c.cluster_id for c in clusters]
                
                # Filter photos that have these clusters
                query = query.join(
                    PhotoCluster,
                    Photo.photo_id == PhotoCluster.photo_id
                ).filter(
                    PhotoCluster.cluster_id.in_(cluster_ids)
                )
        
        # Filter by emotions
        if 'emotions' in filters and filters['emotions']:
            emotions = filters['emotions']
            logger.info(f"Filtering by emotions: {emotions}")
            query = query.filter(Photo.dominant_emotion.in_(emotions))
        
        # Filter by objects
        if 'objects' in filters and filters['objects']:
            object_labels = filters['objects']
            logger.info(f"Filtering by objects: {object_labels}")
            
            # Subquery to find photos with these objects
            object_photos = session.query(DetectedObject.photo_id).filter(
                DetectedObject.label.in_(object_labels)
            ).distinct()
            
            query = query.filter(Photo.photo_id.in_(object_photos))
        
        # Filter by scene type
        if 'scene_type' in filters and filters['scene_type']:
            scene = filters['scene_type']
            logger.info(f"Filtering by scene: {scene}")
            query = query.filter(Photo.scene_type == scene)
        
        # Filter by location
        if 'location' in filters and filters['location']:
            location = filters['location']
            logger.info(f"Filtering by location: {location}")
            query = query.filter(Photo.location_type == location)
        
        # Filter by date range
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            logger.info(f"Filtering by date range: {start_date} to {end_date}")
            query = query.filter(
                and_(
                    Photo.date_taken >= start_date,
                    Photo.date_taken <= end_date
                )
            )
        
        # Filter by season
        if 'season' in filters and filters['season']:
            season = filters['season']
            logger.info(f"Filtering by season: {season}")
            query = query.filter(Photo.season == season)
        
        # Filter by time of day
        if 'time_of_day' in filters and filters['time_of_day']:
            time_of_day = filters['time_of_day']
            logger.info(f"Filtering by time of day: {time_of_day}")
            query = query.filter(Photo.time_of_day == time_of_day)
        
        # Filter by color (check detected objects)
        if 'color' in filters and filters['color']:
            color = filters['color']
            logger.info(f"Filtering by color: {color}")
            
            color_photos = session.query(DetectedObject.photo_id).filter(
                DetectedObject.color_name.ilike(f'%{color}%')
            ).distinct()
            
            query = query.filter(Photo.photo_id.in_(color_photos))
        
        return query
    
    def _get_photo_people(self, photo_id: str, session) -> List[str]:
        """Get list of people names in a photo"""
        photo_clusters = session.query(PhotoCluster).filter_by(
            photo_id=photo_id
        ).all()
        
        people = []
        for pc in photo_clusters:
            cluster = session.query(Cluster).filter_by(
                cluster_id=pc.cluster_id
            ).first()
            if cluster:
                people.append(cluster.name)
        
        return people
    
    def _get_photo_objects(self, photo_id: str, session) -> List[Dict]:
        """Get detected objects in a photo"""
        objects = session.query(DetectedObject).filter_by(
            photo_id=photo_id
        ).all()
        
        return [
            {
                'label': obj.label,
                'confidence': round(obj.confidence, 2),
                'color': obj.color_name
            }
            for obj in objects
        ]
    
    def _get_match_reasons(self, photo, filters: Optional[Dict], similarity: float) -> List[str]:
        """
        Explain why this photo matched the query
        
        Returns list of reason strings for transparency
        """
        reasons = []
        
        # Semantic similarity
        if similarity >= 0.7:
            reasons.append(f"High semantic match ({similarity:.2f})")
        elif similarity >= 0.5:
            reasons.append(f"Moderate semantic match ({similarity:.2f})")
        else:
            reasons.append(f"Semantic similarity: {similarity:.2f}")
        
        if not filters:
            return reasons
        
        # Check each filter
        if 'emotions' in filters and photo.dominant_emotion in filters['emotions']:
            reasons.append(f"Emotion: {photo.dominant_emotion}")
        
        if 'scene_type' in filters and photo.scene_type == filters['scene_type']:
            reasons.append(f"Scene: {photo.scene_type}")
        
        if 'location' in filters and photo.location_type == filters['location']:
            reasons.append(f"Location: {photo.location_type}")
        
        if 'season' in filters and photo.season == filters['season']:
            reasons.append(f"Season: {photo.season}")
        
        if 'time_of_day' in filters and photo.time_of_day == filters['time_of_day']:
            reasons.append(f"Time: {photo.time_of_day}")
        
        return reasons
    
    def search_by_similar_photo(
        self,
        photo_id: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Find photos similar to a given photo
        
        Args:
            photo_id: Reference photo ID
            top_k: Number of similar photos to return
            exclude_self: Whether to exclude the reference photo
        
        Returns:
            List of similar photos
        """
        session = Session()
        
        try:
            # Get reference photo embedding
            photo = session.query(Photo).filter_by(photo_id=photo_id).first()
            
            if not photo or photo.clip_embedding is None:
                session.close()
                logger.warning(f"Photo {photo_id} not found or has no embedding")
                return []
            
            # Use its embedding as query
            query_embedding = np.array(photo.clip_embedding)
            
            # Search
            results = self.semantic_search(query_embedding, top_k=top_k + 1)
            
            # Remove self if requested
            if exclude_self:
                results = [r for r in results if r['photo_id'] != photo_id]
            
            session.close()
            
            return results[:top_k]
            
        except Exception as e:
            session.close()
            logger.error(f"Similar photo search error: {str(e)}")
            return []
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about searchable photos"""
        session = Session()
        
        try:
            stats = {
                'total_photos': session.query(Photo).count(),
                'searchable_photos': session.query(Photo).filter(
                    Photo.clip_embedding.isnot(None)
                ).count(),
                'photos_with_people': session.query(Photo).join(
                    PhotoCluster
                ).distinct().count(),
                'photos_with_emotions': session.query(Photo).filter(
                    Photo.dominant_emotion.isnot(None)
                ).count(),
                'photos_with_objects': session.query(Photo).join(
                    DetectedObject
                ).distinct().count(),
                'total_people': session.query(Cluster).count(),
                'total_objects': session.query(DetectedObject).count()
            }
            
            session.close()
            return stats
            
        except Exception as e:
            session.close()
            logger.error(f"Stats error: {str(e)}")
            return {}


# Singleton instance
_retrieval_service = None

def get_retrieval_service():
    """Get or create retrieval service singleton"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


# Example usage
if __name__ == "__main__":
    service = RetrievalService()
    
    # Example: Search for beach photos
    from services.clip_service import get_clip_service
    
    clip = get_clip_service()
    query_emb = clip.encode_text("beach sunset with family")
    
    # Pure semantic search
    results = service.semantic_search(query_emb, top_k=5)
    print(f"Semantic search found {len(results)} photos")
    
    # Hybrid search with filters
    filters = {
        'emotions': ['happy'],
        'scene_type': 'outdoor',
        'min_similarity': 0.5
    }
    
    results = service.hybrid_search(query_emb, filters, top_k=5)
    print(f"Hybrid search found {len(results)} photos")
    for r in results:
        print(f"  - {r['filename']}: {r['similarity']} - {r['match_reasons']}")
    
    # Get stats
    stats = service.get_retrieval_stats()
    print(f"Stats: {stats}")