"""
Object Cluster Service
Groups photos by detected object categories.
Photos with no faces go into the "Objects Only" section automatically.

FIX APPLIED:
- get_all_clusters: thumbnail_url was built as /thumbnails/{path.name} which
  points at the face-thumbnail folder.  Object cluster thumbnails are full
  photos, so the correct URL is /uploads/{photo.filename}.
- get_photos_in_cluster: likewise uses /uploads/{photo.filename}.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# YOLO label  →  user-friendly cluster name
# ─────────────────────────────────────────────────────────────
LABEL_TO_CATEGORY = {
    # Vehicles
    'bicycle':       'Bikes',
    'motorcycle':    'Bikes',
    'car':           'Cars',
    'truck':         'Cars',
    'bus':           'Cars',
    'boat':          'Boats',
    'airplane':      'Airplanes',
    'train':         'Trains',

    # Animals
    'dog':           'Dogs',
    'cat':           'Cats',
    'bird':          'Birds',
    'horse':         'Horses',
    'cow':           'Cows',
    'sheep':         'Animals',
    'elephant':      'Animals',
    'bear':          'Animals',
    'zebra':         'Animals',
    'giraffe':       'Animals',

    # Food
    'sandwich':      'Food',
    'pizza':         'Food',
    'cake':          'Food',
    'hot dog':       'Food',
    'donut':         'Food',
    'apple':         'Food',
    'banana':        'Food',
    'orange':        'Food',
    'broccoli':      'Food',
    'carrot':        'Food',

    # Drinks
    'bottle':        'Drinks',
    'wine glass':    'Drinks',
    'cup':           'Drinks',

    # Electronics
    'laptop':        'Electronics',
    'tv':            'Electronics',
    'cell phone':    'Electronics',
    'keyboard':      'Electronics',
    'mouse':         'Electronics',
    'remote':        'Electronics',

    # Sports
    'sports ball':   'Sports',
    'tennis racket': 'Sports',
    'baseball bat':  'Sports',
    'skateboard':    'Sports',
    'skis':          'Sports',
    'snowboard':     'Sports',
    'kite':          'Sports',
    'frisbee':       'Sports',
    'surfboard':     'Sports',

    # Accessories / bags
    'backpack':      'Bags',
    'handbag':       'Bags',
    'suitcase':      'Bags',

    # Nature / outdoors
    'potted plant':  'Plants',

    # Furniture
    'chair':         'Furniture',
    'couch':         'Furniture',
    'bed':           'Furniture',
    'dining table':  'Furniture',

    # Street
    'traffic light': 'Street Scenes',
    'stop sign':     'Street Scenes',
    'fire hydrant':  'Street Scenes',
    'parking meter': 'Street Scenes',
    'bench':         'Street Scenes',
}

SCENE_TO_CATEGORY = {
    'beach':        'Beach',
    'mountain':     'Mountains',
    'nature':       'Nature',
    'forest':       'Forests',
    'road/street':  'Streets',
    'kitchen':      'Kitchen',
    'dining':       'Dining',
    'office':       'Office',
    'sports':       'Sports',
    'party':        'Parties',
    'living room':  'Living Rooms',
    'bedroom':      'Bedrooms',
    'bathroom':     'Bathrooms',
}

WEATHER_TO_CATEGORY = {
    'rainy':  'Rainy Days',
    'snowy':  'Snowy Days',
    'sunny':  'Sunny Days',
    'cloudy': 'Cloudy Days',
}

CATEGORY_ICON = {
    'Bikes':         '🚲',
    'Cars':          '🚗',
    'Boats':         '⛵',
    'Airplanes':     '✈️',
    'Trains':        '🚂',
    'Dogs':          '🐕',
    'Cats':          '🐈',
    'Birds':         '🐦',
    'Horses':        '🐎',
    'Cows':          '🐄',
    'Animals':       '🐾',
    'Food':          '🍕',
    'Drinks':        '🥤',
    'Electronics':   '💻',
    'Sports':        '⚽',
    'Bags':          '🎒',
    'Plants':        '🌿',
    'Furniture':     '🛋️',
    'Street Scenes': '🚦',
    'Beach':         '🏖️',
    'Mountains':     '⛰️',
    'Nature':        '🌳',
    'Forests':       '🌲',
    'Streets':       '🛣️',
    'Kitchen':       '🍳',
    'Dining':        '🍽️',
    'Office':        '💼',
    'Parties':       '🎉',
    'Living Rooms':  '🛋️',
    'Bedrooms':      '🛏️',
    'Bathrooms':     '🚿',
    'Rainy Days':    '🌧️',
    'Snowy Days':    '❄️',
    'Sunny Days':    '☀️',
    'Cloudy Days':   '☁️',
}


class ObjectClusterService:
    """
    Assigns photos to object-category clusters.
    One photo can belong to multiple clusters.
    """

    def get_categories_for_photo(
        self,
        detected_objects: List[Dict],
        scene_label: Optional[str] = None,
        weather: Optional[str] = None,
        face_count: int = 0,
    ) -> List[str]:
        """Return list of category names this photo belongs to."""
        categories = set()

        for obj in detected_objects:
            label = obj.get('label', '').lower()
            category = LABEL_TO_CATEGORY.get(label)
            if category:
                categories.add(category)

        if scene_label:
            scene_cat = SCENE_TO_CATEGORY.get(scene_label.lower())
            if scene_cat:
                categories.add(scene_cat)

        if weather and weather != 'unknown':
            weather_cat = WEATHER_TO_CATEGORY.get(weather.lower())
            if weather_cat:
                categories.add(weather_cat)

        return list(categories)

    def assign_photo_to_clusters(self, session, photo_id: str, categories: List[str]):
        """
        Upsert ObjectCluster rows for each category and link this photo to them.
        Safe to call multiple times — won't create duplicates.
        """
        from models import ObjectCluster, PhotoObjectCluster, Photo

        for category in categories:
            cluster = session.query(ObjectCluster).filter_by(category=category).first()

            if not cluster:
                cluster = ObjectCluster(
                    category=category,
                    label=category,
                    icon=CATEGORY_ICON.get(category, '📷'),
                    photo_count=0,
                )
                session.add(cluster)
                session.flush()
                logger.info(f"✓ Created new object cluster: {category}")

            existing = session.query(PhotoObjectCluster).filter_by(
                photo_id=photo_id,
                cluster_id=cluster.cluster_id,
            ).first()

            if not existing:
                link = PhotoObjectCluster(
                    photo_id=photo_id,
                    cluster_id=cluster.cluster_id,
                )
                session.add(link)
                cluster.photo_count = (cluster.photo_count or 0) + 1

                if not cluster.thumbnail_photo_id:
                    cluster.thumbnail_photo_id = photo_id
                    logger.info(f"  Set thumbnail for '{category}' → {photo_id}")

        session.commit()

    def update_cluster_thumbnail(self, session, cluster_id: str):
        """Pick the best-quality photo in the cluster as its thumbnail."""
        from models import ObjectCluster, PhotoObjectCluster, Photo

        cluster = session.query(ObjectCluster).filter_by(cluster_id=cluster_id).first()
        if not cluster:
            return

        photos = (
            session.query(Photo)
            .join(PhotoObjectCluster, Photo.photo_id == PhotoObjectCluster.photo_id)
            .filter(PhotoObjectCluster.cluster_id == cluster_id)
            .order_by(Photo.quality_score.desc())
            .all()
        )

        if photos:
            cluster.thumbnail_photo_id = photos[0].photo_id
            session.commit()
            logger.info(f"✓ Updated thumbnail for '{cluster.category}' → {photos[0].photo_id}")

    def get_all_clusters(self, session, objects_only: bool = False) -> List[Dict]:
        """
        Return all object clusters with counts and thumbnail info.

        FIX: thumbnail_url now points to /uploads/<filename> (the correct
        static route) instead of /thumbnails/<path.name> (the face-thumbnail
        route that serves a different folder).
        """
        from models import ObjectCluster, Photo, PhotoObjectCluster

        clusters = (
            session.query(ObjectCluster)
            .filter(ObjectCluster.photo_count > 0)
            .order_by(ObjectCluster.photo_count.desc())
            .all()
        )

        result = []
        for c in clusters:
            # FIX: build correct thumbnail URL from the uploads folder
            thumbnail_url = None
            if c.thumbnail_photo_id:
                photo = session.query(Photo).filter_by(
                    photo_id=c.thumbnail_photo_id
                ).first()
                if photo:
                    thumbnail_url = f"/uploads/{photo.filename}"   # ← FIX

            if objects_only:
                no_people_count = (
                    session.query(PhotoObjectCluster)
                    .join(Photo, Photo.photo_id == PhotoObjectCluster.photo_id)
                    .filter(
                        PhotoObjectCluster.cluster_id == c.cluster_id,
                        Photo.face_count == 0,
                    )
                    .count()
                )
                if no_people_count == 0:
                    continue
                photo_count = no_people_count
            else:
                photo_count = c.photo_count

            result.append({
                'cluster_id':    c.cluster_id,
                'category':      c.category,
                'label':         c.label,
                'icon':          c.icon,
                'photo_count':   photo_count,
                'thumbnail_url': thumbnail_url,
            })

        return result

    def get_photos_in_cluster(
        self,
        session,
        cluster_id: str,
        objects_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Return photos belonging to a cluster.

        FIX: thumbnail_url uses /uploads/<filename> instead of /thumbnails/.
        """
        from models import Photo, PhotoObjectCluster

        query = (
            session.query(Photo)
            .join(PhotoObjectCluster, Photo.photo_id == PhotoObjectCluster.photo_id)
            .filter(PhotoObjectCluster.cluster_id == cluster_id)
        )

        if objects_only:
            query = query.filter(Photo.face_count == 0)

        photos = (
            query
            .order_by(Photo.date_taken.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

        return [
            {
                'photo_id':      p.photo_id,
                'filename':      p.filename,
                'thumbnail_url': f"/uploads/{p.filename}",   # ← FIX
                'date_taken':    p.date_taken.isoformat() if p.date_taken else None,
                'face_count':    p.face_count or 0,
                'scene_type':    p.scene_type,
                'weather':       p.weather,
            }
            for p in photos
        ]

    def rebuild_all_clusters(self, session):
        """Wipe and rebuild all object clusters from scratch."""
        from models import ObjectCluster, PhotoObjectCluster, Photo, DetectedObject

        logger.info("Rebuilding all object clusters from scratch...")

        session.query(PhotoObjectCluster).delete()
        session.query(ObjectCluster).delete()
        session.commit()

        photos = session.query(Photo).all()
        total = len(photos)

        for idx, photo in enumerate(photos):
            objects = session.query(DetectedObject).filter_by(photo_id=photo.photo_id).all()

            obj_list = [{'label': o.label, 'confidence': o.confidence} for o in objects]

            categories = self.get_categories_for_photo(
                detected_objects=obj_list,
                scene_label=getattr(photo, 'scene_type', None),
                weather=getattr(photo, 'weather', None),
                face_count=photo.face_count or 0,
            )

            if categories:
                self.assign_photo_to_clusters(session, photo.photo_id, categories)

            if (idx + 1) % 50 == 0:
                logger.info(f"  Rebuilt {idx + 1}/{total} photos")

        logger.info(f"✓ Rebuild complete — processed {total} photos")


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────
_object_cluster_service = None

def get_object_cluster_service() -> ObjectClusterService:
    global _object_cluster_service
    if _object_cluster_service is None:
        _object_cluster_service = ObjectClusterService()
    return _object_cluster_service