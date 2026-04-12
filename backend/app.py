from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import shutil
import json
import time
import uuid
import numpy as np
from datetime import datetime
import logging

# Import SQLAlchemy models 
from models import (
    Session, Photo, Cluster, FaceEmbedding, 
    DetectedObject, PhotoCluster, Conversation, Message,
    init_db  
)

try:
    from services.pipeline_service import get_pipeline
    from services.face_service import get_face_service
    from services.clip_service import get_clip_service
    from services.query_service import get_query_service
    from services.retrieval_service import get_retrieval_service
    from services.context_service import get_context_service
    from services.query_parser import get_query_parser
    from services.llm_service import get_llm_service
    from services.conversation_service import get_conversation_service
    from services.memory_service import get_memory_service
    from services.object_cluster_service import get_object_cluster_service

    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  WARNING: Services not available: {e}")
    print("    Make sure backend/services/ directory exists with all service modules")
    SERVICES_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ORGANIZED_FOLDER = 'organized_photos'
THUMBNAILS_FOLDER = 'thumbnails'

# Create folders
for folder in [UPLOAD_FOLDER, ORGANIZED_FOLDER, THUMBNAILS_FOLDER]:
    os.makedirs(folder, exist_ok=True)


try:
    init_db()
    logger.info("✓ Database tables verified/created")
except Exception as _db_init_err:
    logger.warning(f"DB init warning: {_db_init_err}")

logger.info("✓ Lumeo backend initialized")
logger.info(f"✓ Services available: {SERVICES_AVAILABLE}")

# ============================================================================
# STATIC FILE ROUTES
# ============================================================================

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    return send_from_directory(THUMBNAILS_FOLDER, filename)

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    session = Session()
    try:
        photo_count = session.query(Photo).count()
        session.close()
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'services': SERVICES_AVAILABLE,
            'photos': photo_count
        })
    except Exception as e:
        session.close()
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    if not SERVICES_AVAILABLE:
        return jsonify({'ready': False, 'error': 'Services not imported'}), 500
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_processing_stats()
        return jsonify({'ready': all(stats.values()), 'services': stats})
    except Exception as e:
        return jsonify({'ready': False, 'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_photos():
    if 'photos' not in request.files:
        return jsonify({'error': 'No photos uploaded'}), 400
    files = request.files.getlist('photos')
    if not files:
        return jsonify({'error': 'No photos selected'}), 400

    uploaded_photos = []
    session = Session()
    try:
        for file in files:
            if file.filename:
                timestamp = datetime.now().timestamp()
                filename = f"{timestamp}_{file.filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                photo_id = f"photo_{timestamp}_{len(uploaded_photos)}"
                photo = Photo(
                    photo_id=photo_id,
                    filename=filename,
                    path=filepath,
                    upload_date=time.time()
                )
                session.add(photo)
                uploaded_photos.append({'photo_id': photo_id, 'filename': filename, 'path': filepath})
                logger.info(f"✓ Uploaded: {filename}")
        session.commit()
        logger.info(f"✓ Uploaded {len(uploaded_photos)} photos")
        return jsonify({'success': True, 'photos_count': len(uploaded_photos), 'photos': uploaded_photos})
    except Exception as e:
        session.rollback()
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ── load centroid for every existing cluster from stored face bytes ──
def _load_existing_cluster_centroids(session):
    """
    Returns {cluster_id: mean_encoding (np.ndarray)} for all clusters
    that have at least one stored FaceEmbedding.
    """
    centroids = {}
    clusters = session.query(Cluster).all()
    for cl in clusters:
        embs = session.query(FaceEmbedding).filter_by(cluster_id=cl.cluster_id).all()
        if not embs:
            continue
        vecs = []
        for fe in embs:
            try:
                # Embeddings are stored as raw bytes (float64)
                v = np.frombuffer(fe.embedding, dtype=np.float64)
                if v.shape[0] == 128:          # face_recognition uses 128-d
                    vecs.append(v)
            except Exception:
                pass
        if vecs:
            centroids[cl.cluster_id] = np.mean(vecs, axis=0)
    return centroids


# ── find closest existing cluster for a new centroid ──
def _match_to_existing_cluster(new_centroid, existing_centroids, threshold=0.55):
    """
    Returns the cluster_id of the closest existing cluster if its distance
    is below *threshold*, otherwise returns None.

    threshold 0.55 is slightly tighter than the DBSCAN eps=0.6 so we only
    merge when we are reasonably confident.
    """
    best_id = None
    best_dist = float('inf')
    for cid, centroid in existing_centroids.items():
        dist = float(np.linalg.norm(new_centroid - centroid))
        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_id = cid
    return best_id


@app.route('/api/process', methods=['POST'])
def process_photos():
    """
    Process photos through vision pipeline and face clustering.

    KEY FIX (face clustering):
    Previously, DBSCAN label indices (0, 1, 2 …) were used directly as
    cluster IDs ("cluster_0", "cluster_1" …).  These indices are positional
    and change on every run, so the same person ends up with a different ID
    each time, breaking any name you assigned and mixing people across runs.

    The fix:
    1. Load the centroid (mean encoding) of every EXISTING cluster from the DB.
    2. After DBSCAN assigns new faces to temporary labels, compute each label's
       centroid and compare it against existing centroids.
    3. If the distance is small enough (< 0.55) → reuse the existing cluster ID
       (and therefore keep its name).
    4. If no existing cluster matches → create a brand-new cluster with a random
       UUID-based ID that is stable across runs.
    """
    if not SERVICES_AVAILABLE:
        return jsonify({
            'error': 'Vision services not available',
            'message': 'Ensure services/ directory exists with all modules'
        }), 500

    try:
        pipeline = get_pipeline()
        face_service = get_face_service()
        session = Session()

        stats = pipeline.get_processing_stats()
        if not all(stats.values()):
            return jsonify({'error': 'Some services not ready', 'service_status': stats}), 500

        photos = session.query(Photo).filter(Photo.clip_embedding == None).all()

        if not photos:
            session.close()
            return jsonify({'message': 'No unprocessed photos found', 'clusters': []}), 200

        logger.info(f"{'='*40}")
        logger.info(f"Processing {len(photos)} photos with vision pipeline")
        logger.info(f"{'='*40}")

        all_faces_data = []
        processed_count = 0

        # =================================================================
        # STEP 1: VISION PIPELINE
        # =================================================================
        for idx, photo in enumerate(photos):
            logger.info(f"\n--- Photo {idx + 1}/{len(photos)}: {photo.filename} ---")

            if not os.path.exists(photo.path):
                logger.warning(f"File not found: {photo.path}")
                continue

            result = pipeline.process_photo(photo.path, photo.photo_id)

            if not result.get('analysis_complete'):
                logger.error(f"Pipeline failed: {result.get('error')}")
                continue

            try:
                if result.get('clip_embedding'):
                    photo.clip_embedding = result['clip_embedding']

                scene = result.get('scene', {})
                photo.scene_type    = scene.get('scene_type', 'general')
                photo.activity      = scene.get('activity')
                photo.face_count    = result.get('face_count', 0)
                photo.weather       = result.get('weather', 'unknown')
 
                # Prefer the specific scene_label from detect_scene_and_weather
                # (e.g. "beach", "kitchen", "road/street") over the generic classify_scene
                # location (e.g. "beach", "dining") because SCENE_TO_CATEGORY uses the
                # detect_scene_and_weather labels. Fall back to classify_scene location.
                specific_scene = result.get('scene_label')  # from detect_scene_and_weather
                generic_location = scene.get('location')    # from classify_scene
                photo.location_type = specific_scene or generic_location
                photo.caption     = result.get('caption')

                photo_emotion = result.get('photo_emotion', {})
                photo.dominant_emotion = photo_emotion.get('dominant_emotion')
                photo.mood_score       = photo_emotion.get('mood_score')

                metadata = result.get('metadata', {})
                if metadata.get('date_taken'):
                    photo.date_taken = metadata['date_taken']
                photo.season      = metadata.get('season')
                photo.time_of_day = metadata.get('time_of_day')
                photo.camera_make  = metadata.get('camera_make')
                photo.camera_model = metadata.get('camera_model')
                photo.image_quality = metadata.get('quality_score')

                gps = metadata.get('gps')
                if gps:
                    photo.gps_latitude  = gps.get('latitude')
                    photo.gps_longitude = gps.get('longitude')

                for obj in result.get('objects', []):
                    detected_obj = DetectedObject(
                        photo_id=photo.photo_id,
                        label=obj['label'],
                        confidence=obj['confidence'],
                        bbox_x1=obj['bbox']['x1'],
                        bbox_y1=obj['bbox']['y1'],
                        bbox_x2=obj['bbox']['x2'],
                        bbox_y2=obj['bbox']['y2'],
                        dominant_color_rgb=str(obj.get('dominant_color_rgb', '')),
                        color_name=obj.get('color_name', '')
                    )
                    session.add(detected_obj)

                for face_data in result.get('faces', []):
                    all_faces_data.append({
                        'photo_id':     photo.photo_id,
                        'photo_path':   photo.path,
                        'encoding':     np.array(face_data['encoding']),
                        'location':     face_data['location'],
                        'quality_score': face_data.get('quality_score', 0.5),
                        'emotion':      face_data.get('emotion', {})
                    })

                processed_count += 1
                logger.info(
                    f"✓ Processed {photo.filename}: "
                    f"{len(result.get('faces', []))} faces, "
                    f"{len(result.get('objects', []))} objects"
                )

            except Exception as e:
                logger.error(f"Error saving data for {photo.filename}: {str(e)}")
                continue

        session.commit()
        logger.info(f"\n✓ Saved vision analysis for {processed_count} photos")

        # =================================================================
        # STEP 1b: ASSIGN OBJECT CLUSTERS
        # ─────────────────────────────────────────────────────────────────
        # Pass photo.location_type (e.g. "beach", "office", "dining")
        # instead of photo.scene_type ("indoor"/"outdoor").
        # SCENE_TO_CATEGORY maps specific location names, not indoor/outdoor.
        # =================================================================
        obj_cluster_svc = get_object_cluster_service()
        for photo in photos:
            db_objs = session.query(DetectedObject).filter_by(photo_id=photo.photo_id).all()
            result_objects = [{'label': o.label, 'confidence': o.confidence} for o in db_objs]

            categories = obj_cluster_svc.get_categories_for_photo(
                detected_objects=result_objects,
                scene_label=photo.location_type,   # specific location like "beach"/"kitchen"
                weather=photo.weather,
                face_count=photo.face_count or 0,
            )
            if categories:
                obj_cluster_svc.assign_photo_to_clusters(session, photo.photo_id, categories)
                logger.info(f"✓ Object clusters for {photo.filename}: {categories}")

        session.commit()

        # =================================================================
        # STEP 2: FACE CLUSTERING  
        # =================================================================
        if len(all_faces_data) == 0:
            session.close()
            logger.warning("No faces detected in any photos")
            return jsonify({
                'success': True,
                'processed_photos': processed_count,
                'total_faces': 0,
                'clusters': [],
                'message': 'Photos processed but no faces detected'
            })

        logger.info(f"\n{'='*40}")
        logger.info(f"Clustering {len(all_faces_data)} faces")
        logger.info(f"{'='*40}")

        encodings     = [face['encoding'] for face in all_faces_data]
        quality_scores = [face['quality_score'] for face in all_faces_data]

        labels = face_service.cluster_faces(encodings, quality_scores, min_samples=1, eps=0.6)

        # ── Step 2a: group new faces by DBSCAN label ──
        dbscan_groups = {}   # dbscan_label -> list of face dicts
        for idx, label in enumerate(labels):
            if label == -1:
                logger.debug(f"Outlier face at index {idx}")
                continue
            dbscan_groups.setdefault(label, []).append(all_faces_data[idx])

        logger.info(f"DBSCAN produced {len(dbscan_groups)} temporary groups")

        # ── Step 2b: load existing cluster centroids from DB ──────────
        existing_centroids = _load_existing_cluster_centroids(session)
        logger.info(f"Loaded centroids for {len(existing_centroids)} existing clusters")

        # ── Step 2c: map each DBSCAN group to a stable cluster ID ─────
        # We also track centroids we've already assigned in this run so two
        # different DBSCAN groups don't collapse into the same new cluster.
        label_to_cluster_id = {}   # dbscan_label -> stable cluster_id
        assigned_centroids  = dict(existing_centroids)  # copy; grows as we assign

        for label, faces in dbscan_groups.items():
            group_centroid = np.mean([f['encoding'] for f in faces], axis=0)
            matched_id = _match_to_existing_cluster(group_centroid, assigned_centroids)

            if matched_id:
                logger.info(
                    f"  DBSCAN group {label} → existing cluster '{matched_id}' "
                    f"(dist < 0.55)"
                )
                label_to_cluster_id[label] = matched_id
                # Update centroid to include new faces
                assigned_centroids[matched_id] = np.mean(
                    [assigned_centroids[matched_id], group_centroid], axis=0
                )
            else:
                new_id = f"cluster_{uuid.uuid4().hex[:8]}"
                logger.info(f"  DBSCAN group {label} → NEW cluster '{new_id}'")
                label_to_cluster_id[label] = new_id
                assigned_centroids[new_id] = group_centroid

        # ── Step 2d: build final clusters dict using stable IDs ───────
        clusters = {}
        for label, faces in dbscan_groups.items():
            cluster_id = label_to_cluster_id[label]
            if cluster_id not in clusters:
                clusters[cluster_id] = {'faces': [], 'photos': set()}
            clusters[cluster_id]['faces'].extend(faces)
            for f in faces:
                clusters[cluster_id]['photos'].add(f['photo_id'])

        logger.info(f"✓ Resolved to {len(clusters)} stable person clusters")

        # ── Step 2e: save clusters to DB ───
        for cluster_id, data in clusters.items():
            best_face = max(data['faces'], key=lambda x: x['quality_score'])
            logger.info(
                f"Cluster {cluster_id}: {len(data['faces'])} faces, "
                f"best quality: {best_face['quality_score']:.2f}"
            )

            thumbnail_filename = f"{cluster_id}_thumb.jpg"
            thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_filename)
            face_service.extract_face_thumbnail(
                best_face['photo_path'], best_face['location'], thumbnail_path
            )

            cluster_obj = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
            if not cluster_obj:
                # Brand-new cluster — assign a default name
                cluster_number = session.query(Cluster).count() + 1
                cluster_obj = Cluster(
                    cluster_id=cluster_id,
                    name=f"Person {cluster_number}",
                    face_count=len(data['faces']),
                    thumbnail=thumbnail_filename,
                    created_at=datetime.utcnow()
                )
                session.add(cluster_obj)
            else:
                # Existing cluster — only update counts/thumbnail, NOT the name
                cluster_obj.face_count = (
                    session.query(FaceEmbedding).filter_by(cluster_id=cluster_id).count()
                    + len(data['faces'])
                )
                cluster_obj.thumbnail  = thumbnail_filename
                cluster_obj.updated_at = datetime.utcnow()

            added_photo_clusters = set()

            for face_data in data['faces']:
                face_embedding = FaceEmbedding(
                    photo_id=face_data['photo_id'],
                    cluster_id=cluster_id,
                    embedding=face_data['encoding'].tobytes(),
                    face_location=json.dumps(face_data['location']),
                    emotion=face_data['emotion'].get('dominant_emotion'),
                    emotion_confidence=face_data['emotion'].get('confidence'),
                    emotion_valence=face_data['emotion'].get('valence'),
                    quality_score=face_data['quality_score']
                )
                session.add(face_embedding)

                pk = (face_data['photo_id'], cluster_id)
                if pk not in added_photo_clusters:
                    existing_link = session.query(PhotoCluster).filter_by(
                        photo_id=face_data['photo_id'],
                        cluster_id=cluster_id
                    ).first()
                    if not existing_link:
                        session.add(PhotoCluster(
                            photo_id=face_data['photo_id'],
                            cluster_id=cluster_id
                        ))
                    added_photo_clusters.add(pk)

        session.commit()
        logger.info("✓ Saved all clusters and face embeddings")

        # Build response
        clusters_list = session.query(Cluster).all()
        cluster_info  = []
        for cl in clusters_list:
            pcs = session.query(PhotoCluster).filter_by(cluster_id=cl.cluster_id).all()
            photos_in_cluster = []
            for pc in pcs:
                p = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
                if p:
                    photos_in_cluster.append({
                        'photo_id': p.photo_id,
                        'filename': p.filename,
                        'path': f'uploads/{p.filename}'
                    })
            cluster_info.append({
                'cluster_id': cl.cluster_id,
                'name': cl.name,
                'face_count': cl.face_count,
                'thumbnail': cl.thumbnail,
                'photos': photos_in_cluster
            })

        session.close()
        logger.info(
            f"\n{'='*40}\n"
            f"✓ Processing complete!\n"
            f"  - Processed photos : {processed_count}\n"
            f"  - Total faces      : {len(all_faces_data)}\n"
            f"  - Person clusters  : {len(clusters)}\n"
            f"{'='*40}\n"
        )

        return jsonify({
            'success': True,
            'processed_photos': processed_count,
            'total_faces': len(all_faces_data),
            'clusters': cluster_info,
            'total_clusters': len(clusters)
        })

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Original app.py
# (clusters, rename, merge, delete, organize, stats, reset, search,
#  chat, conversations, insights, memory, object-clusters …)
# ============================================================================

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    session = Session()
    try:
        clusters = session.query(Cluster).all()
        cluster_list = []
        for cluster in clusters:
            photo_clusters = session.query(PhotoCluster).filter_by(cluster_id=cluster.cluster_id).all()
            photos = []
            for pc in photo_clusters:
                photo = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
                if photo:
                    photos.append({'photo_id': photo.photo_id, 'filename': photo.filename, 'path': f'uploads/{photo.filename}'})
            cluster_list.append({'cluster_id': cluster.cluster_id, 'name': cluster.name, 'face_count': cluster.face_count, 'thumbnail': cluster.thumbnail, 'photos': photos})
        session.close()
        return jsonify({'clusters': cluster_list})
    except Exception as e:
        session.close()
        logger.error(f"Error getting clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster/<cluster_id>/photos', methods=['GET'])
def get_cluster_photos(cluster_id):
    session = Session()
    try:
        cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
        if not cluster:
            session.close()
            return jsonify({'error': 'Cluster not found'}), 404
        photo_clusters = session.query(PhotoCluster).filter_by(cluster_id=cluster_id).all()
        photos = []
        for pc in photo_clusters:
            photo = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
            if photo:
                photos.append({'photo_id': photo.photo_id, 'filename': photo.filename, 'path': f'uploads/{photo.filename}'})
        session.close()
        return jsonify({'cluster_id': cluster_id, 'name': cluster.name, 'face_count': cluster.face_count, 'photos': photos})
    except Exception as e:
        session.close()
        logger.error(f"Error getting cluster photos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster/rename', methods=['POST'])
def rename_cluster():
    data = request.json
    cluster_id = data.get('cluster_id')
    new_name   = data.get('name')
    if not cluster_id or not new_name:
        return jsonify({'error': 'Missing cluster_id or name'}), 400
    session = Session()
    try:
        cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
        if not cluster:
            session.close()
            return jsonify({'error': 'Cluster not found'}), 404
        cluster.name = new_name
        session.commit()
        session.close()
        logger.info(f"✓ Renamed cluster {cluster_id} to '{new_name}'")
        return jsonify({'success': True})
    except Exception as e:
        session.rollback()
        session.close()
        logger.error(f"Error renaming cluster: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters/merge', methods=['POST'])
def merge_clusters():
    data      = request.json
    source_id = data.get('source_cluster_id')
    target_id = data.get('target_cluster_id')
    if not source_id or not target_id:
        return jsonify({'error': 'Missing source_cluster_id or target_cluster_id'}), 400
    if source_id == target_id:
        return jsonify({'error': 'Source and target cannot be the same'}), 400
    session = Session()
    try:
        source = session.query(Cluster).filter_by(cluster_id=source_id).first()
        target = session.query(Cluster).filter_by(cluster_id=target_id).first()
        if not source or not target:
            session.close()
            return jsonify({'error': 'Cluster not found'}), 404
        session.query(FaceEmbedding).filter_by(cluster_id=source_id).update({'cluster_id': target_id})
        source_links = session.query(PhotoCluster).filter_by(cluster_id=source_id).all()
        for link in source_links:
            already = session.query(PhotoCluster).filter_by(photo_id=link.photo_id, cluster_id=target_id).first()
            if not already:
                session.add(PhotoCluster(photo_id=link.photo_id, cluster_id=target_id))
            session.delete(link)
        target.face_count  = session.query(FaceEmbedding).filter_by(cluster_id=target_id).count()
        target.photo_count = session.query(PhotoCluster).filter_by(cluster_id=target_id).count()
        target.updated_at  = datetime.utcnow()
        session.delete(source)
        session.commit()
        logger.info(f"✓ Merged cluster {source_id} into {target_id}")
        return jsonify({'success': True, 'merged_into': target_id})
    except Exception as e:
        session.rollback()
        logger.error(f"Merge error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/clusters/<cluster_id>', methods=['DELETE'])
def delete_cluster(cluster_id):
    session = Session()
    try:
        cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
        if not cluster:
            session.close()
            return jsonify({'error': 'Cluster not found'}), 404
        if cluster.thumbnail:
            thumb_path = os.path.join(THUMBNAILS_FOLDER, cluster.thumbnail)
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
        session.delete(cluster)
        session.commit()
        logger.info(f"✓ Deleted cluster {cluster_id}")
        return jsonify({'success': True})
    except Exception as e:
        session.rollback()
        logger.error(f"Delete cluster error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/clusters/<cluster_id>/photos/<photo_id>', methods=['DELETE'])
def remove_photo_from_cluster(cluster_id, photo_id):
    session = Session()
    try:
        link = session.query(PhotoCluster).filter_by(cluster_id=cluster_id, photo_id=photo_id).first()
        if not link:
            session.close()
            return jsonify({'error': 'Photo not in this cluster'}), 404
        session.delete(link)
        session.query(FaceEmbedding).filter_by(cluster_id=cluster_id, photo_id=photo_id).delete()
        cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
        if cluster:
            cluster.face_count  = session.query(FaceEmbedding).filter_by(cluster_id=cluster_id).count()
            cluster.photo_count = session.query(PhotoCluster).filter_by(cluster_id=cluster_id).count()
            cluster.updated_at  = datetime.utcnow()
        session.commit()
        logger.info(f"✓ Removed photo {photo_id} from cluster {cluster_id}")
        return jsonify({'success': True})
    except Exception as e:
        session.rollback()
        logger.error(f"Remove photo error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/organize', methods=['POST'])
def organize_photos():
    session = Session()
    try:
        clusters = session.query(Cluster).all()
        organized_count = 0
        for cluster in clusters:
            person_folder = os.path.join(ORGANIZED_FOLDER, cluster.name)
            os.makedirs(person_folder, exist_ok=True)
            photo_clusters = session.query(PhotoCluster).filter_by(cluster_id=cluster.cluster_id).all()
            for pc in photo_clusters:
                photo = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
                if photo and os.path.exists(photo.path):
                    dest_path = os.path.join(person_folder, photo.filename)
                    if not os.path.exists(dest_path):
                        shutil.copy2(photo.path, dest_path)
                        organized_count += 1
        session.close()
        logger.info(f"✓ Organized {organized_count} photos into folders")
        return jsonify({'success': True, 'organized_count': organized_count, 'output_folder': ORGANIZED_FOLDER})
    except Exception as e:
        session.close()
        logger.error(f"Error organizing photos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    session = Session()
    try:
        stats = {
            'total_photos':          session.query(Photo).count(),
            'total_clusters':        session.query(Cluster).count(),
            'processed_faces':       session.query(FaceEmbedding).count(),
            'detected_objects':      session.query(DetectedObject).count(),
            'photos_with_emotions':  session.query(Photo).filter(Photo.dominant_emotion != None).count(),
            'photos_with_scenes':    session.query(Photo).filter(Photo.scene_type != None).count()
        }
        session.close()
        return jsonify(stats)
    except Exception as e:
        session.close()
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_database():
    session = Session()
    try:
        session.query(PhotoCluster).delete()
        session.query(FaceEmbedding).delete()
        session.query(DetectedObject).delete()
        session.query(Cluster).delete()
        session.query(Photo).delete()
        session.commit()
        session.close()
        for folder in [UPLOAD_FOLDER, THUMBNAILS_FOLDER, ORGANIZED_FOLDER]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
        logger.info("✓ Database reset complete")
        return jsonify({'success': True, 'message': 'All data reset'})
    except Exception as e:
        session.rollback()
        session.close()
        logger.error(f"Error resetting database: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_photos():
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data        = request.json
        query       = data.get('query', '')
        top_k       = data.get('top_k', 10)
        use_filters = data.get('use_filters', True)
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        logger.info(f"=== SEARCH REQUEST ===\nQuery: {query}")
        query_service     = get_query_service()
        retrieval_service = get_retrieval_service()
        session           = Session()
        clusters          = session.query(Cluster).all()
        known_people      = [c.name for c in clusters]
        session.close()
        parser          = get_query_parser(known_people)
        parsed_filters  = parser.parse(query)
        query_embedding = query_service.encode_query(query)
        if query_embedding is None:
            return jsonify({'error': 'Failed to encode query'}), 500
        if use_filters and len(parsed_filters) > 1:
            db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
            results = retrieval_service.hybrid_search(query_embedding, filters=db_filters, top_k=top_k)
        else:
            results = retrieval_service.semantic_search(query_embedding, top_k=top_k, min_similarity=0.3)
        logger.info(f"✓ Retrieved {len(results)} photos")
        return jsonify({'success': True, 'query': query, 'parsed_filters': parsed_filters, 'results_count': len(results), 'results': results})
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/context', methods=['POST'])
def get_search_context():
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data                 = request.json
        query                = data.get('query', '')
        top_k                = data.get('top_k', 10)
        include_system_prompt = data.get('include_system_prompt', True)
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        query_service     = get_query_service()
        retrieval_service = get_retrieval_service()
        context_service   = get_context_service()
        session           = Session()
        clusters          = session.query(Cluster).all()
        known_people      = [c.name for c in clusters]
        session.close()
        parser          = get_query_parser(known_people)
        parsed_filters  = parser.parse(query)
        query_embedding = query_service.encode_query(query)
        if query_embedding is None:
            return jsonify({'error': 'Failed to encode query'}), 500
        db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
        results    = retrieval_service.hybrid_search(query_embedding, filters=db_filters, top_k=top_k)
        context    = context_service.build_context(results, query, include_system_prompt=include_system_prompt)
        return jsonify({'success': True, 'query': query, 'results_count': len(results), 'context': context, 'estimated_tokens': context_service.estimate_tokens(context)})
    except Exception as e:
        logger.error(f"Context generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/similar/<photo_id>', methods=['GET'])
def find_similar_photos(photo_id):
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        top_k             = request.args.get('top_k', 10, type=int)
        retrieval_service = get_retrieval_service()
        results           = retrieval_service.search_by_similar_photo(photo_id, top_k=top_k, exclude_self=True)
        return jsonify({'success': True, 'reference_photo_id': photo_id, 'results_count': len(results), 'results': results})
    except Exception as e:
        logger.error(f"Similar photo search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/summary', methods=['POST'])
def generate_summary():
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data         = request.json
        summary_type = data.get('summary_type', 'general')
        session      = Session()
        photos       = session.query(Photo).filter(Photo.clip_embedding.isnot(None)).limit(100).all()
        photo_dicts  = []
        for photo in photos:
            pcs    = session.query(PhotoCluster).filter_by(photo_id=photo.photo_id).all()
            people = []
            for pc in pcs:
                cl = session.query(Cluster).filter_by(cluster_id=pc.cluster_id).first()
                if cl:
                    people.append(cl.name)
            photo_dicts.append({'photo_id': photo.photo_id, 'people': people, 'dominant_emotion': photo.dominant_emotion, 'location': photo.location_type, 'activity': photo.activity, 'season': photo.season, 'time_of_day': photo.time_of_day})
        session.close()
        context_service = get_context_service()
        summary         = context_service.build_summary_context(photo_dicts, summary_type=summary_type)
        return jsonify({'success': True, 'summary_type': summary_type, 'photos_analyzed': len(photo_dicts), 'summary': summary})
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrieval/stats', methods=['GET'])
def retrieval_stats():
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        retrieval_service = get_retrieval_service()
        query_service     = get_query_service()
        return jsonify({'success': True, 'retrieval_stats': retrieval_service.get_retrieval_stats(), 'cache_stats': query_service.get_cache_stats()})
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query/parse', methods=['POST'])
def parse_query():
    try:
        data  = request.json
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        session      = Session()
        clusters     = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        session.close()
        parser         = get_query_parser(known_people)
        filters        = parser.parse(query)
        semantic_query = parser.get_semantic_query(filters)
        return jsonify({'success': True, 'original_query': query, 'parsed_filters': filters, 'semantic_query': semantic_query, 'known_people': known_people})
    except Exception as e:
        logger.error(f"Query parsing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# CHAT / LLM ROUTES
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat(custom_data=None):
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data              = custom_data if custom_data else request.json
        message           = data.get('message', '')
        conversation_id   = data.get('conversation_id')
        top_k             = data.get('top_k', 5)
        use_streaming     = data.get('stream', False)
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        logger.info(f"=== CHAT REQUEST ===\nMessage: {message}\nConversation: {conversation_id}")
        query_service        = get_query_service()
        retrieval_service    = get_retrieval_service()
        context_service      = get_context_service()
        llm_service          = get_llm_service()
        conversation_service = get_conversation_service()
        session              = Session()
        if not conversation_id:
            conversation_id = conversation_service.create_conversation(session)
            logger.info(f"Created new conversation: {conversation_id}")
        elif conversation_id == 'default':
            from models import Conversation
            existing = session.query(Conversation).filter_by(conversation_id='default').first()
            if not existing:
                default_conversation = Conversation(conversation_id='default', created_at=time.time(), updated_at=time.time())
                session.add(default_conversation)
                session.commit()
        clusters     = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        parser         = get_query_parser(known_people)
        parsed_filters = parser.parse(message)
        logger.info(f"Parsed filters: {parsed_filters}")
        query_embedding = query_service.encode_query(message)
        if query_embedding is None:
            session.close()
            return jsonify({'error': 'Failed to encode query'}), 500
        db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
        if db_filters:
            retrieved_photos = retrieval_service.hybrid_search(query_embedding, filters=db_filters, top_k=top_k)
        else:
            retrieved_photos = retrieval_service.semantic_search(query_embedding, top_k=top_k, min_similarity=0.25)
        logger.info(f"✓ Retrieved {len(retrieved_photos)} photos")
        photo_context = context_service.build_context(retrieved_photos, message, include_system_prompt=False)
        full_context  = conversation_service.build_context_with_history(session, conversation_id, message, photo_context)
        photo_ids     = [p['photo_id'] for p in retrieved_photos]
        conversation_service.add_message(session, conversation_id, role='user', content=message, retrieved_photo_ids=photo_ids, metadata={'filters': parsed_filters, 'results_count': len(retrieved_photos)})
        session.commit()
        if use_streaming:
            session.close()
            return Response(stream_chat_response(llm_service, conversation_service, conversation_id, full_context, message, retrieved_photos), mimetype='text/event-stream')
        else:
            response_text = llm_service.generate_response(context=full_context, query=message)
            validation    = llm_service.validate_response(response_text, full_context)
            conversation_service.add_message(session, conversation_id, role='assistant', content=response_text, metadata={'validation': validation})
            session.commit()
            session.close()
            logger.info(f"✓ Generated response ({len(response_text)} chars)")
            return jsonify({'success': True, 'conversation_id': conversation_id, 'message': message, 'response': response_text, 'retrieved_photos': retrieved_photos, 'validation': validation})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def stream_chat_response(llm_service, conversation_service, conversation_id, context, query, retrieved_photos):
    full_response = []
    try:
        yield f"data: {json.dumps({'type': 'photos', 'photos': retrieved_photos})}\n\n"
        for chunk in llm_service.generate_streaming_response(context, query):
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        response_text = ''.join(full_response)
        session = Session()
        conversation_service.add_message(session, conversation_id, role='assistant', content=response_text)
        session.commit()
        session.close()
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json() or {}
    data['stream'] = True
    return chat(custom_data=data)

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    try:
        limit                = request.args.get('limit', 20, type=int)
        session              = Session()
        conversation_service = get_conversation_service()
        conversations        = conversation_service.get_all_conversations(session, limit=limit)
        session.close()
        return jsonify({'success': True, 'conversations': conversations})
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    try:
        session              = Session()
        conversation_service = get_conversation_service()
        history = conversation_service.get_conversation_history(session, conversation_id)
        stats   = conversation_service.get_conversation_stats(session, conversation_id)
        session.close()
        return jsonify({'success': True, 'conversation_id': conversation_id, 'messages': history, 'stats': stats})
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/new', methods=['POST'])
def new_conversation():
    try:
        session              = Session()
        conversation_service = get_conversation_service()
        conversation_id      = conversation_service.create_conversation(session)
        session.close()
        return jsonify({'success': True, 'conversation_id': conversation_id})
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    try:
        session              = Session()
        conversation_service = get_conversation_service()
        success              = conversation_service.delete_conversation(session, conversation_id)
        session.close()
        if success:
            return jsonify({'success': True, 'message': f'Conversation {conversation_id} deleted'})
        return jsonify({'error': 'Failed to delete conversation'}), 500
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights', methods=['POST'])
def generate_insights():
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data         = request.json
        insight_type = data.get('insight_type', 'general')
        session      = Session()
        photos       = session.query(Photo).filter(Photo.clip_embedding.isnot(None)).limit(100).all()
        photo_dicts  = []
        for photo in photos:
            pcs    = session.query(PhotoCluster).filter_by(photo_id=photo.photo_id).all()
            people = []
            for pc in pcs:
                cl = session.query(Cluster).filter_by(cluster_id=pc.cluster_id).first()
                if cl:
                    people.append(cl.name)
            photo_dicts.append({'photo_id': photo.photo_id, 'people': people, 'dominant_emotion': photo.dominant_emotion, 'location': photo.location_type, 'activity': photo.activity, 'season': photo.season, 'time_of_day': photo.time_of_day})
        context_service = get_context_service()
        summary         = context_service.build_summary_context(photo_dicts, summary_type=insight_type)
        llm_service     = get_llm_service()
        insights        = llm_service.generate_insight(summary_context=summary, insight_type=insight_type)
        session.close()
        return jsonify({'success': True, 'insight_type': insight_type, 'photos_analyzed': len(photo_dicts), 'summary': summary, 'insights': insights})
    except Exception as e:
        logger.error(f"Insights generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/llm/status', methods=['GET'])
def llm_status():
    try:
        llm_service = get_llm_service()
        model_info  = llm_service.get_model_info()
        return jsonify({'success': True, 'model': llm_service.model, 'base_url': llm_service.base_url, 'model_info': model_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/enhanced', methods=['POST'])
def enhanced_chat():
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data             = request.json
        message          = data.get('message', '')
        conversation_id  = data.get('conversation_id')
        use_memory       = data.get('use_memory', True)
        top_k            = data.get('top_k', 5)
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        query_service        = get_query_service()
        retrieval_service    = get_retrieval_service()
        context_service      = get_context_service()
        llm_service          = get_llm_service()
        conversation_service = get_conversation_service()
        memory_service       = get_memory_service()
        session              = Session()
        if not conversation_id:
            conversation_id = conversation_service.create_conversation(session)
        clusters     = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        parser         = get_query_parser(known_people)
        parsed_filters = parser.parse(message)
        query_embedding = query_service.encode_query(message)
        if query_embedding is None:
            session.close()
            return jsonify({'error': 'Failed to encode query'}), 500
        db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
        if db_filters:
            retrieved_photos = retrieval_service.hybrid_search(query_embedding, filters=db_filters, top_k=top_k)
        else:
            retrieved_photos = retrieval_service.semantic_search(query_embedding, top_k=top_k, min_similarity=0.25)
        photo_context      = context_service.build_context(retrieved_photos, message, include_system_prompt=False)
        memory_context     = ""
        relevant_memories  = []
        if use_memory:
            relevant_memories = memory_service.get_relevant_memories(session, message, user_id="default_user", limit=3)
            if relevant_memories:
                memory_context = memory_service.build_memory_context(relevant_memories)
        full_context = conversation_service.build_context_with_history(session, conversation_id, message, photo_context, use_summary=True)
        if memory_context:
            full_context = memory_context + "\n" + full_context
        photo_ids = [p['photo_id'] for p in retrieved_photos]
        conversation_service.add_message(session, conversation_id, role='user', content=message, retrieved_photo_ids=photo_ids, metadata={'filters': parsed_filters, 'results_count': len(retrieved_photos), 'memories_used': len(relevant_memories)})
        response_text = llm_service.generate_response(context=full_context, query=message)
        validation    = llm_service.validate_response(response_text, full_context)
        conversation_service.add_message(session, conversation_id, role='assistant', content=response_text, metadata={'validation': validation})
        summary = conversation_service.auto_summarize_if_needed(session, conversation_id, llm_service)
        session.commit()
        session.close()
        return jsonify({'success': True, 'conversation_id': conversation_id, 'response': response_text, 'retrieved_photos': retrieved_photos, 'relevant_memories': relevant_memories, 'validation': validation, 'summarized': summary is not None})
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>/summarize', methods=['POST'])
def summarize_conversation(conversation_id):
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data                 = request.json or {}
        recursive            = data.get('recursive', True)
        session              = Session()
        conversation_service = get_conversation_service()
        llm_service          = get_llm_service()
        summary              = conversation_service.summarize_conversation(session, conversation_id, llm_service, recursive=recursive)
        session.close()
        return jsonify({'success': True, 'conversation_id': conversation_id, 'summary': summary})
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>/optimize', methods=['POST'])
def optimize_conversation(conversation_id):
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    try:
        data            = request.json or {}
        target_messages = data.get('target_messages', 20)
        session         = Session()
        memory_service  = get_memory_service()
        llm_service     = get_llm_service()
        result          = memory_service.optimize_long_conversation(session, conversation_id, llm_service, target_messages=target_messages)
        session.close()
        return jsonify({'success': True, **result})
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/topics', methods=['GET'])
def get_topics():
    try:
        limit          = request.args.get('limit', 10, type=int)
        session        = Session()
        memory_service = get_memory_service()
        topics         = memory_service.get_frequently_discussed_topics(session, limit=limit)
        session.close()
        return jsonify({'success': True, 'topics': topics})
    except Exception as e:
        logger.error(f"Topics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/timeline', methods=['GET'])
def get_memory_timeline():
    try:
        days           = request.args.get('days', 30, type=int)
        session        = Session()
        memory_service = get_memory_service()
        timeline       = memory_service.get_conversation_timeline(session, days=days)
        session.close()
        return jsonify({'success': True, 'timeline': timeline, 'days': days})
    except Exception as e:
        logger.error(f"Timeline error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/photo/<photo_id>/history', methods=['GET'])
def get_photo_history(photo_id):
    try:
        session        = Session()
        memory_service = get_memory_service()
        history        = memory_service.get_photo_interaction_history(session, photo_id)
        session.close()
        return jsonify({'success': True, 'photo_id': photo_id, 'interactions': history, 'interaction_count': len(history)})
    except Exception as e:
        logger.error(f"Photo history error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/search', methods=['POST'])
def search_conversations():
    try:
        data                 = request.json
        query                = data.get('query', '')
        limit                = data.get('limit', 10)
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        session              = Session()
        conversation_service = get_conversation_service()
        results              = conversation_service.search_conversations(session, query, limit=limit)
        session.close()
        return jsonify({'success': True, 'query': query, 'results': results, 'count': len(results)})
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<conversation_id>/export', methods=['GET'])
def export_conversation(conversation_id):
    try:
        format_type          = request.args.get('format', 'json')
        session              = Session()
        conversation_service = get_conversation_service()
        from models import Conversation
        conversation = session.query(Conversation).filter_by(conversation_id=conversation_id).first()
        if not conversation:
            session.close()
            return jsonify({'error': 'Conversation not found'}), 404
        history = conversation_service.get_conversation_history(session, conversation_id)
        session.close()
        if format_type == 'markdown':
            md_lines = [f"# Conversation {conversation_id}", f"Created: {datetime.fromtimestamp(conversation.created_at).strftime('%Y-%m-%d %H:%M')}", f"Messages: {len(history)}", ""]
            if conversation.summary:
                md_lines.extend(["## Summary", conversation.summary, ""])
            md_lines.append("## Messages\n")
            for msg in history:
                role      = msg['role'].upper()
                content   = msg['content']
                timestamp = datetime.fromtimestamp(msg['created_at']).strftime('%Y-%m-%d %H:%M')
                md_lines.append(f"### {role} ({timestamp})")
                md_lines.append(content)
                md_lines.append("")
            return Response('\n'.join(md_lines), mimetype='text/markdown', headers={'Content-Disposition': f'attachment; filename=conversation_{conversation_id}.md'})
        else:
            return jsonify({'conversation_id': conversation_id, 'created_at': conversation.created_at, 'updated_at': conversation.updated_at, 'message_count': len(history), 'summary': conversation.summary, 'messages': history})
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# OBJECT CLUSTER ROUTES
# ============================================================================

@app.route('/api/object-clusters', methods=['GET'])
def get_object_clusters():
    objects_only = request.args.get('objects_only', 'false').lower() == 'true'
    session = Session()
    try:
        service  = get_object_cluster_service()
        clusters = service.get_all_clusters(session, objects_only=objects_only)
        return jsonify({'clusters': clusters, 'total': len(clusters)})
    except Exception as e:
        logger.error(f"Error fetching object clusters: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/object-clusters/<cluster_id>/photos', methods=['GET'])
def get_object_cluster_photos(cluster_id):
    objects_only = request.args.get('objects_only', 'false').lower() == 'true'
    limit  = int(request.args.get('limit',  50))
    offset = int(request.args.get('offset',  0))
    session = Session()
    try:
        service = get_object_cluster_service()
        photos  = service.get_photos_in_cluster(session, cluster_id, objects_only=objects_only, limit=limit, offset=offset)
        return jsonify({'photos': photos, 'cluster_id': cluster_id})
    except Exception as e:
        logger.error(f"Error fetching cluster photos: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/object-clusters/rebuild', methods=['POST'])
def rebuild_object_clusters():
    session = Session()
    try:
        service  = get_object_cluster_service()
        service.rebuild_all_clusters(session)
        clusters = service.get_all_clusters(session)
        return jsonify({'message': 'Rebuild complete', 'cluster_count': len(clusters)})
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting Lumeo Photo Organizer Backend")
    logger.info("="*60)
    logger.info(f"   Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"   Services: {'✓ Available' if SERVICES_AVAILABLE else '✗ Not Available'}")
    logger.info("="*60 + "\n")
    app.run(debug=True, port=5002, host='0.0.0.0')