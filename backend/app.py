"""
Lumeo Photo Organizer Backend - Complete Phase 2
Uses PostgreSQL + SQLAlchemy + Vision Intelligence Pipeline

Features:
- Face recognition with quality assessment
- Emotion detection
- Object detection with YOLO
- Scene classification
- CLIP embeddings
- Metadata extraction
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import json
import time
import numpy as np
from datetime import datetime
import logging

# Import SQLAlchemy models (Phase 1)
from models import (
    Session, Photo, Cluster, FaceEmbedding, 
    DetectedObject, PhotoCluster
)

# Import vision services (Phase 2)
try:
    from services.pipeline_service import get_pipeline
    from services.face_service import get_face_service
    from services.clip_service import get_clip_service
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ORGANIZED_FOLDER = 'organized_photos'
THUMBNAILS_FOLDER = 'thumbnails'

# Create folders
for folder in [UPLOAD_FOLDER, ORGANIZED_FOLDER, THUMBNAILS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

logger.info("✓ Lumeo backend initialized")
logger.info(f"✓ Services available: {SERVICES_AVAILABLE}")

# ============================================================================
# STATIC FILE ROUTES
# ============================================================================

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded photos"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve face thumbnails"""
    return send_from_directory(THUMBNAILS_FOLDER, filename)

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
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
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    """Check if AI services are ready"""
    if not SERVICES_AVAILABLE:
        return jsonify({
            'ready': False,
            'error': 'Services not imported'
        }), 500
    
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_processing_stats()
        
        return jsonify({
            'ready': all(stats.values()),
            'services': stats
        })
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_photos():
    """
    Upload photos to the system
    
    Request: multipart/form-data with 'photos' field
    Response: List of uploaded photo info
    """
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
                # Generate unique filename
                timestamp = datetime.now().timestamp()
                filename = f"{timestamp}_{file.filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # Save file
                file.save(filepath)
                
                # Create database record
                photo_id = f"photo_{timestamp}_{len(uploaded_photos)}"
                photo = Photo(
                    photo_id=photo_id,
                    filename=filename,
                    path=filepath,
                    upload_date=time.time()
                )
                session.add(photo)
                
                uploaded_photos.append({
                    'photo_id': photo_id,
                    'filename': filename,
                    'path': filepath
                })
                
                logger.info(f"✓ Uploaded: {filename}")
        
        session.commit()
        logger.info(f"✓ Uploaded {len(uploaded_photos)} photos")
        
        return jsonify({
            'success': True,
            'photos_count': len(uploaded_photos),
            'photos': uploaded_photos
        })
        
    except Exception as e:
        session.rollback()
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/process', methods=['POST'])
def process_photos():
    """
    Process photos through vision pipeline and face clustering
    
    Phase 2 Enhanced:
    - Face detection with quality scores
    - Emotion detection per face
    - Object detection with YOLO
    - Scene classification
    - CLIP embeddings
    - Metadata extraction
    - Face clustering with emotions
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
        
        # Check if services are ready
        stats = pipeline.get_processing_stats()
        if not all(stats.values()):
            return jsonify({
                'error': 'Some services not ready',
                'service_status': stats
            }), 500
        
        # Get unprocessed photos (those without CLIP embeddings)
        photos = session.query(Photo).filter(Photo.clip_embedding == None).all()
        
        if not photos:
            session.close()
            return jsonify({
                'message': 'No unprocessed photos found',
                'clusters': []
            }), 200
        
        logger.info(f"========================================")
        logger.info(f"Processing {len(photos)} photos with vision pipeline")
        logger.info(f"========================================")
        
        all_faces_data = []  # For clustering
        processed_count = 0
        
        # =====================================================================
        # STEP 1: VISION PIPELINE - Analyze each photo
        # =====================================================================
        
        for idx, photo in enumerate(photos):
            logger.info(f"\n--- Photo {idx + 1}/{len(photos)}: {photo.filename} ---")
            
            if not os.path.exists(photo.path):
                logger.warning(f"File not found: {photo.path}")
                continue
            
            # Run full vision pipeline
            result = pipeline.process_photo(photo.path, photo.photo_id)
            
            if not result.get('analysis_complete'):
                logger.error(f"Pipeline failed: {result.get('error')}")
                continue
            
            try:
                # Update photo with vision analysis results
                
                # CLIP embedding (for semantic search in Phase 3)
                if result.get('clip_embedding'):
                    # Convert list back to numpy array for pgvector
                    photo.clip_embedding = result['clip_embedding']
                
                # Scene classification
                scene = result.get('scene', {})
                photo.scene_type = scene.get('scene_type')
                photo.location_type = scene.get('location')
                photo.activity = scene.get('activity')
                
                # Caption
                photo.caption = result.get('caption')
                
                # Emotion aggregation
                photo_emotion = result.get('photo_emotion', {})
                photo.dominant_emotion = photo_emotion.get('dominant_emotion')
                photo.mood_score = photo_emotion.get('mood_score')
                
                # Metadata from EXIF
                metadata = result.get('metadata', {})
                if metadata.get('date_taken'):
                    photo.date_taken = metadata['date_taken']
                photo.season = metadata.get('season')
                photo.time_of_day = metadata.get('time_of_day')
                photo.camera_make = metadata.get('camera_make')
                photo.camera_model = metadata.get('camera_model')
                photo.image_quality = metadata.get('quality_score')
                
                # GPS coordinates
                gps = metadata.get('gps')
                if gps:
                    photo.gps_latitude = gps.get('latitude')
                    photo.gps_longitude = gps.get('longitude')
                
                # Save detected objects
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
                
                # Collect face data for clustering
                for face_data in result.get('faces', []):
                    all_faces_data.append({
                        'photo_id': photo.photo_id,
                        'photo_path': photo.path,
                        'encoding': np.array(face_data['encoding']),
                        'location': face_data['location'],
                        'quality_score': face_data.get('quality_score', 0.5),
                        'emotion': face_data.get('emotion', {})
                    })
                
                processed_count += 1
                logger.info(f"✓ Processed {photo.filename}: {len(result.get('faces', []))} faces, {len(result.get('objects', []))} objects")
                
            except Exception as e:
                logger.error(f"Error saving data for {photo.filename}: {str(e)}")
                continue
        
        # Commit photo updates and objects
        session.commit()
        logger.info(f"\n✓ Saved vision analysis for {processed_count} photos")
        
        # =====================================================================
        # STEP 2: FACE CLUSTERING
        # =====================================================================
        
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
        
        logger.info(f"\n========================================")
        logger.info(f"Clustering {len(all_faces_data)} faces")
        logger.info(f"========================================")
        
        # Extract encodings and quality scores
        encodings = [face['encoding'] for face in all_faces_data]
        quality_scores = [face['quality_score'] for face in all_faces_data]
        
        # Cluster faces
        labels = face_service.cluster_faces(encodings, quality_scores, min_samples=1, eps=0.6)
        
        # Organize faces by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Skip noise/outliers
                logger.debug(f"Outlier face at index {idx}")
                continue
            
            cluster_id = f"cluster_{label}"
            
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'faces': [],
                    'photos': set()
                }
            
            clusters[cluster_id]['faces'].append(all_faces_data[idx])
            clusters[cluster_id]['photos'].add(all_faces_data[idx]['photo_id'])
        
        logger.info(f"✓ Created {len(clusters)} person clusters")
        
        # Save clusters to database
        for cluster_id, data in clusters.items():
            # Find best quality face for thumbnail
            best_face = max(data['faces'], key=lambda x: x['quality_score'])
            
            logger.info(f"Cluster {cluster_id}: {len(data['faces'])} faces, best quality: {best_face['quality_score']:.2f}")
            
            # Create thumbnail from best face
            thumbnail_filename = f"{cluster_id}_thumb.jpg"
            thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_filename)
            
            face_service.extract_face_thumbnail(
                best_face['photo_path'],
                best_face['location'],
                thumbnail_path
            )
            
            # Create or update cluster
            cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
            if not cluster:
                cluster = Cluster(
                    cluster_id=cluster_id,
                    name=f"Person {cluster_id.split('_')[1]}",
                    face_count=len(data['faces']),
                    thumbnail=thumbnail_filename,
                    created_at=time.time()
                )
                session.add(cluster)
            else:
                cluster.face_count = len(data['faces'])
                cluster.thumbnail = thumbnail_filename
            
            # Save face embeddings with emotion and quality
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
                
                # Link photo to cluster (many-to-many)
                photo_cluster = PhotoCluster(
                    photo_id=face_data['photo_id'],
                    cluster_id=cluster_id
                )
                session.add(photo_cluster)
        
        session.commit()
        logger.info(f"✓ Saved all clusters and face embeddings")
        
        # Get cluster info for response
        clusters_list = session.query(Cluster).all()
        cluster_info = []
        
        for cluster in clusters_list:
            # Get photos for this cluster
            photo_clusters = session.query(PhotoCluster).filter_by(
                cluster_id=cluster.cluster_id
            ).all()
            
            photos_in_cluster = []
            for pc in photo_clusters:
                p = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
                if p:
                    photos_in_cluster.append({
                        'photo_id': p.photo_id,
                        'filename': p.filename,
                        'path': p.filename  # Frontend expects filename
                    })
            
            cluster_info.append({
                'cluster_id': cluster.cluster_id,
                'name': cluster.name,
                'face_count': cluster.face_count,
                'thumbnail': cluster.thumbnail,
                'photos': photos_in_cluster
            })
        
        session.close()
        
        logger.info(f"\n========================================")
        logger.info(f"✓ Processing complete!")
        logger.info(f"  - Processed photos: {processed_count}")
        logger.info(f"  - Total faces: {len(all_faces_data)}")
        logger.info(f"  - Person clusters: {len(clusters)}")
        logger.info(f"========================================\n")
        
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

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get all person clusters with their photos"""
    session = Session()
    
    try:
        clusters = session.query(Cluster).all()
        cluster_list = []
        
        for cluster in clusters:
            # Get photos for this cluster via junction table
            photo_clusters = session.query(PhotoCluster).filter_by(
                cluster_id=cluster.cluster_id
            ).all()
            
            photos = []
            for pc in photo_clusters:
                photo = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
                if photo:
                    photos.append({
                        'photo_id': photo.photo_id,
                        'filename': photo.filename,
                        'path': photo.filename
                    })
            
            cluster_list.append({
                'cluster_id': cluster.cluster_id,
                'name': cluster.name,
                'face_count': cluster.face_count,
                'thumbnail': cluster.thumbnail,
                'photos': photos
            })
        
        session.close()
        return jsonify({'clusters': cluster_list})
        
    except Exception as e:
        session.close()
        logger.error(f"Error getting clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster/<cluster_id>/photos', methods=['GET'])
def get_cluster_photos(cluster_id):
    """Get all photos for a specific person/cluster"""
    session = Session()
    
    try:
        # Get cluster info
        cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
        
        if not cluster:
            session.close()
            return jsonify({'error': 'Cluster not found'}), 404
        
        # Get photos via junction table
        photo_clusters = session.query(PhotoCluster).filter_by(
            cluster_id=cluster_id
        ).all()
        
        photos = []
        for pc in photo_clusters:
            photo = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
            if photo:
                photos.append({
                    'photo_id': photo.photo_id,
                    'filename': photo.filename,
                    'path': photo.filename
                })
        
        session.close()
        
        return jsonify({
            'cluster_id': cluster_id,
            'name': cluster.name,
            'face_count': cluster.face_count,
            'photos': photos
        })
        
    except Exception as e:
        session.close()
        logger.error(f"Error getting cluster photos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster/rename', methods=['POST'])
def rename_cluster():
    """Rename a person/cluster"""
    data = request.json
    cluster_id = data.get('cluster_id')
    new_name = data.get('name')
    
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

@app.route('/api/organize', methods=['POST'])
def organize_photos():
    """
    Organize photos into folders by person/cluster
    Photos with multiple people will be copied to multiple folders
    """
    session = Session()
    
    try:
        clusters = session.query(Cluster).all()
        organized_count = 0
        
        for cluster in clusters:
            # Create folder for this person
            person_folder = os.path.join(ORGANIZED_FOLDER, cluster.name)
            os.makedirs(person_folder, exist_ok=True)
            
            # Get photos for this cluster
            photo_clusters = session.query(PhotoCluster).filter_by(
                cluster_id=cluster.cluster_id
            ).all()
            
            for pc in photo_clusters:
                photo = session.query(Photo).filter_by(photo_id=pc.photo_id).first()
                if photo and os.path.exists(photo.path):
                    dest_path = os.path.join(person_folder, photo.filename)
                    if not os.path.exists(dest_path):  # Avoid duplicates
                        shutil.copy2(photo.path, dest_path)
                        organized_count += 1
        
        session.close()
        
        logger.info(f"✓ Organized {organized_count} photos into folders")
        
        return jsonify({
            'success': True,
            'organized_count': organized_count,
            'output_folder': ORGANIZED_FOLDER
        })
        
    except Exception as e:
        session.close()
        logger.error(f"Error organizing photos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the photo library"""
    session = Session()
    
    try:
        stats = {
            'total_photos': session.query(Photo).count(),
            'total_clusters': session.query(Cluster).count(),
            'processed_faces': session.query(FaceEmbedding).count(),
            'detected_objects': session.query(DetectedObject).count(),
            'photos_with_emotions': session.query(Photo).filter(
                Photo.dominant_emotion != None
            ).count(),
            'photos_with_scenes': session.query(Photo).filter(
                Photo.scene_type != None
            ).count()
        }
        
        session.close()
        return jsonify(stats)
        
    except Exception as e:
        session.close()
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_database():
    """
    Reset all data (for testing/development)
    WARNING: This deletes everything!
    """
    session = Session()
    
    try:
        # Delete all records (cascade will handle relationships)
        session.query(PhotoCluster).delete()
        session.query(FaceEmbedding).delete()
        session.query(DetectedObject).delete()
        session.query(Cluster).delete()
        session.query(Photo).delete()
        session.commit()
        session.close()
        
        # Clear folders
        for folder in [UPLOAD_FOLDER, THUMBNAILS_FOLDER, ORGANIZED_FOLDER]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
        
        logger.info("✓ Database reset complete")
        
        return jsonify({
            'success': True,
            'message': 'All data reset'
        })
        
    except Exception as e:
        session.rollback()
        session.close()
        logger.error(f"Error resetting database: {str(e)}")
        return jsonify({'error': str(e)}), 500

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