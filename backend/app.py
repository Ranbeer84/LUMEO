from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
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

    from services.query_service import get_query_service
    from services.retrieval_service import get_retrieval_service
    from services.context_service import get_context_service
    from services.query_parser import get_query_parser

    from services.llm_service import get_llm_service
    from services.conversation_service import get_conversation_service
    
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
    
@app.route('/api/search', methods=['POST'])
def search_photos():
    """
    Natural language photo search with hybrid retrieval
    
    Request body:
        {
            "query": "beach photos with Mom from last summer",
            "top_k": 10,
            "use_filters": true
        }
    
    Returns: List of retrieved photos with similarity scores
    """
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        use_filters = data.get('use_filters', True)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"=== SEARCH REQUEST ===")
        logger.info(f"Query: {query}")
        logger.info(f"Top K: {top_k}")
        logger.info(f"Use filters: {use_filters}")
        
        # Get services
        query_service = get_query_service()
        retrieval_service = get_retrieval_service()
        
        # Step 1: Parse query into structured filters
        session = Session()
        clusters = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        session.close()
        
        parser = get_query_parser(known_people)
        parsed_filters = parser.parse(query)
        
        logger.info(f"Parsed filters: {parsed_filters}")
        
        # Step 2: Generate query embedding
        query_embedding = query_service.encode_query(query)
        
        if query_embedding is None:
            return jsonify({'error': 'Failed to encode query'}), 500
        
        # Step 3: Perform hybrid search
        if use_filters and len(parsed_filters) > 1:  # Has filters beyond raw_query
            # Remove raw_query from filters for database search
            db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
            results = retrieval_service.hybrid_search(
                query_embedding,
                filters=db_filters,
                top_k=top_k
            )
        else:
            # Pure semantic search
            results = retrieval_service.semantic_search(
                query_embedding,
                top_k=top_k,
                min_similarity=0.3
            )
        
        logger.info(f"✓ Retrieved {len(results)} photos")
        
        return jsonify({
            'success': True,
            'query': query,
            'parsed_filters': parsed_filters,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/context', methods=['POST'])
def get_search_context():
    """
    Get LLM-ready context for search results
    
    Request body:
        {
            "query": "beach photos with Mom",
            "top_k": 10,
            "include_system_prompt": true
        }
    
    Returns: Formatted context string ready for LLM
    """
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        include_system_prompt = data.get('include_system_prompt', True)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get services
        query_service = get_query_service()
        retrieval_service = get_retrieval_service()
        context_service = get_context_service()
        
        # Parse query
        session = Session()
        clusters = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        session.close()
        
        parser = get_query_parser(known_people)
        parsed_filters = parser.parse(query)
        
        # Generate embedding
        query_embedding = query_service.encode_query(query)
        
        if query_embedding is None:
            return jsonify({'error': 'Failed to encode query'}), 500
        
        # Search
        db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
        results = retrieval_service.hybrid_search(
            query_embedding,
            filters=db_filters,
            top_k=top_k
        )
        
        # Build context
        context = context_service.build_context(
            results,
            query,
            include_system_prompt=include_system_prompt
        )
        
        estimated_tokens = context_service.estimate_tokens(context)
        
        logger.info(f"✓ Built context: {len(results)} photos, ~{estimated_tokens} tokens")
        
        return jsonify({
            'success': True,
            'query': query,
            'results_count': len(results),
            'context': context,
            'estimated_tokens': estimated_tokens
        })
        
    except Exception as e:
        logger.error(f"Context generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/similar/<photo_id>', methods=['GET'])
def find_similar_photos(photo_id):
    """
    Find photos similar to a given photo
    
    URL params:
        - top_k: Number of results (default: 10)
    
    Returns: List of similar photos
    """
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        top_k = request.args.get('top_k', 10, type=int)
        
        retrieval_service = get_retrieval_service()
        
        results = retrieval_service.search_by_similar_photo(
            photo_id,
            top_k=top_k,
            exclude_self=True
        )
        
        logger.info(f"✓ Found {len(results)} similar photos to {photo_id}")
        
        return jsonify({
            'success': True,
            'reference_photo_id': photo_id,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Similar photo search error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights/summary', methods=['POST'])
def generate_summary():
    """
    Generate aggregated summary/insights from photos
    
    Request body:
        {
            "filters": {...},  # Optional filters
            "summary_type": "general"  # "general", "emotional", "temporal", "people"
        }
    
    Returns: Summary context for LLM
    """
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        data = request.json
        filters = data.get('filters', {})
        summary_type = data.get('summary_type', 'general')
        
        # Get all photos or filtered subset
        session = Session()
        query = session.query(Photo).filter(Photo.clip_embedding.isnot(None))
        
        # Apply filters if provided
        # (You can expand this to apply filters similar to hybrid search)
        
        photos = query.limit(100).all()  # Limit for performance
        
        # Convert to dict format
        photo_dicts = []
        for photo in photos:
            # Get people
            photo_clusters = session.query(PhotoCluster).filter_by(
                photo_id=photo.photo_id
            ).all()
            
            people = []
            for pc in photo_clusters:
                cluster = session.query(Cluster).filter_by(
                    cluster_id=pc.cluster_id
                ).first()
                if cluster:
                    people.append(cluster.name)
            
            photo_dicts.append({
                'photo_id': photo.photo_id,
                'people': people,
                'dominant_emotion': photo.dominant_emotion,
                'location': photo.location_type,
                'activity': photo.activity,
                'season': photo.season,
                'time_of_day': photo.time_of_day
            })
        
        session.close()
        
        # Generate summary
        context_service = get_context_service()
        summary = context_service.build_summary_context(
            photo_dicts,
            summary_type=summary_type
        )
        
        logger.info(f"✓ Generated {summary_type} summary for {len(photo_dicts)} photos")
        
        return jsonify({
            'success': True,
            'summary_type': summary_type,
            'photos_analyzed': len(photo_dicts),
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrieval/stats', methods=['GET'])
def retrieval_stats():
    """Get retrieval system statistics"""
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        retrieval_service = get_retrieval_service()
        query_service = get_query_service()
        
        stats = retrieval_service.get_retrieval_stats()
        cache_stats = query_service.get_cache_stats()
        
        return jsonify({
            'success': True,
            'retrieval_stats': stats,
            'cache_stats': cache_stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query/parse', methods=['POST'])
def parse_query():
    """
    Test endpoint: Parse a query and show extracted filters
    
    Useful for debugging query parsing
    
    Request body:
        {
            "query": "beach photos with Mom from last summer"
        }
    
    Returns: Parsed filters
    """
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get known people
        session = Session()
        clusters = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        session.close()
        
        # Parse
        parser = get_query_parser(known_people)
        filters = parser.parse(query)
        
        # Get semantic query reconstruction
        semantic_query = parser.get_semantic_query(filters)
        
        return jsonify({
            'success': True,
            'original_query': query,
            'parsed_filters': filters,
            'semantic_query': semantic_query,
            'known_people': known_people
        })
        
    except Exception as e:
        logger.error(f"Query parsing error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Example: Complete search workflow
# ============================================================================
"""
USAGE EXAMPLE:

1. Simple search:
POST /api/search
{
    "query": "beach photos with Mom",
    "top_k": 10
}

2. Get LLM context:
POST /api/search/context
{
    "query": "show me happy family photos from last summer",
    "top_k": 5,
    "include_system_prompt": true
}

3. Find similar photos:
GET /api/search/similar/photo_123?top_k=10

4. Generate insights:
POST /api/insights/summary
{
    "summary_type": "emotional"
}

5. Test query parsing:
POST /api/query/parse
{
    "query": "photos where I'm wearing a red dress at the beach"
}
"""

# ============================================================================
# PHASE 4: CHAT / LLM ROUTES
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Send a message and get LLM response
    
    Request body:
        {
            "message": "Show me happy beach photos",
            "conversation_id": "conv_abc123",  // optional
            "top_k": 5,
            "stream": false
        }
    
    Returns: LLM response with retrieved photos
    """
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        data = request.json
        message = data.get('message', '')
        conversation_id = data.get('conversation_id')
        top_k = data.get('top_k', 5)
        use_streaming = data.get('stream', False)
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        logger.info(f"=== CHAT REQUEST ===")
        logger.info(f"Message: {message}")
        logger.info(f"Conversation: {conversation_id}")
        
        # Get services
        query_service = get_query_service()
        retrieval_service = get_retrieval_service()
        context_service = get_context_service()
        llm_service = get_llm_service()
        conversation_service = get_conversation_service()
        
        session = Session()
        
        # Create new conversation if needed
        if not conversation_id:
            conversation_id = conversation_service.create_conversation(session)
            logger.info(f"Created new conversation: {conversation_id}")
        
        # Parse query
        clusters = session.query(Cluster).all()
        known_people = [c.name for c in clusters]
        
        parser = get_query_parser(known_people)
        parsed_filters = parser.parse(message)
        
        # Generate query embedding
        query_embedding = query_service.encode_query(message)
        
        if query_embedding is None:
            session.close()
            return jsonify({'error': 'Failed to encode query'}), 500
        
        # Retrieve photos
        db_filters = {k: v for k, v in parsed_filters.items() if k != 'raw_query'}
        retrieved_photos = retrieval_service.hybrid_search(
            query_embedding,
            filters=db_filters,
            top_k=top_k
        )
        
        logger.info(f"✓ Retrieved {len(retrieved_photos)} photos")
        
        # Build context
        photo_context = context_service.build_context(
            retrieved_photos,
            message,
            include_system_prompt=False  # We'll add it in LLM service
        )
        
        # Include conversation history
        full_context = conversation_service.build_context_with_history(
            session,
            conversation_id,
            message,
            photo_context
        )
        
        # Save user message
        photo_ids = [p['photo_id'] for p in retrieved_photos]
        conversation_service.add_message(
            session,
            conversation_id,
            role='user',
            content=message,
            retrieved_photo_ids=photo_ids,
            metadata={
                'filters': parsed_filters,
                'results_count': len(retrieved_photos)
            }
        )
        
        # Generate LLM response
        if use_streaming:
            # Return streaming response
            session.close()
            return Response(
                stream_chat_response(
                    llm_service,
                    conversation_service,
                    conversation_id,
                    full_context,
                    message,
                    retrieved_photos
                ),
                mimetype='text/event-stream'
            )
        else:
            # Generate complete response
            response_text = llm_service.generate_response(
                context=full_context,
                query=message
            )
            
            # Validate response
            validation = llm_service.validate_response(response_text, full_context)
            
            # Save assistant message
            conversation_service.add_message(
                session,
                conversation_id,
                role='assistant',
                content=response_text,
                metadata={
                    'validation': validation
                }
            )
            
            session.close()
            
            logger.info(f"✓ Generated response ({len(response_text)} chars)")
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id,
                'message': message,
                'response': response_text,
                'retrieved_photos': retrieved_photos,
                'validation': validation
            })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def stream_chat_response(llm_service, conversation_service, conversation_id, context, query, retrieved_photos):
    """
    Generator for streaming chat responses
    
    Yields Server-Sent Events (SSE) format
    """
    full_response = []
    
    try:
        # First, send retrieved photos
        yield f"data: {json.dumps({'type': 'photos', 'photos': retrieved_photos})}\n\n"
        
        # Then stream LLM response
        for chunk in llm_service.generate_streaming_response(context, query):
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        
        # Save complete response
        response_text = ''.join(full_response)
        
        session = Session()
        conversation_service.add_message(
            session,
            conversation_id,
            role='assistant',
            content=response_text
        )
        session.close()
        
        # Send completion event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint (alias for chat with stream=true)
    """
    data = request.json or {}
    data['stream'] = True
    request.json = data
    return chat()


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """
    Get all conversations
    
    Query params:
        - limit: Max conversations to return (default: 20)
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        
        session = Session()
        conversation_service = get_conversation_service()
        
        conversations = conversation_service.get_all_conversations(
            session,
            limit=limit
        )
        
        session.close()
        
        return jsonify({
            'success': True,
            'conversations': conversations
        })
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get conversation history"""
    try:
        session = Session()
        conversation_service = get_conversation_service()
        
        # Get messages
        history = conversation_service.get_conversation_history(
            session,
            conversation_id
        )
        
        # Get stats
        stats = conversation_service.get_conversation_stats(
            session,
            conversation_id
        )
        
        session.close()
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'messages': history,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/new', methods=['POST'])
def new_conversation():
    """Start a new conversation"""
    try:
        session = Session()
        conversation_service = get_conversation_service()
        
        conversation_id = conversation_service.create_conversation(session)
        
        session.close()
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation"""
    try:
        session = Session()
        conversation_service = get_conversation_service()
        
        success = conversation_service.delete_conversation(
            session,
            conversation_id
        )
        
        session.close()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Conversation {conversation_id} deleted'
            })
        else:
            return jsonify({'error': 'Failed to delete conversation'}), 500
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights', methods=['POST'])
def generate_insights():
    """
    Generate insights from photo collection
    
    Request body:
        {
            "insight_type": "emotional",  // emotional, social, temporal, general
            "filters": {...}  // optional filters
        }
    """
    if not SERVICES_AVAILABLE:
        return jsonify({'error': 'Services not available'}), 500
    
    try:
        data = request.json
        insight_type = data.get('insight_type', 'general')
        filters = data.get('filters', {})
        
        # Get photos
        session = Session()
        query = session.query(Photo).filter(Photo.clip_embedding.isnot(None))
        photos = query.limit(100).all()
        
        # Convert to dicts
        photo_dicts = []
        for photo in photos:
            photo_clusters = session.query(PhotoCluster).filter_by(
                photo_id=photo.photo_id
            ).all()
            
            people = []
            for pc in photo_clusters:
                cluster = session.query(Cluster).filter_by(
                    cluster_id=pc.cluster_id
                ).first()
                if cluster:
                    people.append(cluster.name)
            
            photo_dicts.append({
                'photo_id': photo.photo_id,
                'people': people,
                'dominant_emotion': photo.dominant_emotion,
                'location': photo.location_type,
                'activity': photo.activity,
                'season': photo.season,
                'time_of_day': photo.time_of_day
            })
        
        # Generate summary context
        context_service = get_context_service()
        summary = context_service.build_summary_context(
            photo_dicts,
            summary_type=insight_type
        )
        
        # Generate insights with LLM
        llm_service = get_llm_service()
        insights = llm_service.generate_insight(
            summary_context=summary,
            insight_type=insight_type
        )
        
        session.close()
        
        logger.info(f"✓ Generated {insight_type} insights")
        
        return jsonify({
            'success': True,
            'insight_type': insight_type,
            'photos_analyzed': len(photo_dicts),
            'summary': summary,
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Insights generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/status', methods=['GET'])
def llm_status():
    """Check LLM service status"""
    try:
        llm_service = get_llm_service()
        model_info = llm_service.get_model_info()
        
        return jsonify({
            'success': True,
            'model': llm_service.model,
            'base_url': llm_service.base_url,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Example: Complete chat workflow
# ============================================================================
"""
USAGE EXAMPLE:

1. Start new conversation:
POST /api/conversation/new
Response: {"conversation_id": "conv_abc123"}

2. Send message:
POST /api/chat
{
    "message": "Show me happy beach photos with Mom",
    "conversation_id": "conv_abc123",
    "top_k": 5
}

Response:
{
    "response": "I found 5 happy beach photos with Mom. In Photo 1...",
    "retrieved_photos": [...],
    "validation": {
        "is_grounded": true,
        "confidence": 0.95
    }
}

3. Continue conversation:
POST /api/chat
{
    "message": "When were these taken?",
    "conversation_id": "conv_abc123"
}
// LLM has context from previous exchange

4. Streaming chat:
POST /api/chat/stream
{
    "message": "Tell me about my summer photos",
    "conversation_id": "conv_abc123"
}
// Returns SSE stream of tokens

5. Get insights:
POST /api/insights
{
    "insight_type": "emotional"
}

Response:
{
    "insights": "You appear happiest in beach photos (78% happy vs 45% overall)..."
}

6. Get conversation history:
GET /api/conversation/conv_abc123

7. List all conversations:
GET /api/conversations?limit=10
"""

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