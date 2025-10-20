"""
Photo Organizer Backend - Flask API with Face Recognition (FIXED)
Fixed clustering algorithm with proper face-to-person mapping
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import os
import shutil
import json
from datetime import datetime
import sqlite3
import time
from pathlib import Path
import cv2

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ORGANIZED_FOLDER = 'organized_photos'
THUMBNAILS_FOLDER = 'thumbnails'
DB_PATH = 'photo_organizer.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORGANIZED_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    return send_from_directory(THUMBNAILS_FOLDER, filename)

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Photos table - removed cluster_id since photos can have multiple people
    c.execute('''CREATE TABLE IF NOT EXISTS photos
                 (photo_id TEXT PRIMARY KEY,
                  filename TEXT,
                  path TEXT,
                  upload_date REAL)''')
    
    # Clusters/People table
    c.execute('''CREATE TABLE IF NOT EXISTS clusters
                 (cluster_id TEXT PRIMARY KEY, 
                  name TEXT, 
                  face_count INTEGER,
                  thumbnail TEXT,
                  created_at REAL)''')
    
    # Face embeddings - links faces to both photos and clusters
    c.execute('''CREATE TABLE IF NOT EXISTS face_embeddings
                 (embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  photo_id TEXT,
                  cluster_id TEXT,
                  embedding BLOB,
                  face_location TEXT,
                  FOREIGN KEY (photo_id) REFERENCES photos(photo_id),
                  FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id))''')
    
    # Junction table for many-to-many relationship between photos and clusters
    c.execute('''CREATE TABLE IF NOT EXISTS photo_clusters
                 (photo_id TEXT,
                  cluster_id TEXT,
                  PRIMARY KEY (photo_id, cluster_id),
                  FOREIGN KEY (photo_id) REFERENCES photos(photo_id),
                  FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id))''')
    
    conn.commit()
    conn.close()

init_db()

class FaceRecognitionService:
    """Handle all face recognition operations"""
    
    def __init__(self):
        self.face_encodings = []
        self.face_locations = []
        self.photo_ids = []
        
    def detect_faces(self, image_path):
        """Detect faces in an image and return encodings"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find all face locations and encodings
            # Using 'hog' for speed, change to 'cnn' for better accuracy
            face_locations = face_recognition.face_locations(image, model="hog")
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            return face_encodings, face_locations
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {str(e)}")
            return [], []
    
    def cluster_faces(self, encodings, min_samples=1, eps=0.6):
        """
        Cluster face encodings using DBSCAN
        
        FIXED: Better parameters for face clustering
        - min_samples=1: Each face can form a cluster (good for single photos of a person)
        - eps=0.6: Distance threshold (0.6 is standard for face_recognition library)
        """
        if len(encodings) == 0:
            return []
        
        # Convert to numpy array
        encodings_array = np.array(encodings)
        
        # Use DBSCAN for clustering
        # eps=0.6 is the recommended threshold for face_recognition encodings
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(encodings_array)
        
        return labels
    
    def extract_face_thumbnail(self, image_path, face_location, output_path):
        """Extract and save face thumbnail"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
                
            top, right, bottom, left = face_location
            
            # Add padding
            padding = 40
            height, width = image.shape[:2]
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(height, bottom + padding)
            right = min(width, right + padding)
            
            face_image = image[top:bottom, left:right]
            
            # Resize to consistent thumbnail size
            face_image = cv2.resize(face_image, (200, 200))
            cv2.imwrite(output_path, face_image)
            return True
        except Exception as e:
            print(f"Error extracting face: {str(e)}")
            return False

face_service = FaceRecognitionService()

@app.route('/api/upload', methods=['POST'])
def upload_photos():
    """Upload photos endpoint"""
    if 'photos' not in request.files:
        return jsonify({'error': 'No photos uploaded'}), 400
    
    files = request.files.getlist('photos')
    uploaded_photos = []
    
    for file in files:
        if file.filename:
            # Save file
            timestamp = datetime.now().timestamp()
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Store in database
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            photo_id = f"photo_{timestamp}_{len(uploaded_photos)}"
            c.execute('''INSERT INTO photos (photo_id, filename, path, upload_date)
                        VALUES (?, ?, ?, ?)''',
                     (photo_id, filename, filepath, time.time()))
            conn.commit()
            conn.close()
            
            uploaded_photos.append({
                'photo_id': photo_id,
                'filename': filename,
                'path': filepath
            })
    
    return jsonify({
        'success': True,
        'photos_count': len(uploaded_photos),
        'photos': uploaded_photos
    })

@app.route('/api/process', methods=['POST'])
def process_photos():
    """
    FIXED: Process photos with proper face-to-cluster mapping
    Each face gets its own cluster assignment, photos can belong to multiple clusters
    """
    try:
        # Get all unprocessed photos
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get photos that haven't been processed yet
        c.execute('''SELECT DISTINCT p.photo_id, p.path 
                    FROM photos p 
                    LEFT JOIN face_embeddings fe ON p.photo_id = fe.photo_id 
                    WHERE fe.photo_id IS NULL''')
        photos = c.fetchall()
        
        if not photos:
            conn.close()
            return jsonify({'message': 'No unprocessed photos found'}), 200
        
        # Data structures to track all faces
        all_encodings = []
        all_face_data = []  # Store photo_id, path, and location for each face
        
        # Process each photo and extract all faces
        print(f"Processing {len(photos)} photos...")
        for photo_id, path in photos:
            if not os.path.exists(path):
                print(f"Photo not found: {path}")
                continue
                
            encodings, locations = face_service.detect_faces(path)
            print(f"Found {len(encodings)} faces in {path}")
            
            for encoding, location in zip(encodings, locations):
                all_encodings.append(encoding)
                all_face_data.append({
                    'photo_id': photo_id,
                    'path': path,
                    'location': location,
                    'encoding': encoding
                })
        
        if len(all_encodings) == 0:
            conn.close()
            return jsonify({'error': 'No faces detected in photos'}), 400
        
        print(f"Total faces detected: {len(all_encodings)}")
        
        # Cluster all faces together
        labels = face_service.cluster_faces(all_encodings, min_samples=1, eps=0.6)
        print(f"Clustering result: {len(set(labels))} unique clusters")
        
        # Organize faces by cluster
        clusters = {}
        
        for idx, label in enumerate(labels):
            if label == -1:  # Skip outliers (though with min_samples=1, should be rare)
                print(f"Outlier face detected at index {idx}")
                continue
            
            cluster_id = f"cluster_{label}"
            
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'faces': [],
                    'photos': set(),  # Track unique photos
                    'encodings': []
                }
            
            face_data = all_face_data[idx]
            clusters[cluster_id]['faces'].append(face_data)
            clusters[cluster_id]['photos'].add(face_data['photo_id'])
            clusters[cluster_id]['encodings'].append(face_data['encoding'])
        
        print(f"Created {len(clusters)} clusters")
        
        # Save clusters to database
        for cluster_id, data in clusters.items():
            # Find the most representative face (closest to cluster center)
            encodings_array = np.array(data['encodings'])
            cluster_center = np.mean(encodings_array, axis=0)
            
            # Calculate distances from each face to the cluster center
            distances = [np.linalg.norm(enc - cluster_center) for enc in data['encodings']]
            
            # Get the face closest to center (most representative)
            best_face_idx = np.argmin(distances)
            representative_face = data['faces'][best_face_idx]
            
            print(f"{cluster_id}: Selected face {best_face_idx} out of {len(data['faces'])} as thumbnail")
            
            # Create thumbnail from most representative face
            thumbnail_filename = f"{cluster_id}_thumb.jpg"
            thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_filename)
            
            face_service.extract_face_thumbnail(
                representative_face['path'],
                representative_face['location'],
                thumbnail_path
            )
            
            # Insert or update cluster
            c.execute('''INSERT OR REPLACE INTO clusters 
                        (cluster_id, name, face_count, thumbnail, created_at)
                        VALUES (?, ?, ?, ?, ?)''',
                     (cluster_id, f"Person {cluster_id.split('_')[1]}", 
                      len(data['faces']), thumbnail_filename, time.time()))
            
            # Store each face embedding with its photo and cluster
            for face_data in data['faces']:
                c.execute('''INSERT INTO face_embeddings 
                            (photo_id, cluster_id, embedding, face_location)
                            VALUES (?, ?, ?, ?)''',
                         (face_data['photo_id'], cluster_id, 
                          face_data['encoding'].tobytes(), 
                          json.dumps(face_data['location'])))
                
                # Link photo to cluster (many-to-many)
                c.execute('''INSERT OR IGNORE INTO photo_clusters 
                            (photo_id, cluster_id)
                            VALUES (?, ?)''',
                         (face_data['photo_id'], cluster_id))
        
        conn.commit()
        
        # Get cluster info for response
        c.execute('SELECT cluster_id, name, face_count, thumbnail FROM clusters')
        cluster_info = [{'cluster_id': row[0], 'name': row[1], 
                        'face_count': row[2], 'thumbnail': row[3]} 
                       for row in c.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'clusters': cluster_info,
            'total_faces': len(all_encodings),
            'total_clusters': len(clusters)
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get all clusters with their photos"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT cluster_id, name, face_count, thumbnail FROM clusters')
    clusters = []
    
    for row in c.fetchall():
        cluster_id, name, face_count, thumbnail = row
        
        # Get photos for this cluster (via junction table)
        c.execute('''SELECT DISTINCT p.photo_id, p.filename, p.path 
                    FROM photos p
                    JOIN photo_clusters pc ON p.photo_id = pc.photo_id
                    WHERE pc.cluster_id = ?''',
                 (cluster_id,))
        photos = [{'photo_id': p[0], 'filename': p[1], 'path': p[1]} 
                 for p in c.fetchall()]
        
        clusters.append({
            'cluster_id': cluster_id,
            'name': name,
            'face_count': face_count,
            'photos': photos,
            'thumbnail': thumbnail
        })
    
    conn.close()
    return jsonify({'clusters': clusters})

@app.route('/api/cluster/<cluster_id>/photos', methods=['GET'])
def get_cluster_photos(cluster_id):
    """Get all photos for a specific cluster"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get cluster info
    c.execute('SELECT name, face_count FROM clusters WHERE cluster_id = ?', (cluster_id,))
    cluster_info = c.fetchone()
    
    if not cluster_info:
        conn.close()
        return jsonify({'error': 'Cluster not found'}), 404
    
    # Get photos via junction table
    c.execute('''SELECT DISTINCT p.photo_id, p.filename, p.path 
                FROM photos p
                JOIN photo_clusters pc ON p.photo_id = pc.photo_id
                WHERE pc.cluster_id = ?''',
             (cluster_id,))
    photos = [{'photo_id': p[0], 'filename': p[1], 'path': p[1]} 
             for p in c.fetchall()]
    
    conn.close()
    
    return jsonify({
        'cluster_id': cluster_id,
        'name': cluster_info[0],
        'face_count': cluster_info[1],
        'photos': photos
    })

@app.route('/api/cluster/rename', methods=['POST'])
def rename_cluster():
    """Rename a cluster"""
    data = request.json
    cluster_id = data.get('cluster_id')
    new_name = data.get('name')
    
    if not cluster_id or not new_name:
        return jsonify({'error': 'Missing cluster_id or name'}), 400
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE clusters SET name = ? WHERE cluster_id = ?',
             (new_name, cluster_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/organize', methods=['POST'])
def organize_photos():
    """
    FIXED: Organize photos into folders by cluster/person
    Photos with multiple people will be copied to multiple folders
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('SELECT cluster_id, name FROM clusters')
        clusters = c.fetchall()
        
        organized_count = 0
        
        for cluster_id, name in clusters:
            # Create folder for this person
            person_folder = os.path.join(ORGANIZED_FOLDER, name)
            os.makedirs(person_folder, exist_ok=True)
            
            # Get photos for this cluster via junction table
            c.execute('''SELECT DISTINCT p.photo_id, p.path, p.filename 
                        FROM photos p
                        JOIN photo_clusters pc ON p.photo_id = pc.photo_id
                        WHERE pc.cluster_id = ?''',
                     (cluster_id,))
            photos = c.fetchall()
            
            # Copy photos to person's folder
            for photo_id, path, filename in photos:
                if os.path.exists(path):
                    dest_path = os.path.join(person_folder, filename)
                    if not os.path.exists(dest_path):  # Avoid duplicate copies
                        shutil.copy2(path, dest_path)
                        organized_count += 1
        
        conn.close()
        
        return jsonify({
            'success': True,
            'organized_count': organized_count,
            'output_folder': ORGANIZED_FOLDER
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about processed photos"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM photos')
    total_photos = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM clusters')
    total_clusters = c.fetchone()[0]
    
    c.execute('SELECT SUM(face_count) FROM clusters')
    total_faces = c.fetchone()[0] or 0
    
    c.execute('SELECT COUNT(*) FROM face_embeddings')
    processed_faces = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_photos': total_photos,
        'total_clusters': total_clusters,
        'total_faces': total_faces,
        'processed_faces': processed_faces
    })

@app.route('/api/reset', methods=['POST'])
def reset_database():
    """Reset all data (for testing)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM photo_clusters')
    c.execute('DELETE FROM face_embeddings')
    c.execute('DELETE FROM photos')
    c.execute('DELETE FROM clusters')
    conn.commit()
    conn.close()
    
    # Clear folders
    for folder in [UPLOAD_FOLDER, THUMBNAILS_FOLDER, ORGANIZED_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    return jsonify({'success': True, 'message': 'All data reset'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)