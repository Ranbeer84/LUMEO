-- Phase 2 Schema Enhancements
-- Add new columns and tables for advanced vision features

-- 1. Add emotion and quality columns to face_embeddings
ALTER TABLE face_embeddings 
ADD COLUMN IF NOT EXISTS emotion VARCHAR(20),
ADD COLUMN IF NOT EXISTS emotion_confidence FLOAT,
ADD COLUMN IF NOT EXISTS emotion_valence FLOAT,
ADD COLUMN IF NOT EXISTS quality_score FLOAT;

-- 2. Add CLIP embedding and metadata to photos
ALTER TABLE photos
ADD COLUMN IF NOT EXISTS clip_embedding vector(512),
ADD COLUMN IF NOT EXISTS scene_type VARCHAR(20),
ADD COLUMN IF NOT EXISTS location_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS activity VARCHAR(50),
ADD COLUMN IF NOT EXISTS season VARCHAR(20),
ADD COLUMN IF NOT EXISTS time_of_day VARCHAR(20),
ADD COLUMN IF NOT EXISTS date_taken TIMESTAMP,
ADD COLUMN IF NOT EXISTS camera_make VARCHAR(100),
ADD COLUMN IF NOT EXISTS camera_model VARCHAR(100),
ADD COLUMN IF NOT EXISTS gps_latitude FLOAT,
ADD COLUMN IF NOT EXISTS gps_longitude FLOAT,
ADD COLUMN IF NOT EXISTS image_quality FLOAT,
ADD COLUMN IF NOT EXISTS caption TEXT,
ADD COLUMN IF NOT EXISTS dominant_emotion VARCHAR(20),
ADD COLUMN IF NOT EXISTS mood_score FLOAT;

-- 3. Create detected_objects table
CREATE TABLE IF NOT EXISTS detected_objects (
    object_id SERIAL PRIMARY KEY,
    photo_id VARCHAR(255) REFERENCES photos(photo_id) ON DELETE CASCADE,
    label VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    dominant_color_rgb VARCHAR(50),
    color_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create index on photo_id for faster queries
CREATE INDEX IF NOT EXISTS idx_objects_photo ON detected_objects(photo_id);

-- 5. Create index on label for object searches
CREATE INDEX IF NOT EXISTS idx_objects_label ON detected_objects(label);

-- 6. Create index on color for color searches
CREATE INDEX IF NOT EXISTS idx_objects_color ON detected_objects(color_name);

-- 7. Create vector index for CLIP embeddings (for similarity search)
CREATE INDEX IF NOT EXISTS idx_photos_clip_embedding ON photos 
USING ivfflat (clip_embedding vector_cosine_ops)
WITH (lists = 100);

-- 8. Create indexes for common search patterns
CREATE INDEX IF NOT EXISTS idx_photos_date_taken ON photos(date_taken);
CREATE INDEX IF NOT EXISTS idx_photos_season ON photos(season);
CREATE INDEX IF NOT EXISTS idx_photos_time_of_day ON photos(time_of_day);
CREATE INDEX IF NOT EXISTS idx_photos_scene_type ON photos(scene_type);
CREATE INDEX IF NOT EXISTS idx_photos_emotion ON photos(dominant_emotion);
CREATE INDEX IF NOT EXISTS idx_photos_mood ON photos(mood_score);

-- 9. Create index for face emotions
CREATE INDEX IF NOT EXISTS idx_faces_emotion ON face_embeddings(emotion);

-- 10. Add comments for documentation
COMMENT ON COLUMN photos.clip_embedding IS 'CLIP image embedding for semantic search';
COMMENT ON COLUMN photos.scene_type IS 'Indoor/outdoor classification';
COMMENT ON COLUMN photos.location_type IS 'Specific location (beach, office, home, etc.)';
COMMENT ON COLUMN photos.activity IS 'Activity detected (sports, dining, party, etc.)';
COMMENT ON COLUMN photos.caption IS 'Auto-generated natural language caption';
COMMENT ON COLUMN photos.mood_score IS 'Overall mood score from -1 (negative) to +1 (positive)';
COMMENT ON TABLE detected_objects IS 'Objects detected by YOLO in photos';