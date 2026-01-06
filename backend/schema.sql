
-- LUMEO DATABASE SCHEMA - PostgreSQL + pgvector
-- Multi-Modal AI Photo Memory System

-- This schema supports:
-- - Face recognition and clustering
-- - Emotion detection
-- - Object detection (YOLO)
-- - Scene classification
-- - CLIP embeddings for semantic search
-- - Conversational AI with memory
-- - RAG-based retrieval


-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- CORE PHOTO TABLES

-- Photos table (enhanced with AI metadata)
CREATE TABLE photos (
    photo_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    file_hash VARCHAR(64),  -- For duplicate detection
    
    -- Timestamps
    upload_date TIMESTAMP DEFAULT NOW(),
    taken_date TIMESTAMP,  -- From EXIF
    processed_date TIMESTAMP,
    
    -- Image metadata
    width INTEGER,
    height INTEGER,
    format VARCHAR(10),  -- jpg, png, etc.
    camera_make VARCHAR(100),
    camera_model VARCHAR(100),
    
    -- Location (from EXIF GPS)
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    location_name TEXT,  -- Reverse geocoded
    
    -- Temporal context (inferred from taken_date)
    season VARCHAR(20),  -- winter, spring, summer, autumn
    time_of_day VARCHAR(20),  -- morning, afternoon, evening, night
    year INTEGER,
    month INTEGER,
    day_of_week VARCHAR(10),
    
    -- Scene classification
    scene_type VARCHAR(50),  -- indoor, outdoor, beach, mountain, city, etc.
    activity VARCHAR(50),  -- party, sports, dining, working, etc.
    is_indoor BOOLEAN,
    
    -- Image quality metrics
    blur_score DECIMAL(5, 2),  -- 0-100, higher = sharper
    brightness_score DECIMAL(5, 2),  -- 0-100
    overall_quality DECIMAL(5, 2),  -- 0-100
    
    -- Mood/emotion aggregation
    dominant_emotion VARCHAR(20),  -- happy, sad, neutral, etc.
    emotion_confidence DECIMAL(5, 2),
    overall_mood_score DECIMAL(5, 2),  -- -1 (negative) to +1 (positive)
    
    -- CLIP embeddings for semantic search
    clip_embedding vector(512),  -- CLIP ViT-B/32 produces 512-d vectors
    
    -- Caption and description
    generated_caption TEXT,  -- Auto-generated description
    user_caption TEXT,  -- User-provided description
    searchable_text TEXT,  -- Combined text for full-text search
    
    -- Processing status
    is_processed BOOLEAN DEFAULT FALSE,
    processing_error TEXT,
    
    -- Soft delete
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_date TIMESTAMP,
    
    CONSTRAINT valid_mood_score CHECK (overall_mood_score BETWEEN -1 AND 1),
    CONSTRAINT valid_quality CHECK (overall_quality BETWEEN 0 AND 100)
);

-- Indexes for photos table
CREATE INDEX idx_photos_upload_date ON photos(upload_date);
CREATE INDEX idx_photos_taken_date ON photos(taken_date);
CREATE INDEX idx_photos_season ON photos(season);
CREATE INDEX idx_photos_scene_type ON photos(scene_type);
CREATE INDEX idx_photos_is_processed ON photos(is_processed);
CREATE INDEX idx_photos_is_deleted ON photos(is_deleted) WHERE is_deleted = FALSE;

-- Vector index for CLIP embeddings (add after you have 1000+ photos)
-- CREATE INDEX idx_photos_clip_embedding ON photos 
-- USING ivfflat (clip_embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index
CREATE INDEX idx_photos_searchable_text ON photos 
USING gin(to_tsvector('english', searchable_text));


-- FACE DETECTION & CLUSTERING


-- Clusters (people) table
CREATE TABLE clusters (
    cluster_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL DEFAULT 'Unknown Person',
    
    -- Cluster metadata
    face_count INTEGER DEFAULT 0,
    photo_count INTEGER DEFAULT 0,
    
    -- Representative face
    thumbnail_path TEXT,
    representative_embedding vector(128),  -- face_recognition uses 128-d
    
    -- Quality metrics for this cluster
    avg_face_quality DECIMAL(5, 2),
    
    -- User metadata
    notes TEXT,
    relationship VARCHAR(50),  -- family, friend, colleague, etc.
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Soft delete
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_clusters_name ON clusters(name);
CREATE INDEX idx_clusters_is_deleted ON clusters(is_deleted) WHERE is_deleted = FALSE;

-- Face embeddings table (links faces to photos and clusters)
CREATE TABLE face_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    photo_id UUID NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
    cluster_id UUID REFERENCES clusters(cluster_id) ON DELETE SET NULL,
    
    -- Face detection data
    face_location JSONB NOT NULL,  -- {top, right, bottom, left}
    face_landmarks JSONB,  -- Detailed face landmarks
    face_encoding vector(128) NOT NULL,  -- 128-d face embedding
    
    -- Face quality assessment
    quality_score DECIMAL(5, 2),  -- Overall face quality 0-100
    sharpness DECIMAL(5, 2),
    brightness DECIMAL(5, 2),
    face_size INTEGER,  -- Pixels
    angle_score DECIMAL(5, 2),  -- Frontal = 100, profile = 0
    
    -- Emotion detection
    emotion VARCHAR(20),  -- happy, sad, angry, surprise, fear, disgust, neutral
    emotion_confidence DECIMAL(5, 2),
    emotion_scores JSONB,  -- All emotion probabilities
    
    -- Age/gender (optional, can add later)
    -- estimated_age INTEGER,
    -- estimated_gender VARCHAR(10),
    
    -- Timestamps
    detected_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_quality CHECK (quality_score BETWEEN 0 AND 100),
    CONSTRAINT valid_emotion_confidence CHECK (emotion_confidence BETWEEN 0 AND 100)
);

CREATE INDEX idx_face_embeddings_photo ON face_embeddings(photo_id);
CREATE INDEX idx_face_embeddings_cluster ON face_embeddings(cluster_id);
CREATE INDEX idx_face_embeddings_emotion ON face_embeddings(emotion);
CREATE INDEX idx_face_embeddings_quality ON face_embeddings(quality_score);

-- Vector index for face embeddings (for re-clustering)
-- CREATE INDEX idx_face_embeddings_encoding ON face_embeddings 
-- USING ivfflat (face_encoding vector_cosine_ops) WITH (lists = 50);


-- MANY-TO-MANY: PHOTOS <-> CLUSTERS


-- Junction table (a photo can have multiple people)
CREATE TABLE photo_clusters (
    photo_id UUID NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
    cluster_id UUID NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    
    -- Additional metadata
    face_count_in_photo INTEGER DEFAULT 1,  -- How many faces of this person
    is_primary_subject BOOLEAN DEFAULT FALSE,  -- Is this person the main subject?
    added_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (photo_id, cluster_id)
);

CREATE INDEX idx_photo_clusters_photo ON photo_clusters(photo_id);
CREATE INDEX idx_photo_clusters_cluster ON photo_clusters(cluster_id);


-- OBJECT DETECTION (YOLO)


-- Detected objects table
CREATE TABLE detected_objects (
    object_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    photo_id UUID NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
    
    -- Object detection data
    label VARCHAR(100) NOT NULL,  -- COCO class: person, car, dog, etc.
    confidence DECIMAL(5, 2),
    bounding_box JSONB NOT NULL,  -- {x, y, width, height}
    
    -- Object attributes
    dominant_color VARCHAR(50),  -- Extracted from bbox region
    color_hex VARCHAR(7),  -- #RRGGBB
    size_category VARCHAR(20),  -- small, medium, large
    
    -- Timestamps
    detected_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_confidence CHECK (confidence BETWEEN 0 AND 100)
);

CREATE INDEX idx_detected_objects_photo ON detected_objects(photo_id);
CREATE INDEX idx_detected_objects_label ON detected_objects(label);
CREATE INDEX idx_detected_objects_color ON detected_objects(dominant_color);


-- EMOTIONS (AGGREGATED PER PHOTO)


-- Aggregated emotion data per photo
CREATE TABLE photo_emotions (
    photo_id UUID PRIMARY KEY REFERENCES photos(photo_id) ON DELETE CASCADE,
    
    -- Emotion counts
    happy_count INTEGER DEFAULT 0,
    sad_count INTEGER DEFAULT 0,
    angry_count INTEGER DEFAULT 0,
    surprise_count INTEGER DEFAULT 0,
    fear_count INTEGER DEFAULT 0,
    disgust_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    
    -- Dominant emotion
    dominant_emotion VARCHAR(20),
    dominant_emotion_percentage DECIMAL(5, 2),
    
    -- Overall mood
    overall_mood VARCHAR(20),  -- positive, negative, neutral, mixed
    mood_score DECIMAL(5, 2),  -- -1 to +1
    
    -- Timestamps
    analyzed_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_mood_score CHECK (mood_score BETWEEN -1 AND 1)
);

CREATE INDEX idx_photo_emotions_dominant ON photo_emotions(dominant_emotion);
CREATE INDEX idx_photo_emotions_mood ON photo_emotions(overall_mood);


-- CONVERSATIONS & MESSAGES (RAG / CHAT)


-- Conversations table (chat sessions)
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Conversation metadata
    title TEXT,  -- Auto-generated from first query
    summary TEXT,  -- Summarized conversation for context compression
    
    -- Timestamps
    started_at TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP DEFAULT NOW(),
    
    -- Conversation state
    is_active BOOLEAN DEFAULT TRUE,
    message_count INTEGER DEFAULT 0,
    
    -- User metadata (if multi-user later)
    -- user_id UUID,
    
    CONSTRAINT valid_message_count CHECK (message_count >= 0)
);

CREATE INDEX idx_conversations_started_at ON conversations(started_at);
CREATE INDEX idx_conversations_active ON conversations(is_active) WHERE is_active = TRUE;

-- Messages table (user queries + AI responses)
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    
    -- Message content
    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    
    -- Query metadata (for user messages)
    query_embedding vector(512),  -- Embedding of user query
    parsed_entities JSONB,  -- Extracted entities (people, dates, etc.)
    
    -- Retrieved context (for assistant messages)
    retrieved_photo_ids UUID[],  -- Array of photo IDs used in response
    retrieval_scores DECIMAL(5, 2)[],  -- Similarity scores
    context_used TEXT,  -- The actual context sent to LLM
    
    -- Generation metadata (for assistant messages)
    model_name VARCHAR(50),  -- llama-3.3, etc.
    tokens_used INTEGER,
    generation_time_ms INTEGER,
    
    -- Feedback
    thumbs_up BOOLEAN,
    thumbs_down BOOLEAN,
    user_feedback TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant'))
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_retrieved_photos ON messages USING gin(retrieved_photo_ids);

-- Vector index for query embeddings (semantic search over past queries)
-- CREATE INDEX idx_messages_query_embedding ON messages 
-- USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 50);


-- SEARCH INDEX (OPTIMIZED FOR RETRIEVAL)


-- Materialized view for fast search
CREATE MATERIALIZED VIEW search_index AS
SELECT 
    p.photo_id,
    p.taken_date,
    p.season,
    p.time_of_day,
    p.scene_type,
    p.activity,
    p.clip_embedding,
    p.searchable_text,
    
    -- Aggregated people names
    array_agg(DISTINCT c.name) FILTER (WHERE c.name IS NOT NULL) AS people_names,
    
    -- Aggregated emotions
    array_agg(DISTINCT fe.emotion) FILTER (WHERE fe.emotion IS NOT NULL) AS emotions,
    
    -- Aggregated objects
    array_agg(DISTINCT do.label) FILTER (WHERE do.label IS NOT NULL) AS objects,
    array_agg(DISTINCT do.dominant_color) FILTER (WHERE do.dominant_color IS NOT NULL) AS colors,
    
    -- Photo metadata
    pe.dominant_emotion,
    pe.mood_score,
    p.overall_quality
    
FROM photos p
LEFT JOIN photo_clusters pc ON p.photo_id = pc.photo_id
LEFT JOIN clusters c ON pc.cluster_id = c.cluster_id
LEFT JOIN face_embeddings fe ON p.photo_id = fe.photo_id
LEFT JOIN detected_objects do ON p.photo_id = do.photo_id
LEFT JOIN photo_emotions pe ON p.photo_id = pe.photo_id

WHERE p.is_deleted = FALSE AND p.is_processed = TRUE
GROUP BY 
    p.photo_id, pe.dominant_emotion, pe.mood_score;

-- Indexes on materialized view
CREATE INDEX idx_search_people ON search_index USING gin(people_names);
CREATE INDEX idx_search_emotions ON search_index USING gin(emotions);
CREATE INDEX idx_search_objects ON search_index USING gin(objects);
CREATE INDEX idx_search_colors ON search_index USING gin(colors);
CREATE INDEX idx_search_date ON search_index(taken_date);
CREATE INDEX idx_search_season ON search_index(season);

-- Vector index on search_index
-- CREATE INDEX idx_search_clip_embedding ON search_index 
-- USING ivfflat (clip_embedding vector_cosine_ops) WITH (lists = 100);

-- Refresh command (run after adding new photos)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY search_index;


-- UTILITY TABLES


-- System settings
CREATE TABLE system_settings (
    setting_key VARCHAR(100) PRIMARY KEY,
    setting_value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default settings
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
('clustering_eps', '0.6', 'DBSCAN epsilon for face clustering'),
('clustering_min_samples', '1', 'DBSCAN min_samples'),
('emotion_threshold', '0.7', 'Minimum confidence for emotion detection'),
('object_confidence_threshold', '0.5', 'Minimum confidence for object detection'),
('clip_model', 'ViT-B/32', 'CLIP model version'),
('llm_model', 'llama-3.3', 'LLM model for generation');

-- Processing logs
CREATE TABLE processing_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    photo_id UUID REFERENCES photos(photo_id) ON DELETE CASCADE,
    stage VARCHAR(50),  -- upload, face_detection, emotion, objects, clip, etc.
    status VARCHAR(20),  -- success, error, warning
    message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_processing_logs_photo ON processing_logs(photo_id);
CREATE INDEX idx_processing_logs_stage ON processing_logs(stage);
CREATE INDEX idx_processing_logs_status ON processing_logs(status);


-- TRIGGERS & FUNCTIONS


-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_clusters_updated_at 
BEFORE UPDATE ON clusters
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update cluster counts
CREATE OR REPLACE FUNCTION update_cluster_counts()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE clusters SET
        face_count = (
            SELECT COUNT(*) FROM face_embeddings 
            WHERE cluster_id = NEW.cluster_id
        ),
        photo_count = (
            SELECT COUNT(DISTINCT photo_id) FROM photo_clusters 
            WHERE cluster_id = NEW.cluster_id
        )
    WHERE cluster_id = NEW.cluster_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_cluster_counts_on_face_insert
AFTER INSERT ON face_embeddings
FOR EACH ROW EXECUTE FUNCTION update_cluster_counts();

-- Update conversation message count
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversations SET
        message_count = (
            SELECT COUNT(*) FROM messages 
            WHERE conversation_id = NEW.conversation_id
        ),
        last_message_at = NEW.created_at
    WHERE conversation_id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_conversation_on_message_insert
AFTER INSERT ON messages
FOR EACH ROW EXECUTE FUNCTION update_conversation_message_count();



-- VIEWS FOR COMMON QUERIES


-- View: Photos with all metadata
CREATE VIEW photo_details AS
SELECT 
    p.*,
    array_agg(DISTINCT c.name) FILTER (WHERE c.name IS NOT NULL) AS people,
    array_agg(DISTINCT fe.emotion) FILTER (WHERE fe.emotion IS NOT NULL) AS emotions,
    array_agg(DISTINCT do.label) FILTER (WHERE do.label IS NOT NULL) AS objects,
    COUNT(DISTINCT fe.embedding_id) AS face_count
FROM photos p
LEFT JOIN photo_clusters pc ON p.photo_id = pc.photo_id
LEFT JOIN clusters c ON pc.cluster_id = c.cluster_id
LEFT JOIN face_embeddings fe ON p.photo_id = fe.photo_id
LEFT JOIN detected_objects do ON p.photo_id = do.photo_id
GROUP BY p.photo_id;

-- View: Cluster statistics
CREATE VIEW cluster_stats AS
SELECT 
    c.*,
    COUNT(DISTINCT pc.photo_id) AS photo_count,
    COUNT(DISTINCT fe.embedding_id) AS face_count,
    AVG(fe.quality_score) AS avg_face_quality,
    mode() WITHIN GROUP (ORDER BY fe.emotion) AS common_emotion
FROM clusters c
LEFT JOIN photo_clusters pc ON c.cluster_id = pc.cluster_id
LEFT JOIN face_embeddings fe ON c.cluster_id = fe.cluster_id
GROUP BY c.cluster_id;


-- SAMPLE QUERIES (FOR REFERENCE)


-- Find photos with a person
-- SELECT * FROM photo_details WHERE 'Mom' = ANY(people);

-- Semantic search for "beach sunset"
-- SELECT photo_id, clip_embedding <=> '[...]'::vector AS distance
-- FROM photos
-- ORDER BY distance
-- LIMIT 10;

-- Photos with happy people
-- SELECT DISTINCT p.* FROM photos p
-- JOIN face_embeddings fe ON p.photo_id = fe.photo_id
-- WHERE fe.emotion = 'happy';

-- Photos from last summer
-- SELECT * FROM photos 
-- WHERE season = 'summer' 
--   AND year = 2023
-- ORDER BY taken_date;

-- Top 10 most frequently appearing people
-- SELECT c.name, COUNT(*) as appearance_count
-- FROM photo_clusters pc
-- JOIN clusters c ON pc.cluster_id = c.cluster_id
-- GROUP BY c.name
-- ORDER BY appearance_count DESC
-- LIMIT 10;


-- SCHEMA DOCUMENTATION


COMMENT ON TABLE photos IS 'Core photo storage with AI-extracted metadata';
COMMENT ON TABLE clusters IS 'Face clusters representing unique people';
COMMENT ON TABLE face_embeddings IS 'Individual detected faces with embeddings';
COMMENT ON TABLE detected_objects IS 'YOLO-detected objects in photos';
COMMENT ON TABLE photo_emotions IS 'Aggregated emotion analysis per photo';
COMMENT ON TABLE conversations IS 'Chat conversation sessions';
COMMENT ON TABLE messages IS 'Individual messages in conversations';
COMMENT ON TABLE search_index IS 'Optimized materialized view for fast retrieval';

COMMENT ON COLUMN photos.clip_embedding IS 'CLIP ViT-B/32 512-dimensional embedding';
COMMENT ON COLUMN photos.searchable_text IS 'Combined text for full-text search';
COMMENT ON COLUMN face_embeddings.face_encoding IS 'face_recognition 128-d encoding';
COMMENT ON COLUMN messages.query_embedding IS 'Semantic embedding of user query';