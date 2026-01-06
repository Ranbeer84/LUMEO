"""
SQLAlchemy ORM Models for Lumeo
Replaces direct SQLite database access with ORM
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, ARRAY, ForeignKey, CheckConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime


# DATABASE CONFIGURATION


# Database URL (from environment variable or direct)
import os
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://lumeo_user:lumeo_123@localhost:5432/lumeo_db')

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Verify connections before use
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ORM MODELS


class Photo(Base):
    """Photos table - core photo storage"""
    __tablename__ = 'photos'
    
    # Primary key
    photo_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer)
    file_hash = Column(String(64))
    
    # Timestamps
    upload_date = Column(DateTime, default=datetime.utcnow)
    taken_date = Column(DateTime)
    processed_date = Column(DateTime)
    
    # Image metadata
    width = Column(Integer)
    height = Column(Integer)
    format = Column(String(10))
    camera_make = Column(String(100))
    camera_model = Column(String(100))
    
    # Location
    latitude = Column(Float)
    longitude = Column(Float)
    location_name = Column(Text)
    
    # Temporal context
    season = Column(String(20))
    time_of_day = Column(String(20))
    year = Column(Integer)
    month = Column(Integer)
    day_of_week = Column(String(10))
    
    # Scene classification
    scene_type = Column(String(50))
    activity = Column(String(50))
    is_indoor = Column(Boolean)
    
    # Quality metrics
    blur_score = Column(Float)
    brightness_score = Column(Float)
    overall_quality = Column(Float)
    
    # Emotions
    dominant_emotion = Column(String(20))
    emotion_confidence = Column(Float)
    overall_mood_score = Column(Float)
    
    # CLIP embeddings
    clip_embedding = Column(Vector(512))
    
    # Captions
    generated_caption = Column(Text)
    user_caption = Column(Text)
    searchable_text = Column(Text)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text)
    
    # Soft delete
    is_deleted = Column(Boolean, default=False)
    deleted_date = Column(DateTime)
    
    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="photo", cascade="all, delete-orphan")
    detected_objects = relationship("DetectedObject", back_populates="photo", cascade="all, delete-orphan")
    photo_emotion = relationship("PhotoEmotion", back_populates="photo", uselist=False)
    clusters = relationship("Cluster", secondary="photo_clusters", back_populates="photos")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('overall_mood_score >= -1 AND overall_mood_score <= 1', name='valid_mood_score'),
        CheckConstraint('overall_quality >= 0 AND overall_quality <= 100', name='valid_quality'),
        Index('idx_photos_upload_date', 'upload_date'),
        Index('idx_photos_taken_date', 'taken_date'),
        Index('idx_photos_season', 'season'),
        Index('idx_photos_scene_type', 'scene_type'),
        Index('idx_photos_is_processed', 'is_processed'),
    )
    
    def __repr__(self):
        return f"<Photo(id={self.photo_id}, filename={self.filename})>"


class Cluster(Base):
    """Clusters table - represents unique people"""
    __tablename__ = 'clusters'
    
    # Primary key
    cluster_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Cluster info
    name = Column(String(100), nullable=False, default='Unknown Person')
    face_count = Column(Integer, default=0)
    photo_count = Column(Integer, default=0)
    
    # Representative face
    thumbnail_path = Column(Text)
    representative_embedding = Column(Vector(128))
    
    # Quality
    avg_face_quality = Column(Float)
    
    # User metadata
    notes = Column(Text)
    relationship_type = Column(String(50))  # family, friend, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    is_deleted = Column(Boolean, default=False)
    
    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="cluster")
    photos = relationship("Photo", secondary="photo_clusters", back_populates="clusters")
    
    __table_args__ = (
        Index('idx_clusters_name', 'name'),
    )
    
    def __repr__(self):
        return f"<Cluster(id={self.cluster_id}, name={self.name}, faces={self.face_count})>"


class FaceEmbedding(Base):
    """Face embeddings table - individual detected faces"""
    __tablename__ = 'face_embeddings'
    
    # Primary key
    embedding_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    photo_id = Column(UUID(as_uuid=True), ForeignKey('photos.photo_id', ondelete='CASCADE'), nullable=False)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey('clusters.cluster_id', ondelete='SET NULL'))
    
    # Face data
    face_location = Column(JSONB, nullable=False)
    face_landmarks = Column(JSONB)
    face_encoding = Column(Vector(128), nullable=False)
    
    # Quality metrics
    quality_score = Column(Float)
    sharpness = Column(Float)
    brightness = Column(Float)
    face_size = Column(Integer)
    angle_score = Column(Float)
    
    # Emotion
    emotion = Column(String(20))
    emotion_confidence = Column(Float)
    emotion_scores = Column(JSONB)
    
    # Timestamp
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    photo = relationship("Photo", back_populates="face_embeddings")
    cluster = relationship("Cluster", back_populates="face_embeddings")
    
    __table_args__ = (
        CheckConstraint('quality_score >= 0 AND quality_score <= 100', name='valid_quality'),
        CheckConstraint('emotion_confidence >= 0 AND emotion_confidence <= 100', name='valid_emotion_confidence'),
        Index('idx_face_embeddings_photo', 'photo_id'),
        Index('idx_face_embeddings_cluster', 'cluster_id'),
        Index('idx_face_embeddings_emotion', 'emotion'),
    )
    
    def __repr__(self):
        return f"<FaceEmbedding(id={self.embedding_id}, emotion={self.emotion})>"


class PhotoCluster(Base):
    """Junction table for many-to-many Photo-Cluster relationship"""
    __tablename__ = 'photo_clusters'
    
    # Composite primary key
    photo_id = Column(UUID(as_uuid=True), ForeignKey('photos.photo_id', ondelete='CASCADE'), primary_key=True)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey('clusters.cluster_id', ondelete='CASCADE'), primary_key=True)
    
    # Metadata
    face_count_in_photo = Column(Integer, default=1)
    is_primary_subject = Column(Boolean, default=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_photo_clusters_photo', 'photo_id'),
        Index('idx_photo_clusters_cluster', 'cluster_id'),
    )


class DetectedObject(Base):
    """Detected objects table - YOLO results"""
    __tablename__ = 'detected_objects'
    
    # Primary key
    object_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key
    photo_id = Column(UUID(as_uuid=True), ForeignKey('photos.photo_id', ondelete='CASCADE'), nullable=False)
    
    # Object data
    label = Column(String(100), nullable=False)
    confidence = Column(Float)
    bounding_box = Column(JSONB, nullable=False)
    
    # Attributes
    dominant_color = Column(String(50))
    color_hex = Column(String(7))
    size_category = Column(String(20))
    
    # Timestamp
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    photo = relationship("Photo", back_populates="detected_objects")
    
    __table_args__ = (
        CheckConstraint('confidence >= 0 AND confidence <= 100', name='valid_confidence'),
        Index('idx_detected_objects_photo', 'photo_id'),
        Index('idx_detected_objects_label', 'label'),
    )
    
    def __repr__(self):
        return f"<DetectedObject(id={self.object_id}, label={self.label})>"


class PhotoEmotion(Base):
    """Aggregated emotion data per photo"""
    __tablename__ = 'photo_emotions'
    
    # Primary key (one-to-one with photos)
    photo_id = Column(UUID(as_uuid=True), ForeignKey('photos.photo_id', ondelete='CASCADE'), primary_key=True)
    
    # Emotion counts
    happy_count = Column(Integer, default=0)
    sad_count = Column(Integer, default=0)
    angry_count = Column(Integer, default=0)
    surprise_count = Column(Integer, default=0)
    fear_count = Column(Integer, default=0)
    disgust_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    
    # Dominant emotion
    dominant_emotion = Column(String(20))
    dominant_emotion_percentage = Column(Float)
    
    # Overall mood
    overall_mood = Column(String(20))
    mood_score = Column(Float)
    
    # Timestamp
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    photo = relationship("Photo", back_populates="photo_emotion")
    
    __table_args__ = (
        CheckConstraint('mood_score >= -1 AND mood_score <= 1', name='valid_mood_score'),
        Index('idx_photo_emotions_dominant', 'dominant_emotion'),
    )


class Conversation(Base):
    """Conversations table - chat sessions"""
    __tablename__ = 'conversations'
    
    # Primary key
    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Conversation metadata
    title = Column(Text)
    summary = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    
    # State
    is_active = Column(Boolean, default=True)
    message_count = Column(Integer, default=0)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint('message_count >= 0', name='valid_message_count'),
        Index('idx_conversations_started_at', 'started_at'),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.conversation_id}, messages={self.message_count})>"


class Message(Base):
    """Messages table - user queries and AI responses"""
    __tablename__ = 'messages'
    
    # Primary key
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.conversation_id', ondelete='CASCADE'), nullable=False)
    
    # Message content
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    
    # Query metadata
    query_embedding = Column(Vector(512))
    parsed_entities = Column(JSONB)
    
    # Retrieved context
    retrieved_photo_ids = Column(ARRAY(UUID(as_uuid=True)))
    retrieval_scores = Column(ARRAY(Float))
    context_used = Column(Text)
    
    # Generation metadata
    model_name = Column(String(50))
    tokens_used = Column(Integer)
    generation_time_ms = Column(Integer)
    
    # Feedback
    thumbs_up = Column(Boolean)
    thumbs_down = Column(Boolean)
    user_feedback = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name='valid_role'),
        Index('idx_messages_conversation', 'conversation_id'),
        Index('idx_messages_role', 'role'),
    )
    
    def __repr__(self):
        return f"<Message(id={self.message_id}, role={self.role})>"



# HELPER FUNCTIONS


def get_db():
    """Get database session (for use in Flask routes)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database (create all tables)"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")

def drop_all_tables():
    """Drop all tables (WARNING: DESTRUCTIVE)"""
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped!")


# EXAMPLE USAGE


if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Example: Create a photo
    session = SessionLocal()
    
    new_photo = Photo(
        filename="test.jpg",
        file_path="/uploads/test.jpg",
        is_processed=False
    )
    
    session.add(new_photo)
    session.commit()
    
    print(f"Created photo: {new_photo}")
    
    # Example: Query photos
    photos = session.query(Photo).filter_by(is_processed=False).all()
    print(f"Unprocessed photos: {len(photos)}")
    
    session.close()