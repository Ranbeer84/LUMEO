"""
SQLAlchemy Models for Lumeo - Phase 4 Enhanced
Includes conversation and message tables for chat interface
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, ForeignKey, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSON
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

# Database configuration
DB_NAME = os.getenv('DB_NAME', 'lumeo_db')
DB_USER = os.getenv('DB_USER', 'lumeo_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'lumeo_password')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Photo(Base):
    """Photo model with enhanced metadata"""
    __tablename__ = 'photos'
    
    # Original fields (Phase 1)
    photo_id = Column(String(255), primary_key=True)
    filename = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)
    upload_date = Column(Float, nullable=False)
    
    # Phase 2: Vision Intelligence
    clip_embedding = Column(Vector(512))  # CLIP embedding for semantic search
    scene_type = Column(String(20))  # indoor/outdoor
    location_type = Column(String(50))  # beach, office, home, etc.
    activity = Column(String(50))  # sports, dining, party, etc.
    
    # Phase 2: Temporal Context
    season = Column(String(20))  # winter, spring, summer, autumn
    time_of_day = Column(String(20))  # morning, afternoon, evening, night
    date_taken = Column(DateTime)  # From EXIF
    
    # Phase 2: Camera Metadata
    camera_make = Column(String(100))
    camera_model = Column(String(100))
    
    # Phase 2: GPS
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    
    # Phase 2: Image Quality
    image_quality = Column(Float)  # 0-1 quality score
    
    # Phase 2: AI-Generated Content
    caption = Column(Text)  # Auto-generated natural language caption
    
    # Phase 2: Emotion Analysis
    dominant_emotion = Column(String(20))  # Overall photo emotion
    mood_score = Column(Float)  # -1 (negative) to +1 (positive)
    
    # Relationships
    face_embeddings = relationship('FaceEmbedding', back_populates='photo', cascade='all, delete-orphan')
    photo_clusters = relationship('PhotoCluster', back_populates='photo', cascade='all, delete-orphan')
    detected_objects = relationship('DetectedObject', back_populates='photo', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Photo(id={self.photo_id}, filename={self.filename}, scene={self.scene_type})>"


class Cluster(Base):
    """Cluster/Person model"""
    __tablename__ = 'clusters'
    
    cluster_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    face_count = Column(Integer, default=0)
    thumbnail = Column(String(255))
    created_at = Column(Float, nullable=False)
    
    # Relationships
    face_embeddings = relationship('FaceEmbedding', back_populates='cluster', cascade='all, delete-orphan')
    photo_clusters = relationship('PhotoCluster', back_populates='cluster', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Cluster(id={self.cluster_id}, name={self.name}, faces={self.face_count})>"


class FaceEmbedding(Base):
    """Face embedding with emotion and quality"""
    __tablename__ = 'face_embeddings'
    
    embedding_id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(String(255), ForeignKey('photos.photo_id', ondelete='CASCADE'))
    cluster_id = Column(String(255), ForeignKey('clusters.cluster_id', ondelete='CASCADE'))
    embedding = Column(LargeBinary, nullable=False)  # Face encoding
    face_location = Column(Text)  # JSON string of (top, right, bottom, left)
    
    # Phase 2: Emotion Analysis
    emotion = Column(String(20))  # happy, sad, angry, surprise, neutral, fear, disgust
    emotion_confidence = Column(Float)  # 0-1 confidence score
    emotion_valence = Column(Float)  # -1 (negative) to +1 (positive)
    
    # Phase 2: Quality Assessment
    quality_score = Column(Float)  # 0-1 face quality score
    
    # Relationships
    photo = relationship('Photo', back_populates='face_embeddings')
    cluster = relationship('Cluster', back_populates='face_embeddings')
    
    def __repr__(self):
        return f"<FaceEmbedding(id={self.embedding_id}, emotion={self.emotion}, quality={self.quality_score})>"


class PhotoCluster(Base):
    """Junction table for many-to-many photo-cluster relationship"""
    __tablename__ = 'photo_clusters'
    
    photo_id = Column(String(255), ForeignKey('photos.photo_id', ondelete='CASCADE'), primary_key=True)
    cluster_id = Column(String(255), ForeignKey('clusters.cluster_id', ondelete='CASCADE'), primary_key=True)
    
    # Relationships
    photo = relationship('Photo', back_populates='photo_clusters')
    cluster = relationship('Cluster', back_populates='photo_clusters')
    
    def __repr__(self):
        return f"<PhotoCluster(photo={self.photo_id}, cluster={self.cluster_id})>"


class DetectedObject(Base):
    """Detected objects from YOLO"""
    __tablename__ = 'detected_objects'
    
    object_id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(String(255), ForeignKey('photos.photo_id', ondelete='CASCADE'), nullable=False)
    
    # Object Detection Data
    label = Column(String(100), nullable=False)  # car, person, cake, etc.
    confidence = Column(Float, nullable=False)  # 0-1 confidence score
    
    # Bounding Box
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Color Information
    dominant_color_rgb = Column(String(50))  # "(255, 128, 64)"
    color_name = Column(String(50))  # "red", "blue", etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    photo = relationship('Photo', back_populates='detected_objects')
    
    def __repr__(self):
        return f"<DetectedObject(id={self.object_id}, label={self.label}, confidence={self.confidence:.2f})>"


# ============================================================================
# PHASE 4: CONVERSATION TABLES
# ============================================================================

class Conversation(Base):
    """Conversation/chat session"""
    __tablename__ = 'conversations'
    
    conversation_id = Column(String(255), primary_key=True, default=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    user_id = Column(String(255), default="default_user")  # For multi-user support later
    
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)
    
    # Conversation metadata
    message_count = Column(Integer, default=0)
    summary = Column(Text)  # Auto-generated summary of conversation
    
    # Relationships
    messages = relationship('Message', back_populates='conversation', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Conversation(id={self.conversation_id}, messages={self.message_count})>"


class Message(Base):
    """Individual message in a conversation"""
    __tablename__ = 'messages'
    
    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(255), ForeignKey('conversations.conversation_id', ondelete='CASCADE'), nullable=False)
    
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)  # Message text
    
    # Retrieved context for this message
    retrieved_photo_ids = Column(Text)  # JSON list of photo IDs retrieved
    meta_data = Column(JSON)  # Additional data (similarity scores, filters used, etc.)
    
    created_at = Column(Float, nullable=False)
    
    # Relationship
    conversation = relationship('Conversation', back_populates='messages')
    
    def __repr__(self):
        return f"<Message(id={self.message_id}, role={self.role}, conv={self.conversation_id})>"


# Create all tables (if they don't exist)
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(engine)
    print("✓ Database tables initialized")


if __name__ == '__main__':
    # Test database connection
    try:
        print(f"Connecting to: {DATABASE_URL.replace(DB_PASSWORD, '***')}")
        init_db()
        
        # Test query
        session = Session()
        photo_count = session.query(Photo).count()
        cluster_count = session.query(Cluster).count()
        object_count = session.query(DetectedObject).count()
        conversation_count = session.query(Conversation).count()
        message_count = session.query(Message).count()
        
        print(f"✓ Database connected successfully")
        print(f"  - Photos: {photo_count}")
        print(f"  - Clusters: {cluster_count}")
        print(f"  - Objects: {object_count}")
        print(f"  - Conversations: {conversation_count}")
        print(f"  - Messages: {message_count}")
        
        session.close()
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")