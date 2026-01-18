"""
SQLAlchemy Models for Lumeo - Phase 5 Enhanced
Added conversation memory and summarization fields
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


# ============================================================================
# PHASE 1-2: PHOTO & VISION MODELS (Unchanged)
# ============================================================================

class Photo(Base):
    """Photo model with enhanced metadata"""
    __tablename__ = 'photos'
    
    photo_id = Column(String(255), primary_key=True)
    filename = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)
    upload_date = Column(Float, nullable=False)
    
    # Vision Intelligence
    clip_embedding = Column(Vector(512))
    scene_type = Column(String(20))
    location_type = Column(String(50))
    activity = Column(String(50))
    
    # Temporal Context
    season = Column(String(20))
    time_of_day = Column(String(20))
    date_taken = Column(DateTime)
    
    # Camera Metadata
    camera_make = Column(String(100))
    camera_model = Column(String(100))
    
    # GPS
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    
    # Image Quality
    image_quality = Column(Float)
    
    # AI-Generated Content
    caption = Column(Text)
    
    # Emotion Analysis
    dominant_emotion = Column(String(20))
    mood_score = Column(Float)
    
    # Relationships
    face_embeddings = relationship('FaceEmbedding', back_populates='photo', cascade='all, delete-orphan')
    photo_clusters = relationship('PhotoCluster', back_populates='photo', cascade='all, delete-orphan')
    detected_objects = relationship('DetectedObject', back_populates='photo', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Photo(id={self.photo_id}, filename={self.filename})>"


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
        return f"<Cluster(id={self.cluster_id}, name={self.name})>"


class FaceEmbedding(Base):
    """Face embedding with emotion and quality"""
    __tablename__ = 'face_embeddings'
    
    embedding_id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(String(255), ForeignKey('photos.photo_id', ondelete='CASCADE'))
    cluster_id = Column(String(255), ForeignKey('clusters.cluster_id', ondelete='CASCADE'))
    embedding = Column(LargeBinary, nullable=False)
    face_location = Column(Text)
    
    # Emotion Analysis
    emotion = Column(String(20))
    emotion_confidence = Column(Float)
    emotion_valence = Column(Float)
    
    # Quality Assessment
    quality_score = Column(Float)
    
    # Relationships
    photo = relationship('Photo', back_populates='face_embeddings')
    cluster = relationship('Cluster', back_populates='face_embeddings')
    
    def __repr__(self):
        return f"<FaceEmbedding(id={self.embedding_id}, emotion={self.emotion})>"


class PhotoCluster(Base):
    """Junction table for photo-cluster relationship"""
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
    
    label = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Bounding Box
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Color Information
    dominant_color_rgb = Column(String(50))
    color_name = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    photo = relationship('Photo', back_populates='detected_objects')
    
    def __repr__(self):
        return f"<DetectedObject(id={self.object_id}, label={self.label})>"


# ============================================================================
# CONVERSATION & MEMORY MODELS
# ============================================================================

class Conversation(Base):
    """
    Conversation/chat session with memory features
    
    Phase 5 Enhancements:
    - needs_summary: Flag for auto-summarization
    - last_summarized_at: Timestamp of last summary
    """
    __tablename__ = 'conversations'
    
    conversation_id = Column(String(255), primary_key=True, default=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    user_id = Column(String(255), default="default_user")
    
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)
    
    # Conversation metadata
    message_count = Column(Integer, default=0)
    summary = Column(Text)  # Auto-generated summary
    
    # Phase 5: Memory Management ✨
    needs_summary = Column(Boolean, default=False)  # Flag for auto-summarization
    last_summarized_at = Column(Float)  # When was last summarized
    
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
    content = Column(Text, nullable=False)
    
    # Retrieved context for this message
    retrieved_photo_ids = Column(Text)  # JSON list of photo IDs
    meta_data = Column(JSON)  # Additional data
    
    created_at = Column(Float, nullable=False)
    
    # Relationship
    conversation = relationship('Conversation', back_populates='messages')
    
    def __repr__(self):
        return f"<Message(id={self.message_id}, role={self.role})>"


# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(engine)
    print("✓ Database tables initialized (Phase 5)")


# Migration script for Phase 5
def migrate_to_phase5():
    """
    Add Phase 5 columns to existing Conversation table
    
    Run this if upgrading from Phase 4
    """
    from sqlalchemy import text
    
    try:
        with engine.connect() as conn:
            # Add new columns if they don't exist
            try:
                conn.execute(text(
                    "ALTER TABLE conversations ADD COLUMN needs_summary BOOLEAN DEFAULT FALSE"
                ))
                print("✓ Added needs_summary column")
            except:
                print("  needs_summary column already exists")
            
            try:
                conn.execute(text(
                    "ALTER TABLE conversations ADD COLUMN last_summarized_at FLOAT"
                ))
                print("✓ Added last_summarized_at column")
            except:
                print("  last_summarized_at column already exists")
            
            conn.commit()
        
        print("✓ Phase 5 migration complete")
        
    except Exception as e:
        print(f"✗ Migration error: {e}")
        print("  This is normal if columns already exist")


if __name__ == '__main__':
    # Test database connection
    try:
        print(f"Connecting to: {DATABASE_URL.replace(DB_PASSWORD, '***')}")
        
        # Initialize tables
        init_db()
        
        # Run migration (safe to run multiple times)
        migrate_to_phase5()
        
        # Test query
        session = Session()
        photo_count = session.query(Photo).count()
        cluster_count = session.query(Cluster).count()
        conversation_count = session.query(Conversation).count()
        message_count = session.query(Message).count()
        
        print(f"\n✓ Database connected successfully")
        print(f"  - Photos: {photo_count}")
        print(f"  - Clusters: {cluster_count}")
        print(f"  - Conversations: {conversation_count}")
        print(f"  - Messages: {message_count}")
        
        session.close()
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")