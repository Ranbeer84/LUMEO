from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, Text,
    ForeignKey, LargeBinary, Boolean, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSON
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

DB_NAME     = os.getenv('DB_NAME',     'lumeo_db')
DB_USER     = os.getenv('DB_USER',     'lumeo_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'lumeo_password')
DB_HOST     = os.getenv('DB_HOST',     'localhost')
DB_PORT     = os.getenv('DB_PORT',     '5432')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

engine  = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base    = declarative_base()


# ============================================================================
# PHOTO & VISION MODELS
# ============================================================================

class Photo(Base):
    __tablename__ = 'photos'

    photo_id    = Column(String(255), primary_key=True)
    filename    = Column(String(255), nullable=False)
    path        = Column(String(500), nullable=False)
    upload_date = Column(Float,       nullable=False)

    clip_embedding = Column(Vector(512))
    scene_type     = Column(String(100), nullable=True)
    location_type  = Column(String(50))
    activity       = Column(String(50))
    weather        = Column(String(50), nullable=True)

    season      = Column(String(20))
    time_of_day = Column(String(20))
    date_taken  = Column(DateTime)

    camera_make  = Column(String(100))
    camera_model = Column(String(100))

    gps_latitude  = Column(Float)
    gps_longitude = Column(Float)

    image_quality = Column(Float)
    caption       = Column(Text)

    dominant_emotion = Column(String(20))
    mood_score       = Column(Float)
    face_count       = Column(Integer, default=0)

    face_embeddings      = relationship('FaceEmbedding',      back_populates='photo', cascade='all, delete-orphan')
    photo_clusters       = relationship('PhotoCluster',        back_populates='photo', cascade='all, delete-orphan')
    detected_objects     = relationship('DetectedObject',      back_populates='photo', cascade='all, delete-orphan')
    object_cluster_links = relationship('PhotoObjectCluster',  back_populates='photo', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Photo(id={self.photo_id}, filename={self.filename})>"


class Cluster(Base):
    __tablename__ = 'clusters'

    cluster_id  = Column(String(255), primary_key=True)
    name        = Column(String(255), nullable=False)
    face_count  = Column(Integer, default=0)
    photo_count = Column(Integer, default=0)
    thumbnail   = Column(String(255))

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    face_embeddings = relationship('FaceEmbedding', back_populates='cluster', cascade='all, delete-orphan')
    photo_clusters  = relationship('PhotoCluster',  back_populates='cluster', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Cluster(id={self.cluster_id}, name={self.name})>"


class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'

    embedding_id  = Column(Integer, primary_key=True, autoincrement=True)
    photo_id      = Column(String(255), ForeignKey('photos.photo_id',     ondelete='CASCADE'))
    cluster_id    = Column(String(255), ForeignKey('clusters.cluster_id', ondelete='CASCADE'))
    embedding     = Column(LargeBinary, nullable=False)
    face_location = Column(Text)

    emotion            = Column(String(20))
    emotion_confidence = Column(Float)
    emotion_valence    = Column(Float)
    quality_score      = Column(Float)

    photo   = relationship('Photo',   back_populates='face_embeddings')
    cluster = relationship('Cluster', back_populates='face_embeddings')

    def __repr__(self):
        return f"<FaceEmbedding(id={self.embedding_id}, emotion={self.emotion})>"


class PhotoCluster(Base):
    __tablename__ = 'photo_clusters'

    photo_id   = Column(String(255), ForeignKey('photos.photo_id',    ondelete='CASCADE'), primary_key=True)
    cluster_id = Column(String(255), ForeignKey('clusters.cluster_id', ondelete='CASCADE'), primary_key=True)

    photo   = relationship('Photo',   back_populates='photo_clusters')
    cluster = relationship('Cluster', back_populates='photo_clusters')

    def __repr__(self):
        return f"<PhotoCluster(photo={self.photo_id}, cluster={self.cluster_id})>"


class DetectedObject(Base):
    __tablename__ = 'detected_objects'

    object_id  = Column(Integer, primary_key=True, autoincrement=True)
    photo_id   = Column(String(255), ForeignKey('photos.photo_id', ondelete='CASCADE'), nullable=False)
    label      = Column(String(100), nullable=False)
    confidence = Column(Float,       nullable=False)

    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)

    dominant_color_rgb = Column(String(50))
    color_name         = Column(String(50))
    created_at         = Column(DateTime, default=datetime.utcnow)

    photo = relationship('Photo', back_populates='detected_objects')

    def __repr__(self):
        return f"<DetectedObject(id={self.object_id}, label={self.label})>"


# ============================================================================
# OBJECT CLUSTER MODELS 
# ============================================================================

class ObjectCluster(Base):
    __tablename__ = 'object_clusters'

    cluster_id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    category   = Column(String(100), unique=True, nullable=False)
    label      = Column(String(100), nullable=False)
    icon       = Column(String(10),  nullable=True)
    photo_count = Column(Integer, default=0)

    thumbnail_photo_id = Column(String(255), ForeignKey('photos.photo_id'), nullable=True)

    thumbnail_photo = relationship('Photo', foreign_keys=[thumbnail_photo_id])
    photo_links     = relationship('PhotoObjectCluster', back_populates='cluster', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<ObjectCluster(category={self.category}, photos={self.photo_count})>"


class PhotoObjectCluster(Base):
    __tablename__ = 'photo_object_clusters'

    id         = Column(Integer, primary_key=True, autoincrement=True)
    photo_id   = Column(String(255), ForeignKey('photos.photo_id'),            nullable=False)
    cluster_id = Column(String(255), ForeignKey('object_clusters.cluster_id'), nullable=False)

    photo   = relationship('Photo',         back_populates='object_cluster_links')
    cluster = relationship('ObjectCluster', back_populates='photo_links')

    __table_args__ = (
        UniqueConstraint('photo_id', 'cluster_id', name='uq_photo_object_cluster'),
    )

    def __repr__(self):
        return f"<PhotoObjectCluster(photo={self.photo_id}, cluster={self.cluster_id})>"


# ============================================================================
# CONVERSATION & MEMORY MODELS
# ============================================================================

class Conversation(Base):
    __tablename__ = 'conversations'

    conversation_id = Column(String(255), primary_key=True, default=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    user_id         = Column(String(255), default="default_user")
    created_at      = Column(Float, nullable=False)
    updated_at      = Column(Float, nullable=False)
    message_count   = Column(Integer, default=0)
    summary         = Column(Text)

    needs_summary      = Column(Boolean, default=False)
    last_summarized_at = Column(Float)

    messages = relationship('Message', back_populates='conversation', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Conversation(id={self.conversation_id}, messages={self.message_count})>"


class Message(Base):
    __tablename__ = 'messages'

    message_id      = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(255), ForeignKey('conversations.conversation_id', ondelete='CASCADE'), nullable=False)
    role            = Column(String(20), nullable=False)
    content         = Column(Text,       nullable=False)
    retrieved_photo_ids = Column(Text)
    meta_data       = Column(JSON)
    created_at      = Column(Float, nullable=False)

    conversation = relationship('Conversation', back_populates='messages')

    def __repr__(self):
        return f"<Message(id={self.message_id}, role={self.role})>"


# ============================================================================
# DATABASE INIT 
# ============================================================================

def migrate_to_phase5():
    """Add Phase 5 columns to conversations table (safe to run multiple times)."""
    from sqlalchemy import text
    with engine.connect() as conn:
        for col_sql, col_name in [
            ("ALTER TABLE conversations ADD COLUMN needs_summary BOOLEAN DEFAULT FALSE", "needs_summary"),
            ("ALTER TABLE conversations ADD COLUMN last_summarized_at FLOAT",            "last_summarized_at"),
        ]:
            try:
                conn.execute(text(col_sql))
                print(f"✓ Added {col_name}")
            except Exception:
                print(f"  {col_name} already exists (skipping)")
        conn.commit()


def migrate_to_phase5_plus():
    """
    Create Phase 5+ tables and columns if they do not yet exist.
    Safe to call on every startup.
    """
    from sqlalchemy import text, inspect as sa_inspect

    # New optional columns on photos
    photo_migrations = [
        ("ALTER TABLE photos ADD COLUMN face_count INTEGER DEFAULT 0", "photos.face_count"),
        ("ALTER TABLE photos ADD COLUMN weather VARCHAR(50)",          "photos.weather"),
    ]
    with engine.connect() as conn:
        for col_sql, col_name in photo_migrations:
            try:
                conn.execute(text(col_sql))
                print(f"✓ Added column {col_name}")
            except Exception:
                print(f"  Column {col_name} already exists (skipping)")
        conn.commit()

    # Create new tables (idempotent)
    inspector = sa_inspect(engine)
    existing  = inspector.get_table_names()
    for table in [ObjectCluster.__table__, PhotoObjectCluster.__table__]:
        if table.name not in existing:
            table.create(engine)
            print(f"✓ Created table '{table.name}'")
        else:
            print(f"  Table '{table.name}' already exists (skipping)")


def init_db():
    
    # create_all is a no-op for tables that already exist
    Base.metadata.create_all(engine)

    # Apply incremental migrations for tables/columns added after initial deploy
    try:
        migrate_to_phase5()
    except Exception as e:
        print(f"Phase 5 migration note: {e}")

    try:
        migrate_to_phase5_plus()
    except Exception as e:
        print(f"Phase 5+ migration note: {e}")

    print("✓ Database tables initialized/verified (Phase 5+)")


if __name__ == '__main__':
    try:
        print(f"Connecting to: {DATABASE_URL.replace(DB_PASSWORD, '***')}")
        init_db()

        session = Session()
        print(f"\n✓ Database connected successfully")
        print(f"  - Photos          : {session.query(Photo).count()}")
        print(f"  - People clusters : {session.query(Cluster).count()}")
        print(f"  - Object clusters : {session.query(ObjectCluster).count()}")
        print(f"  - Conversations   : {session.query(Conversation).count()}")
        print(f"  - Messages        : {session.query(Message).count()}")
        session.close()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")