"""
migrate.py — run once to sync database schema with models.py
"""
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/lumeo_db")
engine = create_engine(DATABASE_URL)

migrations = [
    # Add missing columns to photos table
    "ALTER TABLE photos ADD COLUMN IF NOT EXISTS face_count    INTEGER DEFAULT 0",
    "ALTER TABLE photos ADD COLUMN IF NOT EXISTS weather       VARCHAR(50)",
    "ALTER TABLE photos ADD COLUMN IF NOT EXISTS image_quality FLOAT",

    # Create object_clusters table
    """
    CREATE TABLE IF NOT EXISTS object_clusters (
        cluster_id         VARCHAR PRIMARY KEY,
        category           VARCHAR(100) UNIQUE NOT NULL,
        label              VARCHAR(100) NOT NULL,
        icon               VARCHAR(10),
        photo_count        INTEGER DEFAULT 0,
        thumbnail_photo_id VARCHAR REFERENCES photos(photo_id)
    )
    """,

    # Create photo_object_clusters junction table
    """
    CREATE TABLE IF NOT EXISTS photo_object_clusters (
        id         SERIAL PRIMARY KEY,
        photo_id   VARCHAR NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        cluster_id VARCHAR NOT NULL REFERENCES object_clusters(cluster_id) ON DELETE CASCADE,
        CONSTRAINT uq_photo_object_cluster UNIQUE (photo_id, cluster_id)
    )
    """,
]

with engine.connect() as conn:
    for sql in migrations:
        try:
            conn.execute(text(sql))
            print(f"✓ {sql.strip()[:60]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
    conn.commit()
    print("\n✓ Migration complete")