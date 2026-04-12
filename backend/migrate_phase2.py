"""
Phase 2 Migration Script
Apply schema enhancements for vision intelligence layer
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv
import sys

load_dotenv()

def run_migration():
    """Apply Phase 2 schema changes"""
    
    print("=" * 60)
    print("Phase 2 Database Migration")
    print("=" * 60)
    print()
    
    # Database connection parameters
    db_params = {
        'dbname': os.getenv('DB_NAME', 'lumeo_db'),
        'user': os.getenv('DB_USER', 'lumeo_user'),
        'password': os.getenv('DB_PASSWORD', 'lumeo_password'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    try:
        # Connect to database
        print(f"Connecting to database: {db_params['dbname']}...")
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("✓ Connected successfully\n")
        
        # Read and execute schema file
        print("Reading schema_phase2.sql...")
        with open('schema_phase2.sql', 'r') as f:
            schema_sql = f.read()
        
        print("Applying schema changes...")
        print()
        
        # Split by semicolon and execute each statement
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        
        for idx, statement in enumerate(statements, 1):
            try:
                # Skip comments
                if statement.startswith('--') or statement.startswith('COMMENT'):
                    continue
                
                # Execute statement
                cursor.execute(statement + ';')
                
                # Print what was done
                if 'ALTER TABLE' in statement:
                    print(f"  ✓ Step {idx}: Altered table")
                elif 'CREATE TABLE' in statement:
                    table_name = statement.split('CREATE TABLE')[1].split('(')[0].strip()
                    print(f"  ✓ Step {idx}: Created table {table_name}")
                elif 'CREATE INDEX' in statement:
                    index_name = statement.split('CREATE INDEX')[1].split('ON')[0].strip()
                    print(f"  ✓ Step {idx}: Created index {index_name}")
                
            except psycopg2.Error as e:
                # Check if error is about column/index already existing
                if 'already exists' in str(e):
                    print(f"  ⊙ Step {idx}: Already exists (skipping)")
                else:
                    print(f"  × Step {idx}: ERROR - {e}")
                    raise
        
        print()
        print("=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
        print()
        
        # Verify new schema
        print("Verifying schema...")
        print()
        
        # Check photos table columns
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'photos'
            ORDER BY ordinal_position
        """)
        
        photo_columns = cursor.fetchall()
        print(f"Photos table now has {len(photo_columns)} columns:")
        for col_name, col_type in photo_columns:
            print(f"  - {col_name} ({col_type})")
        
        print()
        
        # Check if detected_objects table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'detected_objects'
            )
        """)
        
        objects_exists = cursor.fetchone()[0]
        if objects_exists:
            print("✓ detected_objects table created")
        else:
            print("× detected_objects table missing")
        
        print()
        
        # Check indexes
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename IN ('photos', 'detected_objects', 'face_embeddings')
            ORDER BY indexname
        """)
        
        indexes = cursor.fetchall()
        print(f"Found {len(indexes)} indexes:")
        for (index_name,) in indexes:
            print(f"  - {index_name}")
        
        print()
        print("=" * 60)
        print("Phase 2 migration verified successfully!")
        print("=" * 60)
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ Error: schema_phase2.sql file not found")
        print("Make sure you're running this script from the backend directory")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def rollback_migration():
    """Rollback Phase 2 changes (use with caution!)"""
    
    print("=" * 60)
    print("WARNING: Rolling back Phase 2 migration")
    print("This will remove all Phase 2 columns and tables!")
    print("=" * 60)
    print()
    
    response = input("Are you sure you want to rollback? (type 'yes' to confirm): ")
    
    if response.lower() != 'yes':
        print("Rollback cancelled")
        return False
    
    db_params = {
        'dbname': os.getenv('DB_NAME', 'lumeo_db'),
        'user': os.getenv('DB_USER', 'lumeo_user'),
        'password': os.getenv('DB_PASSWORD', 'lumeo_password'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    try:
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("Dropping detected_objects table...")
        cursor.execute("DROP TABLE IF EXISTS detected_objects CASCADE")
        
        print("Removing Phase 2 columns from photos...")
        phase2_columns = [
            'clip_embedding', 'scene_type', 'location_type', 'activity',
            'season', 'time_of_day', 'date_taken', 'camera_make',
            'camera_model', 'gps_latitude', 'gps_longitude',
            'image_quality', 'caption', 'dominant_emotion', 'mood_score'
        ]
        
        for col in phase2_columns:
            try:
                cursor.execute(f"ALTER TABLE photos DROP COLUMN IF EXISTS {col}")
                print(f"  ✓ Removed {col}")
            except:
                print(f"  × Failed to remove {col}")
        
        print("Removing Phase 2 columns from face_embeddings...")
        face_columns = ['emotion', 'emotion_confidence', 'emotion_valence', 'quality_score']
        
        for col in face_columns:
            try:
                cursor.execute(f"ALTER TABLE face_embeddings DROP COLUMN IF EXISTS {col}")
                print(f"  ✓ Removed {col}")
            except:
                print(f"  × Failed to remove {col}")
        
        print("\n✓ Rollback complete")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Rollback error: {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'rollback':
        success = rollback_migration()
    else:
        success = run_migration()
    
    sys.exit(0 if success else 1)