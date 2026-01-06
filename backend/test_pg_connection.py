#
# DO NOT PUSH IT TO GITHUB!!!!!! ❌ it stores your password!
#


import psycopg2

# Connection parameters
conn_params = {
    'dbname': 'lumeo_db',
    'user': 'lumeo_user',
    'password': 'lumeo_123',
    'host': 'localhost',
    'port': 5432
}

try:
    # Connect
    print("🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Test query
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"Connected successfully!")
    print(f"PostgreSQL version: {version[0]}")

    # Test create/insert/query
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            data TEXT
        );
    """)
    cursor.execute("INSERT INTO test_table (data) VALUES ('test');")
    conn.commit()

    cursor.execute("SELECT * FROM test_table;")
    result = cursor.fetchall()
    print(f"Test query result: {result}")

    # Cleanup
    cursor.execute("DROP TABLE test_table;")
    conn.commit()

    cursor.close()
    conn.close()
    print("✅ All tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
