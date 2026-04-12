import psycopg2
import numpy as np

# Connect to database
conn = psycopg2.connect(
    dbname='lumeo_db',
    user='lumeo_user',
    password='lumeo_123',
    host='localhost',
    port=5432
)
cursor = conn.cursor()

print("🧪 Testing pgvector with realistic dimensions")
print("=" * 50)

# Create table with 512-dimensional vectors (CLIP size)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_vectors_512 (
        id SERIAL PRIMARY KEY,
        description TEXT,
        embedding vector(512)
    );
""")
conn.commit()
print("✅ Created table with vector(512)")

# Generate random embeddings
np.random.seed(42)
beach_embedding = np.random.rand(512).tolist()
mountain_embedding = np.random.rand(512).tolist()
city_embedding = np.random.rand(512).tolist()

# Insert test vectors
cursor.execute("""
    INSERT INTO test_vectors_512 (description, embedding) VALUES
    ('beach sunset', %s),
    ('mountain landscape', %s),
    ('city skyline', %s);
""", (beach_embedding, mountain_embedding, city_embedding))
conn.commit()
print("✅ Inserted 3 test embeddings (512-d)")

# Test similarity search
query_vector = beach_embedding  # Should find beach as most similar

cursor.execute("""
    SELECT
        description,
        embedding <=> %s::vector AS distance
    FROM test_vectors_512
    ORDER BY distance
    LIMIT 3;
""", (query_vector,))

results = cursor.fetchall()
print("\n📊 Similarity search results:")
print("-" * 50)
for desc, dist in results:
    print(f"  {desc:20s} | Distance: {dist:.6f}")

# Verify top result is beach
assert results[0][0] == 'beach sunset', "Top result should be beach"
assert results[0][1] == 0.0, "Distance to self should be 0"
print("\n✅ Similarity search working correctly!")

# Cleanup
cursor.execute("DROP TABLE test_vectors_512;")
conn.commit()

cursor.close()
conn.close()
print("\n✅ All pgvector tests passed!")
