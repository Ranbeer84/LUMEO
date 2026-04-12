from services.pipeline_service import get_pipeline

test_photo = "uploads/test.jpg"

pipeline = get_pipeline()

stats = pipeline.get_processing_stats()
print("Service readiness:")
for service, ready in stats.items():
    print(f"  {'✓' if ready else '✗'} {service}")

result = pipeline.process_photo(test_photo)

print("\n✓ Pipeline finished")
print("Faces:", result.get("face_count"))
print("Objects:", result.get("object_count"))
print("Scene:", result["scene"].get("scene_type"))
print("Emotion:", result["photo_emotion"].get("dominant_emotion"))
print("Time:", result.get("processing_time"), "seconds")