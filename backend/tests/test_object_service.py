from services.object_service import get_object_service

test_photo = "uploads/test1.jpg"

object_service = get_object_service()
objects = object_service.detect_objects(test_photo)

print(f"✓ Objects detected: {len(objects)}")
for obj in objects[:5]:
    print(f"- {obj['label']} ({obj['confidence']:.2f})")