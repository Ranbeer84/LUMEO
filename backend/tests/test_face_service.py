from services.face_service import get_face_service

test_photo = "uploads/test.jpg"

face_service = get_face_service()
encodings, locations, qualities = face_service.detect_faces(test_photo)

print(f"✓ Faces detected: {len(encodings)}")
print(f"✓ Locations: {locations}")
print(f"✓ Quality scores: {qualities}")