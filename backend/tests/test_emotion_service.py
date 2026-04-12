from services.face_service import get_face_service
from services.emotion_service import get_emotion_service

test_photo = "uploads/test.jpg"

face_service = get_face_service()
emotion_service = get_emotion_service()

encodings, locations, _ = face_service.detect_faces(test_photo)

if not locations:
    print("✗ No faces detected — cannot test emotion")
else:
    emotion = emotion_service.detect_emotion(test_photo, locations[0])
    print(f"✓ Emotion: {emotion['dominant_emotion']}")
    print(f"✓ Confidence: {emotion['confidence']}")