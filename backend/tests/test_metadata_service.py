from services.metadata_service import get_metadata_service

test_photo = "uploads/test.jpg"

metadata_service = get_metadata_service()
metadata = metadata_service.extract_exif(test_photo)

print("✓ Date taken:", metadata.get("date_taken"))
print("✓ Season:", metadata.get("season"))
print("✓ Time of day:", metadata.get("time_of_day"))
print("✓ Quality score:", metadata.get("quality_score"))