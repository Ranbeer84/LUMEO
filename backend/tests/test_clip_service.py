from services.clip_service import get_clip_service

test_photo = "uploads/test.jpg"

clip_service = get_clip_service()

image_emb = clip_service.encode_image(test_photo)
print("✓ Image embedding:", image_emb.shape)

text_emb = clip_service.encode_text("happy people at beach")
print("✓ Text embedding:", text_emb.shape)