import { X } from "lucide-react";
import { BASE_URL } from "../../constants/config";

/**
 * PhotoModal
 * Full-screen lightbox for a single photo.
 *
 * Props:
 *   selectedPhoto – photo object (or null to hide)
 *   onClose       – () => void
 */
const PhotoModal = ({ selectedPhoto, onClose }) => {
  if (!selectedPhoto) return null;

  const src = selectedPhoto.path
    ? `${BASE_URL}/${selectedPhoto.path}`
    : `${BASE_URL}/uploads/${selectedPhoto.filename}`;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.95)",
        backdropFilter: "blur(40px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 2000,
        padding: "20px",
      }}
      onClick={onClose}
    >
      {/* Close button */}
      <button
        onClick={onClose}
        className="glass-button"
        style={{
          position: "absolute",
          top: "20px",
          right: "20px",
          width: "50px",
          height: "50px",
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 0,
        }}
      >
        <X size={24} />
      </button>

      <div
        onClick={(e) => e.stopPropagation()}
        style={{ textAlign: "center", maxWidth: "90vw" }}
      >
        <img
          src={src}
          alt="Full view"
          style={{
            maxWidth: "100%",
            maxHeight: "80vh",
            borderRadius: "16px",
            boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
          }}
        />

        {selectedPhoto.caption && (
          <div
            className="glass-card"
            style={{
              marginTop: "20px",
              padding: "16px 24px",
              borderRadius: "16px",
              display: "inline-block",
            }}
          >
            <div style={{ fontSize: "14px", marginBottom: "8px" }}>
              {selectedPhoto.caption}
            </div>
            <div
              style={{
                fontSize: "12px",
                color: "rgba(255,255,255,0.6)",
                display: "flex",
                gap: "12px",
              }}
            >
              {selectedPhoto.date_taken && (
                <span>
                  📅 {new Date(selectedPhoto.date_taken).toLocaleDateString()}
                </span>
              )}
              {selectedPhoto.dominant_emotion && (
                <span>😊 {selectedPhoto.dominant_emotion}</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhotoModal;
