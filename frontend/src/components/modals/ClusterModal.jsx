import { X, Users } from "lucide-react";
import { BASE_URL } from "../../constants/config";

/**
 * ClusterModal
 * Shows all photos for a given face cluster (person).
 *
 * Props:
 *   viewingCluster   – cluster object (or null to hide)
 *   clusterPhotos    – array of photos for this cluster
 *   onClose          – () => void
 *   onPhotoClick     – (photo) => void  – opens full-screen viewer
 *   updateClusterName – (clusterId, newName) => void
 */
const ClusterModal = ({
  viewingCluster,
  clusterPhotos,
  onClose,
  onPhotoClick,
  updateClusterName,
}) => {
  if (!viewingCluster) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.85)",
        backdropFilter: "blur(20px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: "20px",
      }}
      onClick={onClose}
    >
      <div
        className="glass-card"
        style={{
          borderRadius: "24px",
          padding: "40px",
          maxWidth: "1000px",
          width: "100%",
          maxHeight: "90vh",
          overflowY: "auto",
          position: "relative",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="glass-button"
          style={{
            position: "absolute",
            top: "16px",
            right: "16px",
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 0,
          }}
        >
          <X size={20} />
        </button>

        <h2
          style={{ fontSize: "28px", marginBottom: "8px", fontWeight: "600" }}
        >
          {viewingCluster.name}
        </h2>
        <p style={{ color: "rgba(255,255,255,0.6)", marginBottom: "24px" }}>
          {clusterPhotos.length} photos
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
            gap: "12px",
          }}
        >
          {clusterPhotos.map((photo, idx) => {
            const src = photo.path
              ? `${BASE_URL}/${photo.path}`
              : `${BASE_URL}/uploads/${photo.filename}`;

            return (
              <div
                key={photo.photo_id}
                style={{
                  aspectRatio: "1",
                  borderRadius: "12px",
                  overflow: "hidden",
                  cursor: "pointer",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                }}
                onClick={() => onPhotoClick({ ...photo, index: idx })}
              >
                <img
                  src={src}
                  alt={photo.filename}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ClusterModal;
