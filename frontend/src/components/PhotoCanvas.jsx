import { Image as ImageIcon, LayoutGrid, List } from "lucide-react";
import PhotoCard from "./PhotoCard";
import TimelineView from "./TimelineView";

/**
 * PhotoCanvas (right column in Chat mode)
 *
 * Props:
 *   retrievedPhotos – array of photo objects
 *   viewMode        – 'grid' | 'timeline'
 *   setViewMode     – (mode) => void
 *   onPhotoClick    – (photo) => void  – opens full-screen viewer
 */
const PhotoCanvas = ({
  retrievedPhotos,
  viewMode,
  setViewMode,
  onPhotoClick,
}) => {
  const hasPhotos = retrievedPhotos.length > 0;

  return (
    <div
      className="glass-card"
      style={{ borderRadius: "24px", overflow: "hidden", position: "relative" }}
    >
      {/* Toolbar – only shown when photos are present */}
      {hasPhotos && (
        <div
          style={{
            padding: "16px 20px",
            borderBottom: "1px solid rgba(255,255,255,0.1)",
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <div style={{ flex: 1, fontSize: "14px", fontWeight: "500" }}>
            Search Results
          </div>

          <div
            style={{
              display: "flex",
              background: "rgba(0,0,0,0.2)",
              borderRadius: "10px",
              padding: "4px",
              gap: "4px",
            }}
          >
            {[
              { id: "grid", icon: <LayoutGrid size={14} />, label: "Grid" },
              { id: "timeline", icon: <List size={14} />, label: "Timeline" },
            ].map(({ id, icon, label }) => (
              <button
                key={id}
                onClick={() => setViewMode(id)}
                style={{
                  background:
                    viewMode === id ? "rgba(59, 130, 246, 0.3)" : "transparent",
                  border: "none",
                  padding: "6px 12px",
                  borderRadius: "8px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "12px",
                  color: "white",
                }}
              >
                {icon} {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!hasPhotos && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            textAlign: "center",
            opacity: 0.4,
          }}
        >
          <div
            style={{
              width: 80,
              height: 80,
              borderRadius: "24px",
              background: "rgba(255,255,255,0.05)",
              margin: "0 auto 24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ImageIcon size={40} />
          </div>
          <h3 style={{ margin: "0 0 8px 0", fontWeight: 500 }}>
            Visual Context
          </h3>
          <p style={{ margin: 0, fontSize: "14px" }}>
            Photos will appear here based on your chat
          </p>
        </div>
      )}

      {/* Grid view */}
      {hasPhotos && viewMode === "grid" && (
        <div
          className="chat-scroll"
          style={{
            padding: "20px",
            overflowY: "auto",
            height: "calc(100% - 65px)",
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: "16px",
            }}
          >
            {retrievedPhotos.map((photo, idx) => (
              <PhotoCard
                key={idx}
                photo={photo}
                onClick={() => onPhotoClick(photo)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Timeline view */}
      {hasPhotos && viewMode === "timeline" && (
        <TimelineView photos={retrievedPhotos} onPhotoClick={onPhotoClick} />
      )}
    </div>
  );
};

export default PhotoCanvas;
