import { useState } from "react";
import { Calendar, Smile } from "lucide-react";
import { BASE_URL } from "../constants/config";

/**
 * PhotoCard
 * Props:
 *   photo    – photo object from API
 *   onClick  – click handler
 *   compact  – hide metadata strip (default: false)
 */
const PhotoCard = ({ photo, onClick, compact = false }) => {
  const [isHovered, setIsHovered] = useState(false);

  const src = photo.path
    ? `${BASE_URL}/${photo.path}`
    : `${BASE_URL}/uploads/${photo.filename}`;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        background: "rgba(255, 255, 255, 0.08)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(255, 255, 255, 0.15)",
        borderRadius: compact ? "12px" : "16px",
        overflow: "hidden",
        cursor: "pointer",
        transition: "all 0.3s ease",
        transform: isHovered ? "translateY(-4px) scale(1.02)" : "none",
        boxShadow: isHovered
          ? "0 12px 24px rgba(0, 0, 0, 0.4)"
          : "0 4px 12px rgba(0, 0, 0, 0.2)",
      }}
    >
      {/* Image */}
      <div
        style={{
          width: "100%",
          aspectRatio: "1",
          background: "rgba(0, 0, 0, 0.3)",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <img
          src={src}
          alt={photo.caption || "Photo"}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
          loading="lazy"
        />

        {photo.similarity && (
          <div
            style={{
              position: "absolute",
              top: "8px",
              right: "8px",
              background: "rgba(147, 197, 253, 0.9)",
              backdropFilter: "blur(10px)",
              padding: "4px 10px",
              borderRadius: "12px",
              fontSize: "11px",
              fontWeight: "600",
              color: "#000",
            }}
          >
            {Math.round(photo.similarity * 100)}%
          </div>
        )}
      </div>

      {/* Metadata strip */}
      {!compact && (
        <div style={{ padding: "12px" }}>
          {photo.caption && (
            <div
              style={{
                fontSize: "13px",
                fontWeight: "500",
                marginBottom: "8px",
                color: "rgba(255, 255, 255, 0.9)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
              }}
            >
              {photo.caption}
            </div>
          )}

          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "6px",
              fontSize: "11px",
              color: "rgba(255, 255, 255, 0.6)",
            }}
          >
            {photo.date_taken && (
              <span
                style={{ display: "flex", alignItems: "center", gap: "4px" }}
              >
                <Calendar size={11} />
                {new Date(photo.date_taken).toLocaleDateString()}
              </span>
            )}
            {photo.dominant_emotion && (
              <span
                style={{ display: "flex", alignItems: "center", gap: "4px" }}
              >
                <Smile size={11} />
                {photo.dominant_emotion}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PhotoCard;
