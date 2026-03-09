import { useState, useEffect } from "react";
import { Calendar } from "lucide-react";
import PhotoCard from "./PhotoCard";

/**
 * TimelineView
 * Props:
 *   photos       – array of photo objects
 *   onPhotoClick – (photo) => void
 */
const TimelineView = ({ photos, onPhotoClick }) => {
  const [groupedPhotos, setGroupedPhotos] = useState({});

  useEffect(() => {
    const groups = {};

    photos.forEach((photo) => {
      if (!photo.date_taken) return;

      const date = new Date(photo.date_taken);
      const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
      const label = date.toLocaleDateString("en-US", {
        month: "long",
        year: "numeric",
      });

      if (!groups[key]) groups[key] = { label, photos: [], emotions: {} };

      groups[key].photos.push(photo);

      if (photo.dominant_emotion) {
        groups[key].emotions[photo.dominant_emotion] =
          (groups[key].emotions[photo.dominant_emotion] || 0) + 1;
      }
    });

    // Sort newest first
    const sorted = Object.entries(groups)
      .sort(([a], [b]) => b.localeCompare(a))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

    setGroupedPhotos(sorted);
  }, [photos]);

  if (Object.keys(groupedPhotos).length === 0) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "60px 20px",
          color: "rgba(255, 255, 255, 0.4)",
        }}
      >
        <p>No dated photos found in this search.</p>
      </div>
    );
  }

  return (
    <div
      className="chat-scroll"
      style={{ padding: "20px", overflowY: "auto", height: "100%" }}
    >
      {Object.entries(groupedPhotos).map(([key, group]) => (
        <div key={key} style={{ marginBottom: "32px" }}>
          {/* Month header */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              marginBottom: "16px",
              paddingBottom: "12px",
              borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
            }}
          >
            <Calendar size={16} color="#93c5fd" />
            <span style={{ fontSize: "16px", fontWeight: "600" }}>
              {group.label}
            </span>
            <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.5)" }}>
              ({group.photos.length})
            </span>
          </div>

          {/* Photo grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
              gap: "12px",
            }}
          >
            {group.photos.map((photo, idx) => (
              <PhotoCard
                key={idx}
                photo={photo}
                onClick={() => onPhotoClick(photo)}
                compact
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default TimelineView;