import { useState, useEffect } from "react";
import { BASE_URL } from "../constants/config";
import { Tag, Grid, X } from "lucide-react";

// Matches the accent() palette from ClusterManager / ClusterComponents
const ACCENTS = [
  "#63b3ed",
  "#68d391",
  "#f6ad55",
  "#fc8181",
  "#b794f4",
  "#76e4f7",
  "#fbd38d",
  "#9ae6b4",
];
const accent = (i) => ACCENTS[i % ACCENTS.length];

export default function ObjectClusters({ objectsOnly = false }) {
  const [clusters, setClusters] = useState([]);
  const [selected, setSelected] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${BASE_URL}/api/object-clusters?objects_only=${objectsOnly}`)
      .then((r) => r.json())
      .then((d) => {
        setClusters(d.clusters || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [objectsOnly]);

  useEffect(() => {
    if (!selected) {
      setPhotos([]);
      return;
    }
    fetch(
      `${BASE_URL}/api/object-clusters/${selected}/photos?objects_only=${objectsOnly}&limit=60`,
    )
      .then((r) => r.json())
      .then((d) => setPhotos(d.photos || []));
  }, [selected, objectsOnly]);

  const selectedCluster = clusters.find((c) => c.cluster_id === selected);
  const selectedIdx = clusters.findIndex((c) => c.cluster_id === selected);

  // ── Loading ────────────────────────────────────────────────────────────────
  if (loading)
    return (
      <div style={{ textAlign: "center", padding: "100px 20px" }}>
        <div style={{ display: "inline-flex", gap: "6px" }}>
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#63b3ed",
                animation: "bounce 1.4s infinite ease-in-out",
                animationDelay: `${i * 0.16}s`,
              }}
            />
          ))}
        </div>
        <p
          style={{
            marginTop: "16px",
            fontSize: "13px",
            color: "rgba(255,255,255,0.3)",
          }}
        >
          Loading clusters…
        </p>
        <style>{`@keyframes bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}`}</style>
      </div>
    );

  // ── Empty ──────────────────────────────────────────────────────────────────
  if (clusters.length === 0)
    return (
      <div
        style={{
          borderRadius: "20px",
          padding: "80px 40px",
          textAlign: "center",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.07)",
        }}
      >
        <Grid
          size={48}
          color="rgba(255,255,255,0.1)"
          style={{ marginBottom: "16px" }}
        />
        <h3
          style={{
            margin: "0 0 8px",
            color: "rgba(255,255,255,0.5)",
            fontWeight: 600,
          }}
        >
          No clusters yet
        </h3>
        <p
          style={{
            margin: 0,
            fontSize: "13px",
            color: "rgba(255,255,255,0.25)",
          }}
        >
          Upload and process photos to see object clusters here.
        </p>
      </div>
    );

  // ── Main ───────────────────────────────────────────────────────────────────
  return (
    <div style={{ fontFamily: "inherit" }}>
      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40%            { transform: scale(1); }
        }
        .obj-chip:hover { background: rgba(255,255,255,0.09) !important; }
        .obj-photo:hover img { opacity: 0.85; }
      `}</style>

      {/* Chip grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(155px, 1fr))",
          gap: "12px",
          marginBottom: selected ? "20px" : 0,
        }}
      >
        {clusters.map((c, idx) => {
          const isActive = selected === c.cluster_id;
          const color = accent(idx);
          return (
            <button
              key={c.cluster_id}
              className="obj-chip"
              onClick={() => setSelected(isActive ? null : c.cluster_id)}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: "10px",
                padding: "14px",
                borderRadius: "16px",
                cursor: "pointer",
                textAlign: "left",
                border: isActive
                  ? `1px solid ${color}55`
                  : "1px solid rgba(255,255,255,0.08)",
                background: isActive
                  ? `rgba(${hexToRgb(color)}, 0.08)`
                  : "rgba(255,255,255,0.04)",
                transition: "all 0.2s",
                position: "relative",
                outline: "none",
              }}
            >
              {/* Thumbnail */}
              <div
                style={{
                  width: "100%",
                  aspectRatio: "16/9",
                  borderRadius: "10px",
                  overflow: "hidden",
                  background: "rgba(255,255,255,0.06)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                {c.thumbnail_url ? (
                  <img
                    src={`${BASE_URL}${c.thumbnail_url}`}
                    alt={c.label}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      display: "block",
                    }}
                    onError={(e) => {
                      e.target.style.display = "none";
                      e.target.parentNode.innerHTML = `<span style="font-size:24px">${c.icon || "📷"}</span>`;
                    }}
                  />
                ) : (
                  <span style={{ fontSize: "24px" }}>{c.icon || "📷"}</span>
                )}
              </div>

              {/* Label + count */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "3px",
                  width: "100%",
                }}
              >
                <span
                  style={{
                    fontSize: "13px",
                    fontWeight: 700,
                    letterSpacing: "-0.01em",
                    color: isActive ? color : "#fff",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    textTransform: "capitalize",
                  }}
                >
                  {c.label}
                </span>
                <span
                  style={{
                    fontSize: "11px",
                    color: "rgba(255,255,255,0.35)",
                    fontWeight: 500,
                  }}
                >
                  {c.photo_count} {c.photo_count !== 1 ? "photos" : "photo"}
                </span>
              </div>

              {/* Active dot */}
              {isActive && (
                <div
                  style={{
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    width: "7px",
                    height: "7px",
                    borderRadius: "50%",
                    background: color,
                    boxShadow: `0 0 8px ${color}`,
                  }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Detail panel */}
      {selected && (
        <div
          style={{
            borderRadius: "20px",
            border: `1px solid ${accent(selectedIdx)}33`,
            background: `rgba(${hexToRgb(accent(selectedIdx))}, 0.04)`,
            padding: "20px",
          }}
        >
          {/* Header */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "16px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div
                style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "8px",
                  background: `rgba(${hexToRgb(accent(selectedIdx))}, 0.15)`,
                  border: `1px solid ${accent(selectedIdx)}33`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Tag size={14} color={accent(selectedIdx)} />
              </div>
              <div>
                <h3
                  style={{
                    margin: 0,
                    fontSize: "15px",
                    fontWeight: 800,
                    color: "#fff",
                    letterSpacing: "-0.02em",
                    textTransform: "capitalize",
                  }}
                >
                  {selectedCluster?.label}
                </h3>
                <p
                  style={{
                    margin: 0,
                    fontSize: "12px",
                    color: "rgba(255,255,255,0.35)",
                  }}
                >
                  {photos.length} photos
                </p>
              </div>
            </div>
            <button
              onClick={() => setSelected(null)}
              style={{
                padding: "6px",
                borderRadius: "8px",
                cursor: "pointer",
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "rgba(255,255,255,0.45)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <X size={13} />
            </button>
          </div>

          {/* Photos */}
          {photos.length === 0 ? (
            <p
              style={{
                textAlign: "center",
                fontSize: "13px",
                color: "rgba(255,255,255,0.25)",
                padding: "40px 0",
              }}
            >
              No photos in this cluster.
            </p>
          ) : (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
                gap: "8px",
              }}
            >
              {photos.map((p) => (
                <div
                  key={p.photo_id}
                  className="obj-photo"
                  style={{
                    aspectRatio: "1",
                    borderRadius: "10px",
                    overflow: "hidden",
                    background: "rgba(255,255,255,0.06)",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <img
                    src={`${BASE_URL}${p.thumbnail_url}`}
                    alt={p.filename}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      display: "block",
                      transition: "opacity 0.2s",
                    }}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Helper: "#63b3ed" → "99,179,237"
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}
