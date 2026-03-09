import { useState, useEffect, useCallback } from "react";
import {
  ChevronLeft,
  Users,
  Edit2,
  Check,
  X,
  Merge,
  Trash2,
  AlertTriangle,
  Search,
  Camera,
  Layers,
} from "lucide-react";
import { BASE_URL } from "../constants/config";

// ─── Palette per cluster index ──────────────────────────────────────────────
const ACCENT_COLORS = [
  {
    bg: "rgba(99,179,237,0.12)",
    border: "rgba(99,179,237,0.35)",
    dot: "#63b3ed",
  },
  {
    bg: "rgba(154,117,234,0.12)",
    border: "rgba(154,117,234,0.35)",
    dot: "#9a75ea",
  },
  {
    bg: "rgba(72,199,142,0.12)",
    border: "rgba(72,199,142,0.35)",
    dot: "#48c78e",
  },
  {
    bg: "rgba(255,159,67,0.12)",
    border: "rgba(255,159,67,0.35)",
    dot: "#ff9f43",
  },
  {
    bg: "rgba(252,100,113,0.12)",
    border: "rgba(252,100,113,0.35)",
    dot: "#fc6471",
  },
  {
    bg: "rgba(254,215,83,0.12)",
    border: "rgba(254,215,83,0.35)",
    dot: "#fed753",
  },
];
const accent = (i) => ACCENT_COLORS[i % ACCENT_COLORS.length];

// ─── Confirm Dialog ──────────────────────────────────────────────────────────
const ConfirmDialog = ({ message, onConfirm, onCancel }) => (
  <div
    style={{
      position: "fixed",
      inset: 0,
      background: "rgba(0,0,0,0.75)",
      backdropFilter: "blur(20px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 4000,
      padding: "20px",
    }}
    onClick={onCancel}
  >
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        background: "rgba(18,18,32,0.95)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "20px",
        padding: "32px",
        maxWidth: "400px",
        width: "100%",
        boxShadow: "0 32px 64px rgba(0,0,0,0.6)",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "14px",
          marginBottom: "24px",
          alignItems: "flex-start",
        }}
      >
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: "10px",
            flexShrink: 0,
            background: "rgba(245,158,11,0.15)",
            border: "1px solid rgba(245,158,11,0.3)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <AlertTriangle size={18} color="#f59e0b" />
        </div>
        <p
          style={{
            margin: 0,
            fontSize: "14px",
            lineHeight: 1.7,
            color: "rgba(255,255,255,0.85)",
            paddingTop: "8px",
          }}
        >
          {message}
        </p>
      </div>
      <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
        <button
          onClick={onCancel}
          style={{
            padding: "9px 20px",
            borderRadius: "10px",
            fontSize: "13px",
            cursor: "pointer",
            background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.12)",
            color: "rgba(255,255,255,0.7)",
            fontWeight: 500,
          }}
        >
          Cancel
        </button>
        <button
          onClick={onConfirm}
          style={{
            padding: "9px 20px",
            borderRadius: "10px",
            fontSize: "13px",
            cursor: "pointer",
            background: "rgba(239,68,68,0.2)",
            border: "1px solid rgba(239,68,68,0.4)",
            color: "#fca5a5",
            fontWeight: 600,
          }}
        >
          Confirm
        </button>
      </div>
    </div>
  </div>
);

// ─── Cluster Detail Panel ────────────────────────────────────────────────────
const ClusterDetail = ({
  cluster,
  colorAccent,
  onClose,
  onRemovePhoto,
  onRename,
}) => {
  const [editing, setEditing] = useState(false);
  const [nameValue, setNameValue] = useState(cluster.name);
  const [confirmRemove, setConfirmRemove] = useState(null);

  // Sync name if cluster prop changes (e.g. after rename from card)
  useEffect(() => {
    setNameValue(cluster.name);
  }, [cluster.name]);

  const submitRename = () => {
    if (nameValue.trim() && nameValue.trim() !== cluster.name) {
      onRename(cluster.cluster_id, nameValue.trim());
    }
    setEditing(false);
  };

  return (
    // KEY FIX: explicit height so flex child scroll works
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "calc(100vh - 140px)", // ← drives the scroll
        background: "rgba(14,14,26,0.8)",
        backdropFilter: "blur(24px)",
        border: `1px solid ${colorAccent.border}`,
        borderRadius: "20px",
        overflow: "hidden",
        boxShadow: `0 0 40px ${colorAccent.bg}, 0 20px 40px rgba(0,0,0,0.4)`,
      }}
    >
      {/* Coloured top accent bar */}
      <div
        style={{
          height: 3,
          background: `linear-gradient(90deg, ${colorAccent.dot}, transparent)`,
        }}
      />

      {/* Header */}
      <div
        style={{
          padding: "18px 20px",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          flexShrink: 0, // ← never shrink header
        }}
      >
        {/* Avatar */}
        <div
          style={{
            width: 48,
            height: 48,
            borderRadius: "14px",
            overflow: "hidden",
            flexShrink: 0,
            border: `1.5px solid ${colorAccent.border}`,
          }}
        >
          {cluster.thumbnail ? (
            <img
              src={`${BASE_URL}/thumbnails/${cluster.thumbnail}`}
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
              alt=""
            />
          ) : (
            <div
              style={{
                width: "100%",
                height: "100%",
                background: colorAccent.bg,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Users size={20} color={colorAccent.dot} />
            </div>
          )}
        </div>

        {/* Name / edit */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {editing ? (
            <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              <input
                autoFocus
                value={nameValue}
                onChange={(e) => setNameValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submitRename();
                  if (e.key === "Escape") setEditing(false);
                }}
                style={{
                  flex: 1,
                  padding: "6px 10px",
                  borderRadius: "8px",
                  fontSize: "14px",
                  background: "rgba(255,255,255,0.08)",
                  border: "1px solid rgba(255,255,255,0.2)",
                  color: "#fff",
                  outline: "none",
                }}
              />
              <button
                onClick={submitRename}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "7px",
                  border: "1px solid rgba(74,222,128,0.4)",
                  background: "rgba(74,222,128,0.15)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Check size={13} color="#4ade80" />
              </button>
              <button
                onClick={() => setEditing(false)}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "7px",
                  border: "1px solid rgba(248,113,113,0.3)",
                  background: "rgba(248,113,113,0.1)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <X size={13} color="#f87171" />
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span
                style={{
                  fontWeight: 700,
                  fontSize: "15px",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: "#fff",
                }}
              >
                {cluster.name}
              </span>
              <button
                onClick={() => setEditing(true)}
                style={{
                  flexShrink: 0,
                  width: 24,
                  height: 24,
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(255,255,255,0.06)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Edit2 size={11} color="rgba(255,255,255,0.5)" />
              </button>
            </div>
          )}
          <div
            style={{
              fontSize: "11px",
              color: "rgba(255,255,255,0.35)",
              marginTop: "3px",
              display: "flex",
              gap: "10px",
            }}
          >
            <span style={{ color: colorAccent.dot, fontWeight: 600 }}>
              {cluster.photos.length}
            </span>{" "}
            photos
            <span>·</span>
            <span style={{ color: colorAccent.dot, fontWeight: 600 }}>
              {cluster.face_count}
            </span>{" "}
            faces
          </div>
        </div>

        <button
          onClick={onClose}
          style={{
            flexShrink: 0,
            width: 30,
            height: 30,
            borderRadius: "8px",
            border: "1px solid rgba(255,255,255,0.1)",
            background: "rgba(255,255,255,0.05)",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <X size={14} color="rgba(255,255,255,0.5)" />
        </button>
      </div>

      {/* ── Photo scroll area — this is the only scrollable zone ── */}
      <div
        style={{ flex: 1, overflowY: "auto", padding: "16px" }}
        className="chat-scroll"
      >
        {cluster.photos.length === 0 ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: "12px",
              opacity: 0.4,
            }}
          >
            <Camera size={32} />
            <span style={{ fontSize: "13px" }}>No photos in this cluster</span>
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))",
              gap: "8px",
            }}
          >
            {cluster.photos.map((photo) => {
              const src = photo.path
                ? `${BASE_URL}/${photo.path}`
                : `${BASE_URL}/uploads/${photo.filename}`;
              return (
                <div
                  key={photo.photo_id}
                  style={{
                    position: "relative",
                    aspectRatio: "1",
                    borderRadius: "10px",
                    overflow: "hidden",
                    border: "1px solid rgba(255,255,255,0.08)",
                    cursor: "pointer",
                    transition: "transform 0.2s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "scale(1.03)";
                    e.currentTarget.querySelector(".rm-btn").style.opacity =
                      "1";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                    e.currentTarget.querySelector(".rm-btn").style.opacity =
                      "0";
                  }}
                >
                  <img
                    src={src}
                    alt=""
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      display: "block",
                    }}
                  />

                  {/* dark scrim on hover */}
                  <div
                    className="rm-btn"
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: "rgba(0,0,0,0.45)",
                      opacity: 0,
                      transition: "opacity 0.2s",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmRemove(photo.photo_id);
                      }}
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: "50%",
                        border: "1.5px solid rgba(248,113,113,0.7)",
                        background: "rgba(239,68,68,0.6)",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Trash2 size={13} color="#fff" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {confirmRemove && (
        <ConfirmDialog
          message="Remove this photo from the cluster? The photo itself won't be deleted."
          onConfirm={() => {
            onRemovePhoto(cluster.cluster_id, confirmRemove);
            setConfirmRemove(null);
          }}
          onCancel={() => setConfirmRemove(null)}
        />
      )}
    </div>
  );
};

// ─── Cluster Card ────────────────────────────────────────────────────────────
const ClusterCard = ({
  cluster,
  colorAccent,
  index,
  isSelected,
  isMergeSource,
  onSelect,
  onStartMerge,
  onDelete,
  mergeMode,
}) => {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [hovered, setHovered] = useState(false);

  const isActive = isMergeSource || isSelected;

  return (
    <>
      <div
        onClick={
          mergeMode ? () => onStartMerge(cluster) : () => onSelect(cluster)
        }
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          borderRadius: "18px",
          overflow: "hidden",
          cursor: "pointer",
          transition: "all 0.25s cubic-bezier(0.4,0,0.2,1)",
          background: isActive
            ? colorAccent.bg
            : hovered
              ? "rgba(255,255,255,0.06)"
              : "rgba(255,255,255,0.03)",
          border: `1px solid ${isActive ? colorAccent.border : hovered ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.1)"}`,
          transform:
            hovered && !isActive
              ? "translateY(-3px)"
              : isActive
                ? "translateY(-4px)"
                : "none",
          boxShadow: isActive
            ? `0 12px 32px rgba(0,0,0,0.4), 0 0 0 1px ${colorAccent.border}`
            : hovered
              ? "0 8px 20px rgba(0,0,0,0.3)"
              : "none",
          animationDelay: `${index * 40}ms`,
        }}
      >
        {/* Top accent bar */}
        <div
          style={{
            height: 2,
            background: isActive
              ? `linear-gradient(90deg, ${colorAccent.dot}, transparent)`
              : "transparent",
            transition: "background 0.25s",
          }}
        />

        {/* Thumbnail */}
        <div
          style={{
            width: "100%",
            aspectRatio: "1",
            position: "relative",
            overflow: "hidden",
            background: "rgba(0,0,0,0.4)",
          }}
        >
          {cluster.thumbnail ? (
            <img
              src={`${BASE_URL}/thumbnails/${cluster.thumbnail}`}
              alt=""
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                display: "block",
                transform: hovered ? "scale(1.06)" : "scale(1)",
                transition: "transform 0.4s ease",
              }}
            />
          ) : (
            <div
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: colorAccent.bg,
              }}
            >
              <Users
                size={36}
                color={colorAccent.dot}
                style={{ opacity: 0.6 }}
              />
            </div>
          )}

          {/* Photo count badge */}
          <div
            style={{
              position: "absolute",
              bottom: 8,
              right: 8,
              background: "rgba(0,0,0,0.65)",
              backdropFilter: "blur(8px)",
              border: "1px solid rgba(255,255,255,0.15)",
              borderRadius: "8px",
              padding: "3px 8px",
              fontSize: "11px",
              fontWeight: 600,
              color: "#fff",
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <Camera size={9} /> {cluster.photos.length}
          </div>

          {/* Merge source overlay */}
          {isMergeSource && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: "rgba(251,191,36,0.18)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  color: "#fbbf24",
                  background: "rgba(0,0,0,0.6)",
                  padding: "4px 10px",
                  borderRadius: "6px",
                  textTransform: "uppercase",
                }}
              >
                Source
              </span>
            </div>
          )}
        </div>

        {/* Info + actions */}
        <div style={{ padding: "12px 14px 14px" }}>
          <div
            style={{
              fontWeight: 700,
              fontSize: "13px",
              color: "#fff",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              marginBottom: "3px",
            }}
          >
            {cluster.name}
          </div>
          <div
            style={{
              fontSize: "11px",
              color: "rgba(255,255,255,0.38)",
              marginBottom: mergeMode ? 0 : "12px",
            }}
          >
            {cluster.face_count} face{cluster.face_count !== 1 ? "s" : ""}
          </div>

          {!mergeMode && (
            <div
              style={{ display: "flex", gap: "6px" }}
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => onStartMerge(cluster)}
                title="Merge into another cluster"
                style={{
                  flex: 1,
                  padding: "6px 0",
                  borderRadius: "8px",
                  fontSize: "11px",
                  fontWeight: 500,
                  cursor: "pointer",
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  color: "rgba(255,255,255,0.65)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "5px",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(255,255,255,0.12)";
                  e.currentTarget.style.color = "#fff";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(255,255,255,0.06)";
                  e.currentTarget.style.color = "rgba(255,255,255,0.65)";
                }}
              >
                <Merge size={11} /> Merge
              </button>
              <button
                onClick={() => setConfirmDelete(true)}
                title="Delete cluster"
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: "8px",
                  cursor: "pointer",
                  background: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(239,68,68,0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(239,68,68,0.08)";
                }}
              >
                <Trash2 size={12} color="#f87171" />
              </button>
            </div>
          )}
        </div>
      </div>

      {confirmDelete && (
        <ConfirmDialog
          message={`Delete "${cluster.name}"? All face groupings are removed. Your photos are kept.`}
          onConfirm={() => {
            onDelete(cluster.cluster_id);
            setConfirmDelete(false);
          }}
          onCancel={() => setConfirmDelete(false)}
        />
      )}
    </>
  );
};

// ─── Stats Bar ────────────────────────────────────────────────────────────────
const StatsBar = ({ clusters }) => {
  const totalPhotos = clusters.reduce((s, c) => s + c.photos.length, 0);
  const totalFaces = clusters.reduce((s, c) => s + c.face_count, 0);

  const stats = [
    { label: "Clusters", value: clusters.length, color: "#63b3ed" },
    { label: "Total Photos", value: totalPhotos, color: "#9a75ea" },
    { label: "Total Faces", value: totalFaces, color: "#48c78e" },
  ];

  return (
    <div
      style={{
        display: "flex",
        gap: "12px",
        marginBottom: "24px",
        flexWrap: "wrap",
      }}
    >
      {stats.map(({ label, value, color }) => (
        <div
          key={label}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            padding: "10px 18px",
            borderRadius: "12px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
          }}
        >
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: color,
              boxShadow: `0 0 6px ${color}`,
            }}
          />
          <span
            style={{ fontSize: "20px", fontWeight: 700, color, lineHeight: 1 }}
          >
            {value}
          </span>
          <span
            style={{
              fontSize: "12px",
              color: "rgba(255,255,255,0.4)",
              lineHeight: 1,
            }}
          >
            {label}
          </span>
        </div>
      ))}
    </div>
  );
};

// ─── Main ClusterManager ─────────────────────────────────────────────────────
const ClusterManager = ({ setAppMode }) => {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [mergeMode, setMergeMode] = useState(false);
  const [mergeSource, setMergeSource] = useState(null);
  const [confirmMerge, setConfirmMerge] = useState(null);

  const fetchClusters = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BASE_URL}/api/clusters`);
      const data = await res.json();
      if (data.clusters) setClusters(data.clusters);
    } catch {
      setError("Failed to load clusters. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchClusters();
  }, [fetchClusters]);

  const handleRename = async (clusterId, newName) => {
    try {
      await fetch(`${BASE_URL}/api/cluster/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cluster_id: clusterId, name: newName }),
      });
      const upd = (c) =>
        c.cluster_id === clusterId ? { ...c, name: newName } : c;
      setClusters((prev) => prev.map(upd));
      if (selectedCluster?.cluster_id === clusterId)
        setSelectedCluster((prev) => ({ ...prev, name: newName }));
    } catch {
      setError("Rename failed.");
    }
  };

  const handleDelete = async (clusterId) => {
    try {
      await fetch(`${BASE_URL}/api/clusters/${clusterId}`, {
        method: "DELETE",
      });
      setClusters((prev) => prev.filter((c) => c.cluster_id !== clusterId));
      if (selectedCluster?.cluster_id === clusterId) setSelectedCluster(null);
    } catch {
      setError("Delete failed.");
    }
  };

  const handleRemovePhoto = async (clusterId, photoId) => {
    try {
      await fetch(`${BASE_URL}/api/clusters/${clusterId}/photos/${photoId}`, {
        method: "DELETE",
      });
      const upd = (c) => {
        if (c.cluster_id !== clusterId) return c;
        const photos = c.photos.filter((p) => p.photo_id !== photoId);
        return { ...c, photos, photo_count: photos.length };
      };
      setClusters((prev) => prev.map(upd));
      if (selectedCluster?.cluster_id === clusterId)
        setSelectedCluster((prev) => upd(prev));
    } catch {
      setError("Remove failed.");
    }
  };

  const handleMergeStart = (cluster) => {
    if (!mergeMode) {
      setMergeMode(true);
      setMergeSource(cluster);
      setSelectedCluster(null);
    } else if (mergeSource?.cluster_id === cluster.cluster_id) {
      setMergeMode(false);
      setMergeSource(null);
    } else {
      setConfirmMerge({ source: mergeSource, target: cluster });
    }
  };

  const executeMerge = async () => {
    const { source, target } = confirmMerge;
    setConfirmMerge(null);
    setMergeMode(false);
    setMergeSource(null);
    try {
      await fetch(`${BASE_URL}/api/clusters/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_cluster_id: source.cluster_id,
          target_cluster_id: target.cluster_id,
        }),
      });
      await fetchClusters();
    } catch {
      setError("Merge failed.");
    }
  };

  const handleSelectCluster = (cluster, idx) => {
    if (selectedCluster?.cluster_id === cluster.cluster_id) {
      setSelectedCluster(null);
    } else {
      setSelectedCluster(cluster);
      setSelectedIndex(idx);
    }
  };

  const filtered = clusters.filter((c) =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  return (
    <div style={{ maxWidth: "1440px", margin: "0 auto", padding: "4px 0" }}>
      {/* ── Top bar ── */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "20px",
          marginBottom: "28px",
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={() => setAppMode("home")}
          style={{
            padding: "10px 18px",
            borderRadius: "12px",
            cursor: "pointer",
            background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.12)",
            color: "rgba(255,255,255,0.75)",
            fontSize: "13px",
            fontWeight: 500,
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            transition: "all 0.2s",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.background = "rgba(255,255,255,0.12)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.background = "rgba(255,255,255,0.06)")
          }
        >
          <ChevronLeft size={15} /> Back
        </button>

        <div style={{ flex: 1 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              marginBottom: "4px",
            }}
          >
            <Layers size={20} color="#63b3ed" />
            <h1
              style={{
                margin: 0,
                fontSize: "24px",
                fontWeight: 800,
                letterSpacing: "-0.02em",
                color: "#fff",
              }}
            >
              Cluster Manager
            </h1>
          </div>
          <p
            style={{
              margin: 0,
              fontSize: "13px",
              color: "rgba(255,255,255,0.35)",
            }}
          >
            Rename, merge, or clean up your face groups
          </p>
        </div>

        {/* Merge mode banner */}
        {mergeMode && (
          <div
            style={{
              padding: "10px 16px",
              borderRadius: "12px",
              background: "rgba(251,191,36,0.1)",
              border: "1px solid rgba(251,191,36,0.3)",
              fontSize: "13px",
              color: "#fbbf24",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <Merge size={14} />
            <span>
              Merging <strong>"{mergeSource?.name}"</strong> — pick destination
            </span>
            <button
              onClick={() => {
                setMergeMode(false);
                setMergeSource(null);
              }}
              style={{
                padding: "4px 12px",
                borderRadius: "6px",
                fontSize: "12px",
                cursor: "pointer",
                background: "rgba(251,191,36,0.15)",
                border: "1px solid rgba(251,191,36,0.3)",
                color: "#fbbf24",
                fontWeight: 600,
              }}
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      {/* ── Error ── */}
      {error && (
        <div
          style={{
            padding: "12px 18px",
            borderRadius: "12px",
            marginBottom: "20px",
            background: "rgba(239,68,68,0.08)",
            border: "1px solid rgba(239,68,68,0.25)",
            display: "flex",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <AlertTriangle size={15} color="#f87171" />
          <span style={{ fontSize: "13px", color: "#fca5a5", flex: 1 }}>
            {error}
          </span>
          <button
            onClick={() => setError(null)}
            style={{
              background: "none",
              border: "none",
              color: "rgba(255,255,255,0.3)",
              cursor: "pointer",
              padding: 0,
            }}
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* ── Stats + Search row ── */}
      {!loading && clusters.length > 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "16px",
            marginBottom: "20px",
            flexWrap: "wrap",
          }}
        >
          <StatsBar clusters={clusters} />
          <div style={{ position: "relative", minWidth: "220px" }}>
            <Search
              size={13}
              style={{
                position: "absolute",
                left: "12px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "rgba(255,255,255,0.25)",
                pointerEvents: "none",
              }}
            />
            <input
              type="text"
              placeholder="Search clusters..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: "100%",
                padding: "10px 14px 10px 34px",
                borderRadius: "12px",
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#fff",
                fontSize: "13px",
                outline: "none",
                boxSizing: "border-box",
              }}
            />
          </div>
        </div>
      )}

      {/* ── Main content ── */}
      {loading ? (
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
        </div>
      ) : clusters.length === 0 ? (
        <div
          style={{
            borderRadius: "20px",
            padding: "80px 40px",
            textAlign: "center",
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.07)",
          }}
        >
          <Users
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
            Upload and process photos to see face clusters here.
          </p>
        </div>
      ) : (
        // KEY FIX: grid layout with fixed right panel width
        <div
          style={{
            display: "grid",
            gridTemplateColumns: selectedCluster ? "1fr 360px" : "1fr",
            gap: "20px",
            alignItems: "start",
          }}
        >
          {/* Left: scrollable cluster cards grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(155px, 1fr))",
              gap: "12px",
              alignContent: "start",
            }}
          >
            {filtered.map((cluster, idx) => (
              <ClusterCard
                key={cluster.cluster_id}
                cluster={cluster}
                colorAccent={accent(idx)}
                index={idx}
                isSelected={selectedCluster?.cluster_id === cluster.cluster_id}
                isMergeSource={mergeSource?.cluster_id === cluster.cluster_id}
                mergeMode={mergeMode}
                onSelect={(c) => handleSelectCluster(c, idx)}
                onStartMerge={handleMergeStart}
                onDelete={handleDelete}
              />
            ))}
            {filtered.length === 0 && (
              <div
                style={{
                  gridColumn: "1 / -1",
                  textAlign: "center",
                  padding: "60px",
                  color: "rgba(255,255,255,0.25)",
                  fontSize: "13px",
                }}
              >
                No clusters match "{searchQuery}"
              </div>
            )}
          </div>

          {/* Right: sticky detail panel — scroll is INSIDE ClusterDetail */}
          {selectedCluster && (
            <div style={{ position: "sticky", top: "20px" }}>
              <ClusterDetail
                key={selectedCluster.cluster_id}
                cluster={selectedCluster}
                colorAccent={accent(selectedIndex)}
                onClose={() => setSelectedCluster(null)}
                onRemovePhoto={handleRemovePhoto}
                onRename={handleRename}
              />
            </div>
          )}
        </div>
      )}

      {/* Merge confirm */}
      {confirmMerge && (
        <ConfirmDialog
          message={`Merge "${confirmMerge.source.name}" INTO "${confirmMerge.target.name}"? Source cluster will be deleted; all its photos move to the target.`}
          onConfirm={executeMerge}
          onCancel={() => {
            setConfirmMerge(null);
            setMergeMode(false);
            setMergeSource(null);
          }}
        />
      )}
    </div>
  );
};

export default ClusterManager;
