import { useState, useEffect, useRef, useCallback } from "react";
import {
  Users,
  Edit2,
  Check,
  X,
  Merge,
  Trash2,
  AlertTriangle,
  Camera,
  ChevronLeft,
  ChevronRight,
  CheckSquare,
  Square,
  ArrowUpDown,
  SortAsc,
  SortDesc,
  ZoomIn,
  Layers,
} from "lucide-react";
import { BASE_URL } from "../constants/config";

// ─── Palette ────────────────────────────────────────────────────────────────
export const ACCENT_COLORS = [
  {
    bg: "rgba(99,179,237,0.12)",
    border: "rgba(99,179,237,0.35)",
    dot: "#63b3ed",
    glow: "rgba(99,179,237,0.25)",
  },
  {
    bg: "rgba(154,117,234,0.12)",
    border: "rgba(154,117,234,0.35)",
    dot: "#9a75ea",
    glow: "rgba(154,117,234,0.25)",
  },
  {
    bg: "rgba(72,199,142,0.12)",
    border: "rgba(72,199,142,0.35)",
    dot: "#48c78e",
    glow: "rgba(72,199,142,0.25)",
  },
  {
    bg: "rgba(255,159,67,0.12)",
    border: "rgba(255,159,67,0.35)",
    dot: "#ff9f43",
    glow: "rgba(255,159,67,0.25)",
  },
  {
    bg: "rgba(252,100,113,0.12)",
    border: "rgba(252,100,113,0.35)",
    dot: "#fc6471",
    glow: "rgba(252,100,113,0.25)",
  },
  {
    bg: "rgba(254,215,83,0.12)",
    border: "rgba(254,215,83,0.35)",
    dot: "#fed753",
    glow: "rgba(254,215,83,0.25)",
  },
];
export const accent = (i) => ACCENT_COLORS[i % ACCENT_COLORS.length];

// ─── Keyframe injection (once) ───────────────────────────────────────────────
const STYLES_ID = "lumeo-cluster-styles";
if (!document.getElementById(STYLES_ID)) {
  const s = document.createElement("style");
  s.id = STYLES_ID;
  s.textContent = `
    @keyframes lumeo-fadeUp   { from { opacity:0; transform:translateY(14px) } to { opacity:1; transform:none } }
    @keyframes lumeo-scaleIn  { from { opacity:0; transform:scale(0.94) }      to { opacity:1; transform:scale(1) } }
    @keyframes lumeo-shimmer  { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
    @keyframes lumeo-spinSlow { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
    .lumeo-card-enter         { animation: lumeo-fadeUp  0.38s cubic-bezier(0.22,1,0.36,1) both; }
    .lumeo-lightbox-enter     { animation: lumeo-scaleIn 0.22s ease both; }
    .lumeo-shimmer-bar {
      background: linear-gradient(90deg, transparent, var(--dot), transparent);
      background-size: 200% 100%;
      animation: lumeo-shimmer 3s infinite;
    }
  `;
  document.head.appendChild(s);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
const photoSrc = (photo) =>
  photo.path
    ? `${BASE_URL}/${photo.path}`
    : `${BASE_URL}/uploads/${photo.filename}`;

// ─── Confirm Dialog ──────────────────────────────────────────────────────────
export const ConfirmDialog = ({
  message,
  onConfirm,
  onCancel,
  confirmLabel = "Confirm",
  danger = true,
}) => (
  <div
    style={{
      position: "fixed",
      inset: 0,
      background: "rgba(0,0,0,0.82)",
      backdropFilter: "blur(22px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 4000,
      padding: "20px",
    }}
    onClick={onCancel}
  >
    <div
      className="lumeo-lightbox-enter"
      onClick={(e) => e.stopPropagation()}
      style={{
        background: "rgba(12,12,22,0.98)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "20px",
        padding: "32px",
        maxWidth: "420px",
        width: "100%",
        boxShadow: "0 40px 80px rgba(0,0,0,0.75)",
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
            background: danger
              ? "rgba(245,158,11,0.14)"
              : "rgba(99,179,237,0.14)",
            border: `1px solid ${danger ? "rgba(245,158,11,0.3)" : "rgba(99,179,237,0.3)"}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <AlertTriangle size={18} color={danger ? "#f59e0b" : "#63b3ed"} />
        </div>
        <p
          style={{
            margin: 0,
            fontSize: "14px",
            lineHeight: 1.75,
            color: "rgba(255,255,255,0.85)",
            paddingTop: "7px",
          }}
        >
          {message}
        </p>
      </div>
      <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
        <button
          onClick={onCancel}
          style={{
            padding: "9px 22px",
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
            padding: "9px 22px",
            borderRadius: "10px",
            fontSize: "13px",
            cursor: "pointer",
            background: danger ? "rgba(239,68,68,0.2)" : "rgba(99,179,237,0.2)",
            border: `1px solid ${danger ? "rgba(239,68,68,0.4)" : "rgba(99,179,237,0.4)"}`,
            color: danger ? "#fca5a5" : "#93c5fd",
            fontWeight: 600,
          }}
        >
          {confirmLabel}
        </button>
      </div>
    </div>
  </div>
);

// ─── Lightbox ────────────────────────────────────────────────────────────────
const Lightbox = ({ photos, startIndex, onClose, onRemove }) => {
  const [idx, setIdx] = useState(startIndex);
  const [loaded, setLoaded] = useState(false);

  const prev = useCallback(() => {
    setLoaded(false);
    setIdx((i) => (i - 1 + photos.length) % photos.length);
  }, [photos.length]);
  const next = useCallback(() => {
    setLoaded(false);
    setIdx((i) => (i + 1) % photos.length);
  }, [photos.length]);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === "ArrowLeft") prev();
      if (e.key === "ArrowRight") next();
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [prev, next, onClose]);

  const photo = photos[idx];

  // Thumbnail strip: show a 9-wide window centred on current index
  const stripStart = Math.max(0, idx - 4);
  const stripPhotos = photos.slice(stripStart, stripStart + 9);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 5000,
        background: "rgba(0,0,0,0.94)",
        backdropFilter: "blur(32px)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      {/* Top bar */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          padding: "16px 22px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.65), transparent)",
          zIndex: 1,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <span
          style={{
            fontSize: "12px",
            fontWeight: 600,
            color: "rgba(255,255,255,0.38)",
            letterSpacing: "0.1em",
          }}
        >
          {idx + 1} <span style={{ opacity: 0.4 }}>/</span> {photos.length}
        </span>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {onRemove && (
            <button
              onClick={() => {
                onRemove(photo.photo_id);
                photos.length <= 1 ? onClose() : next();
              }}
              style={{
                padding: "7px 14px",
                borderRadius: "8px",
                fontSize: "12px",
                fontWeight: 600,
                cursor: "pointer",
                background: "rgba(239,68,68,0.14)",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "#fca5a5",
                display: "flex",
                alignItems: "center",
                gap: "5px",
              }}
            >
              <Trash2 size={11} /> Remove
            </button>
          )}
          <button
            onClick={onClose}
            style={{
              width: 32,
              height: 32,
              borderRadius: "8px",
              cursor: "pointer",
              background: "rgba(255,255,255,0.07)",
              border: "1px solid rgba(255,255,255,0.13)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <X size={15} color="rgba(255,255,255,0.65)" />
          </button>
        </div>
      </div>

      {/* Main image */}
      <div
        className="lumeo-lightbox-enter"
        style={{ position: "relative", maxWidth: "88vw", maxHeight: "78vh" }}
        onClick={(e) => e.stopPropagation()}
      >
        {!loaded && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: "50%",
                border: "2px solid rgba(255,255,255,0.12)",
                borderTop: "2px solid rgba(255,255,255,0.65)",
                animation: "lumeo-spinSlow 0.75s linear infinite",
              }}
            />
          </div>
        )}
        <img
          key={photoSrc(photo)}
          src={photoSrc(photo)}
          alt=""
          onLoad={() => setLoaded(true)}
          style={{
            maxWidth: "88vw",
            maxHeight: "78vh",
            borderRadius: "12px",
            objectFit: "contain",
            display: "block",
            boxShadow: "0 36px 90px rgba(0,0,0,0.85)",
            opacity: loaded ? 1 : 0,
            transition: "opacity 0.28s",
          }}
        />
      </div>

      {/* Arrow nav */}
      {photos.length > 1 &&
        [
          {
            action: prev,
            icon: <ChevronLeft size={22} />,
            pos: { left: "20px" },
          },
          {
            action: next,
            icon: <ChevronRight size={22} />,
            pos: { right: "20px" },
          },
        ].map(({ action, icon, pos }, i) => (
          <button
            key={i}
            onClick={(e) => {
              e.stopPropagation();
              action();
            }}
            style={{
              position: "absolute",
              top: "50%",
              transform: "translateY(-50%)",
              ...pos,
              width: 46,
              height: 46,
              borderRadius: "50%",
              cursor: "pointer",
              background: "rgba(255,255,255,0.07)",
              border: "1px solid rgba(255,255,255,0.14)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "rgba(255,255,255,0.75)",
              transition: "all 0.18s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(255,255,255,0.17)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(255,255,255,0.07)";
            }}
          >
            {icon}
          </button>
        ))}

      {/* Thumbnail strip */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "18px 24px",
          background: "linear-gradient(to top, rgba(0,0,0,0.65), transparent)",
          display: "flex",
          gap: "6px",
          justifyContent: "center",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {stripPhotos.map((p, i) => {
          const realIdx = stripStart + i;
          const active = realIdx === idx;
          return (
            <div
              key={p.photo_id}
              onClick={() => {
                setLoaded(false);
                setIdx(realIdx);
              }}
              style={{
                width: 42,
                height: 42,
                borderRadius: "6px",
                overflow: "hidden",
                flexShrink: 0,
                cursor: "pointer",
                transition: "all 0.2s",
                border: `2px solid ${active ? "rgba(255,255,255,0.75)" : "rgba(255,255,255,0.12)"}`,
                opacity: active ? 1 : 0.5,
                transform: active ? "scale(1.12)" : "scale(1)",
              }}
            >
              <img
                src={photoSrc(p)}
                alt=""
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ─── Cluster Detail Panel ────────────────────────────────────────────────────
export const ClusterDetail = ({
  cluster,
  colorAccent,
  onClose,
  onRemovePhoto,
  onRename,
}) => {
  const [editing, setEditing] = useState(false);
  const [nameValue, setNameValue] = useState(cluster.name);
  const [confirmRemove, setConfirmRemove] = useState(null);
  const [bulkMode, setBulkMode] = useState(false);
  const [selected, setSelected] = useState(new Set());
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [lightboxIdx, setLightboxIdx] = useState(null);
  const [sortOrder, setSortOrder] = useState("default");
  const [imgErrors, setImgErrors] = useState(new Set());

  useEffect(() => {
    setNameValue(cluster.name);
  }, [cluster.name]);
  useEffect(() => {
    setSelected(new Set());
    setBulkMode(false);
  }, [cluster.cluster_id]);

  const submitRename = () => {
    if (nameValue.trim() && nameValue.trim() !== cluster.name)
      onRename(cluster.cluster_id, nameValue.trim());
    setEditing(false);
  };

  const toggleSelect = (id) =>
    setSelected((prev) => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  const selectAll = () =>
    setSelected(new Set(cluster.photos.map((p) => p.photo_id)));
  const clearSel = () => setSelected(new Set());

  const sortedPhotos = [...cluster.photos].sort((a, b) => {
    if (sortOrder === "default") return 0;
    const na = (a.filename || a.path || "").toLowerCase();
    const nb = (b.filename || b.path || "").toLowerCase();
    return sortOrder === "asc" ? na.localeCompare(nb) : nb.localeCompare(na);
  });

  const SortIcon =
    sortOrder === "asc"
      ? SortAsc
      : sortOrder === "desc"
        ? SortDesc
        : ArrowUpDown;
  const cycleSortOrder = () =>
    setSortOrder((s) =>
      s === "default" ? "asc" : s === "asc" ? "desc" : "default",
    );

  return (
    <>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          height: "calc(100vh - 140px)",
          background: "rgba(10,10,20,0.92)",
          backdropFilter: "blur(28px)",
          border: `1px solid ${colorAccent.border}`,
          borderRadius: "20px",
          overflow: "hidden",
          boxShadow: `0 0 60px ${colorAccent.glow}, 0 24px 48px rgba(0,0,0,0.5)`,
        }}
      >
        {/* Animated accent bar */}
        <div
          className="lumeo-shimmer-bar"
          style={{ height: 3, "--dot": colorAccent.dot }}
        />

        {/* Header */}
        <div
          style={{
            padding: "16px 18px",
            borderBottom: "1px solid rgba(255,255,255,0.07)",
            display: "flex",
            alignItems: "center",
            gap: "12px",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              width: 46,
              height: 46,
              borderRadius: "13px",
              overflow: "hidden",
              flexShrink: 0,
              border: `1.5px solid ${colorAccent.border}`,
              boxShadow: `0 0 14px ${colorAccent.glow}`,
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

          <div style={{ flex: 1, minWidth: 0 }}>
            {editing ? (
              <div
                style={{ display: "flex", gap: "5px", alignItems: "center" }}
              >
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
                    border: `1px solid ${colorAccent.border}`,
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
                  <Check size={12} color="#4ade80" />
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
                  <X size={12} color="#f87171" />
                </button>
              </div>
            ) : (
              <div
                style={{ display: "flex", alignItems: "center", gap: "7px" }}
              >
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
                    width: 22,
                    height: 22,
                    borderRadius: "5px",
                    border: "1px solid rgba(255,255,255,0.13)",
                    background: "rgba(255,255,255,0.05)",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Edit2 size={10} color="rgba(255,255,255,0.45)" />
                </button>
              </div>
            )}
            <div
              style={{
                fontSize: "11px",
                color: "rgba(255,255,255,0.3)",
                marginTop: "3px",
                display: "flex",
                gap: "8px",
              }}
            >
              <span style={{ color: colorAccent.dot, fontWeight: 600 }}>
                {cluster.photos.length}
              </span>{" "}
              photos
              <span style={{ opacity: 0.4 }}>·</span>
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
              width: 28,
              height: 28,
              borderRadius: "7px",
              border: "1px solid rgba(255,255,255,0.1)",
              background: "rgba(255,255,255,0.04)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <X size={13} color="rgba(255,255,255,0.45)" />
          </button>
        </div>

        {/* Toolbar */}
        <div
          style={{
            padding: "9px 13px",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            display: "flex",
            alignItems: "center",
            gap: "7px",
            flexShrink: 0,
            background: "rgba(255,255,255,0.018)",
          }}
        >
          <button
            onClick={() => {
              setBulkMode((b) => !b);
              clearSel();
            }}
            style={{
              padding: "5px 11px",
              borderRadius: "7px",
              fontSize: "11px",
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "5px",
              transition: "all 0.2s",
              background: bulkMode
                ? "rgba(252,100,113,0.14)"
                : "rgba(255,255,255,0.05)",
              border: `1px solid ${bulkMode ? "rgba(252,100,113,0.35)" : "rgba(255,255,255,0.1)"}`,
              color: bulkMode ? "#fc6471" : "rgba(255,255,255,0.5)",
            }}
          >
            {bulkMode ? <CheckSquare size={11} /> : <Square size={11} />}
            {bulkMode ? "Exit" : "Select"}
          </button>

          {bulkMode && (
            <>
              <button
                onClick={selectAll}
                style={{
                  padding: "5px 9px",
                  borderRadius: "7px",
                  fontSize: "11px",
                  cursor: "pointer",
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.09)",
                  color: "rgba(255,255,255,0.48)",
                }}
              >
                All
              </button>
              <button
                onClick={clearSel}
                style={{
                  padding: "5px 9px",
                  borderRadius: "7px",
                  fontSize: "11px",
                  cursor: "pointer",
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.09)",
                  color: "rgba(255,255,255,0.48)",
                }}
              >
                None
              </button>
              {selected.size > 0 && (
                <button
                  onClick={() => setConfirmBulk(true)}
                  style={{
                    padding: "5px 11px",
                    borderRadius: "7px",
                    fontSize: "11px",
                    fontWeight: 700,
                    cursor: "pointer",
                    background: "rgba(239,68,68,0.17)",
                    border: "1px solid rgba(239,68,68,0.33)",
                    color: "#fca5a5",
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                  }}
                >
                  <Trash2 size={10} /> Remove {selected.size}
                </button>
              )}
            </>
          )}

          <div style={{ flex: 1 }} />

          <button
            onClick={cycleSortOrder}
            title={`Sort: ${sortOrder}`}
            style={{
              width: 28,
              height: 28,
              borderRadius: "7px",
              cursor: "pointer",
              transition: "all 0.18s",
              background:
                sortOrder !== "default"
                  ? "rgba(99,179,237,0.12)"
                  : "rgba(255,255,255,0.05)",
              border: `1px solid ${sortOrder !== "default" ? "rgba(99,179,237,0.3)" : "rgba(255,255,255,0.09)"}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color:
                sortOrder !== "default" ? "#63b3ed" : "rgba(255,255,255,0.38)",
            }}
          >
            <SortIcon size={13} />
          </button>
        </div>

        {/* Photo grid */}
        <div
          style={{ flex: 1, overflowY: "auto", padding: "13px" }}
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
                gap: "10px",
                opacity: 0.3,
              }}
            >
              <Layers size={34} />
              <span style={{ fontSize: "13px" }}>
                No photos in this cluster
              </span>
            </div>
          ) : (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(98px, 1fr))",
                gap: "6px",
              }}
            >
              {sortedPhotos.map((photo, photoIdx) => {
                const src = photoSrc(photo);
                const isSelected = selected.has(photo.photo_id);
                const hasError = imgErrors.has(photo.photo_id);

                return (
                  <div
                    key={photo.photo_id}
                    onClick={() =>
                      bulkMode
                        ? toggleSelect(photo.photo_id)
                        : setLightboxIdx(photoIdx)
                    }
                    style={{
                      position: "relative",
                      aspectRatio: "1",
                      borderRadius: "9px",
                      overflow: "hidden",
                      cursor: "pointer",
                      transition: "transform 0.2s, box-shadow 0.2s",
                      border: isSelected
                        ? "2px solid #63b3ed"
                        : "1px solid rgba(255,255,255,0.07)",
                      boxShadow: isSelected
                        ? "0 0 14px rgba(99,179,237,0.4)"
                        : "none",
                    }}
                    onMouseEnter={(e) => {
                      if (!bulkMode) {
                        e.currentTarget.style.transform = "scale(1.04)";
                        e.currentTarget
                          .querySelectorAll(".photo-ov")
                          .forEach((el) => (el.style.opacity = "1"));
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = "scale(1)";
                      e.currentTarget
                        .querySelectorAll(".photo-ov")
                        .forEach((el) => (el.style.opacity = "0"));
                    }}
                  >
                    {hasError ? (
                      <div
                        style={{
                          width: "100%",
                          height: "100%",
                          background: "rgba(255,255,255,0.03)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <Camera size={15} color="rgba(255,255,255,0.18)" />
                      </div>
                    ) : (
                      <img
                        src={src}
                        alt=""
                        onError={() =>
                          setImgErrors((p) => new Set([...p, photo.photo_id]))
                        }
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                          display: "block",
                        }}
                      />
                    )}

                    {/* Hover overlay */}
                    {!bulkMode && (
                      <div
                        className="photo-ov"
                        style={{
                          position: "absolute",
                          inset: 0,
                          background: "rgba(0,0,0,0.36)",
                          opacity: 0,
                          transition: "opacity 0.18s",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <div
                          style={{
                            width: 28,
                            height: 28,
                            borderRadius: "50%",
                            background: "rgba(255,255,255,0.14)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          <ZoomIn size={13} color="#fff" />
                        </div>
                      </div>
                    )}

                    {/* Bulk checkbox */}
                    {bulkMode && (
                      <div
                        style={{
                          position: "absolute",
                          top: 5,
                          right: 5,
                          width: 17,
                          height: 17,
                          borderRadius: "4px",
                          background: isSelected
                            ? "#63b3ed"
                            : "rgba(0,0,0,0.55)",
                          border: `1.5px solid ${isSelected ? "#63b3ed" : "rgba(255,255,255,0.38)"}`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          transition: "all 0.14s",
                        }}
                      >
                        {isSelected && (
                          <Check size={10} color="#fff" strokeWidth={3} />
                        )}
                      </div>
                    )}

                    {/* Single-delete button */}
                    {!bulkMode && (
                      <button
                        className="photo-ov"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmRemove(photo.photo_id);
                        }}
                        style={{
                          position: "absolute",
                          bottom: 5,
                          right: 5,
                          width: 23,
                          height: 23,
                          borderRadius: "50%",
                          cursor: "pointer",
                          opacity: 0,
                          transition: "opacity 0.18s",
                          border: "1px solid rgba(248,113,113,0.6)",
                          background: "rgba(239,68,68,0.55)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <Trash2 size={10} color="#fff" />
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Lightbox */}
      {lightboxIdx !== null && (
        <Lightbox
          photos={sortedPhotos}
          startIndex={lightboxIdx}
          onClose={() => setLightboxIdx(null)}
          onRemove={(photoId) => onRemovePhoto(cluster.cluster_id, photoId)}
        />
      )}

      {confirmRemove && (
        <ConfirmDialog
          message="Remove this photo from the cluster? The photo won't be deleted."
          confirmLabel="Remove"
          onConfirm={() => {
            onRemovePhoto(cluster.cluster_id, confirmRemove);
            setConfirmRemove(null);
          }}
          onCancel={() => setConfirmRemove(null)}
        />
      )}

      {confirmBulk && (
        <ConfirmDialog
          message={`Remove ${selected.size} selected photo${selected.size > 1 ? "s" : ""} from this cluster? Photos won't be deleted.`}
          confirmLabel={`Remove ${selected.size}`}
          onConfirm={() => {
            selected.forEach((id) => onRemovePhoto(cluster.cluster_id, id));
            setSelected(new Set());
            setBulkMode(false);
            setConfirmBulk(false);
          }}
          onCancel={() => setConfirmBulk(false)}
        />
      )}
    </>
  );
};

// ─── Cluster Card ────────────────────────────────────────────────────────────
export const ClusterCard = ({
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

  // Multi-photo grid preview (up to 4 thumbnails shown on hover)
  const previewPhotos = cluster.photos.slice(0, 4);
  const showGrid = hovered && previewPhotos.length >= 2 && !isActive;

  return (
    <>
      <div
        className="lumeo-card-enter"
        style={{ animationDelay: `${index * 35}ms` }}
      >
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
            transition: "all 0.28s cubic-bezier(0.34,1.56,0.64,1)",
            background: isActive
              ? colorAccent.bg
              : hovered
                ? "rgba(255,255,255,0.055)"
                : "rgba(255,255,255,0.025)",
            border: `1px solid ${isActive ? colorAccent.border : hovered ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.08)"}`,
            transform:
              hovered && !isActive
                ? "translateY(-4px) scale(1.01)"
                : isActive
                  ? "translateY(-5px)"
                  : "none",
            boxShadow: isActive
              ? `0 16px 40px rgba(0,0,0,0.45), 0 0 0 1px ${colorAccent.border}, 0 0 28px ${colorAccent.glow}`
              : hovered
                ? "0 10px 28px rgba(0,0,0,0.35)"
                : "none",
          }}
        >
          {/* Animated accent bar */}
          <div
            className={isActive ? "lumeo-shimmer-bar" : ""}
            style={{
              height: 2.5,
              "--dot": colorAccent.dot,
              background: isActive ? undefined : "transparent",
              transition: "background 0.3s",
            }}
          />

          {/* Thumbnail */}
          <div
            style={{
              width: "100%",
              aspectRatio: "1",
              position: "relative",
              overflow: "hidden",
              background: "rgba(0,0,0,0.35)",
            }}
          >
            {showGrid ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gridTemplateRows:
                    previewPhotos.length >= 3 ? "1fr 1fr" : "1fr",
                  width: "100%",
                  height: "100%",
                  gap: "1px",
                }}
              >
                {previewPhotos.map((p) => (
                  <div key={p.photo_id} style={{ overflow: "hidden" }}>
                    <img
                      src={photoSrc(p)}
                      alt=""
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        display: "block",
                        transform: "scale(1.08)",
                        transition: "transform 0.4s",
                      }}
                    />
                  </div>
                ))}
              </div>
            ) : cluster.thumbnail ? (
              <img
                src={`${BASE_URL}/thumbnails/${cluster.thumbnail}`}
                alt=""
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                  transform: hovered ? "scale(1.07)" : "scale(1)",
                  transition: "transform 0.45s ease",
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
                  size={34}
                  color={colorAccent.dot}
                  style={{ opacity: 0.55 }}
                />
              </div>
            )}

            {/* Photo count badge */}
            <div
              style={{
                position: "absolute",
                bottom: 7,
                right: 7,
                background: "rgba(0,0,0,0.72)",
                backdropFilter: "blur(10px)",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: "7px",
                padding: "3px 7px",
                fontSize: "11px",
                fontWeight: 700,
                color: "#fff",
                display: "flex",
                alignItems: "center",
                gap: "3px",
              }}
            >
              <Camera size={9} /> {cluster.photos.length}
            </div>

            {isMergeSource && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(251,191,36,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span
                  style={{
                    fontSize: "10px",
                    fontWeight: 800,
                    letterSpacing: "0.1em",
                    color: "#fbbf24",
                    background: "rgba(0,0,0,0.65)",
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
          <div style={{ padding: "11px 13px 13px" }}>
            <div
              style={{
                fontWeight: 700,
                fontSize: "13px",
                color: "#fff",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                marginBottom: "2px",
              }}
            >
              {cluster.name}
            </div>
            <div
              style={{
                fontSize: "11px",
                color: "rgba(255,255,255,0.35)",
                marginBottom: mergeMode ? 0 : "11px",
              }}
            >
              {cluster.face_count} face{cluster.face_count !== 1 ? "s" : ""}
            </div>

            {!mergeMode && (
              <div
                style={{ display: "flex", gap: "5px" }}
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => onStartMerge(cluster)}
                  style={{
                    flex: 1,
                    padding: "5px 0",
                    borderRadius: "7px",
                    fontSize: "11px",
                    fontWeight: 500,
                    cursor: "pointer",
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: "rgba(255,255,255,0.55)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "4px",
                    transition: "all 0.18s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "rgba(255,255,255,0.11)";
                    e.currentTarget.style.color = "#fff";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "rgba(255,255,255,0.05)";
                    e.currentTarget.style.color = "rgba(255,255,255,0.55)";
                  }}
                >
                  <Merge size={10} /> Merge
                </button>
                <button
                  onClick={() => setConfirmDelete(true)}
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: "7px",
                    cursor: "pointer",
                    background: "rgba(239,68,68,0.07)",
                    border: "1px solid rgba(239,68,68,0.18)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.18s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "rgba(239,68,68,0.22)";
                    e.currentTarget.style.borderColor = "rgba(239,68,68,0.5)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "rgba(239,68,68,0.07)";
                    e.currentTarget.style.borderColor = "rgba(239,68,68,0.18)";
                  }}
                >
                  <Trash2 size={11} color="#f87171" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {confirmDelete && (
        <ConfirmDialog
          message={`Delete "${cluster.name}"? Face groupings removed. Photos are kept.`}
          confirmLabel="Delete"
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

// ─── Animated count-up hook ──────────────────────────────────────────────────
const useCountUp = (target, duration = 900) => {
  const [count, setCount] = useState(0);
  const raf = useRef(null);
  useEffect(() => {
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const pct = Math.min((ts - start) / duration, 1);
      setCount(Math.round((1 - Math.pow(1 - pct, 3)) * target));
      if (pct < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);
  return count;
};

// ─── Stats Bar ────────────────────────────────────────────────────────────────
const StatPill = ({ label, value, color, maxValue }) => {
  const animated = useCountUp(value);
  const pct = maxValue > 0 ? (value / maxValue) * 100 : 0;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "12px",
        padding: "11px 18px",
        borderRadius: "13px",
        background: "rgba(255,255,255,0.033)",
        border: "1px solid rgba(255,255,255,0.07)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Proportional fill bar */}
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: `${pct}%`,
          background: `linear-gradient(90deg, ${color}1a, transparent)`,
          transition: "width 1.1s cubic-bezier(0.34,1.56,0.64,1)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: color,
          boxShadow: `0 0 8px ${color}`,
          flexShrink: 0,
          position: "relative",
        }}
      />
      <span
        style={{
          fontSize: "21px",
          fontWeight: 800,
          color,
          lineHeight: 1,
          fontVariantNumeric: "tabular-nums",
          position: "relative",
        }}
      >
        {animated}
      </span>
      <span
        style={{
          fontSize: "12px",
          color: "rgba(255,255,255,0.37)",
          lineHeight: 1,
          position: "relative",
        }}
      >
        {label}
      </span>
    </div>
  );
};

export const StatsBar = ({ clusters }) => {
  const totalPhotos = clusters.reduce((s, c) => s + c.photos.length, 0);
  const totalFaces = clusters.reduce((s, c) => s + c.face_count, 0);
  const maxVal = Math.max(clusters.length, totalPhotos, totalFaces, 1);
  return (
    <div
      style={{
        display: "flex",
        gap: "10px",
        marginBottom: "24px",
        flexWrap: "wrap",
      }}
    >
      <StatPill
        label="Clusters"
        value={clusters.length}
        color="#63b3ed"
        maxValue={maxVal}
      />
      <StatPill
        label="Total Photos"
        value={totalPhotos}
        color="#9a75ea"
        maxValue={maxVal}
      />
      <StatPill
        label="Total Faces"
        value={totalFaces}
        color="#48c78e"
        maxValue={maxVal}
      />
    </div>
  );
};
