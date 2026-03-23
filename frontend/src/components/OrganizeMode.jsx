import React from "react";
import {
  Camera,
  Upload,
  Users,
  FolderOpen,
  CheckCircle,
  AlertCircle,
  Loader,
  ChevronLeft,
  Sparkles,
} from "lucide-react";
import { BASE_URL } from "../constants/config";

// ─── Step indicator ────────────────────────────────────────────────────────────

const STEPS = ["upload", "process", "label", "complete"];

const StepBar = ({ currentStep }) => {
  const normalised =
    currentStep === "processing"
      ? "process"
      : currentStep === "organize"
        ? "complete"
        : currentStep;

  const activeIdx = STEPS.indexOf(normalised);

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        marginBottom: "32px",
        gap: "12px",
        overflowX: "auto",
      }}
    >
      {STEPS.map((step, idx) => {
        const isActive = idx === activeIdx;
        const isCompleted = activeIdx > idx;

        return (
          <React.Fragment key={step}>
            <div
              className="glass-button"
              style={{
                padding: "10px 20px",
                borderRadius: "12px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "13px",
                fontWeight: "500",
                background: isActive
                  ? "rgba(147, 197, 253, 0.2)"
                  : isCompleted
                    ? "rgba(74, 222, 128, 0.15)"
                    : "rgba(255, 255, 255, 0.08)",
                borderColor: isActive
                  ? "rgba(147, 197, 253, 0.4)"
                  : isCompleted
                    ? "rgba(74, 222, 128, 0.3)"
                    : "rgba(255, 255, 255, 0.15)",
                color: isActive
                  ? "#93c5fd"
                  : isCompleted
                    ? "#4ade80"
                    : "rgba(255,255,255,0.5)",
              }}
            >
              <div
                style={{
                  width: "20px",
                  height: "20px",
                  borderRadius: "50%",
                  border: "2px solid",
                  borderColor: "currentColor",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "11px",
                }}
              >
                {idx + 1}
              </div>
              <span style={{ textTransform: "capitalize" }}>{step}</span>
            </div>
            {idx < STEPS.length - 1 && (
              <div
                style={{
                  width: "24px",
                  height: "2px",
                  background: isCompleted
                    ? "rgba(74, 222, 128, 0.3)"
                    : "rgba(255, 255, 255, 0.1)",
                  marginTop: "18px",
                }}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};

// ─── Individual step panels ────────────────────────────────────────────────────

const UploadStep = ({ processing, onUpload }) => (
  <div
    className="glass-card"
    style={{ borderRadius: "24px", padding: "60px", textAlign: "center" }}
  >
    <div
      style={{
        width: "80px",
        height: "80px",
        margin: "0 auto 24px",
        borderRadius: "20px",
        background: "rgba(147, 197, 253, 0.1)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        border: "1px solid rgba(147, 197, 253, 0.2)",
      }}
    >
      <Upload size={40} color="#93c5fd" strokeWidth={1.5} />
    </div>
    <h2 style={{ fontSize: "28px", marginBottom: "12px" }}>Upload Photos</h2>
    <p style={{ color: "rgba(255,255,255,0.6)", marginBottom: "32px" }}>
      Select multiple photos to organize
    </p>
    <label>
      <input
        type="file"
        multiple
        accept="image/*"
        onChange={onUpload}
        style={{ display: "none" }}
        disabled={processing}
      />
      <div
        className="glass-button"
        style={{
          padding: "16px 32px",
          borderRadius: "16px",
          cursor: "pointer",
          display: "inline-flex",
          alignItems: "center",
          gap: "12px",
          fontSize: "16px",
          fontWeight: "500",
        }}
      >
        <Upload size={20} />
        {processing ? "Uploading..." : "Choose Photos"}
      </div>
    </label>
  </div>
);

const ProcessStep = ({ photosCount, processing, onProcess }) => (
  <div
    className="glass-card"
    style={{ borderRadius: "24px", padding: "60px", textAlign: "center" }}
  >
    <h2 style={{ fontSize: "28px", marginBottom: "12px" }}>Ready to Process</h2>
    <p style={{ color: "rgba(255,255,255,0.6)", marginBottom: "32px" }}>
      {photosCount} photos uploaded
    </p>
    <button
      onClick={onProcess}
      disabled={processing}
      className="glass-button"
      style={{
        padding: "16px 32px",
        borderRadius: "16px",
        border: "none",
        cursor: "pointer",
        display: "inline-flex",
        alignItems: "center",
        gap: "12px",
        fontSize: "16px",
        fontWeight: "500",
        color: "#ffffff",
      }}
    >
      <Users size={20} /> Start Face Detection
    </button>
  </div>
);

const ProcessingStep = () => (
  <div
    className="glass-card"
    style={{ borderRadius: "24px", padding: "60px", textAlign: "center" }}
  >
    <Loader
      size={60}
      color="#93c5fd"
      strokeWidth={1.5}
      style={{ margin: "0 auto 24px", animation: "spin 1s linear infinite" }}
    />
    <h2 style={{ fontSize: "28px", marginBottom: "12px" }}>Processing...</h2>
    <p style={{ color: "rgba(255,255,255,0.6)" }}>
      Detecting and grouping faces
    </p>
  </div>
);

const LabelStep = ({
  clusters,
  stats,
  processing,
  hoveredCluster,
  setHoveredCluster,
  onViewCluster,
  onRenameCluster,
  onFinish,
}) => (
  <div className="glass-card" style={{ borderRadius: "24px", padding: "40px" }}>
    <div style={{ textAlign: "center", marginBottom: "32px" }}>
      <h2 style={{ fontSize: "28px", marginBottom: "12px" }}>Label People</h2>
      <p style={{ color: "rgba(255,255,255,0.6)" }}>
        Found {clusters.length} people ({stats.total_faces} faces)
      </p>
    </div>

    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gap: "16px",
        marginBottom: "32px",
      }}
    >
      {clusters
        .filter((c) => c.photos && c.photos.length > 0)
        .map((cluster) => (
          <div
            key={cluster.cluster_id}
            className="glass-card"
            style={{
              borderRadius: "20px",
              padding: "16px",
              cursor: "pointer",
              transition: "all 0.3s",
              transform:
                hoveredCluster === cluster.cluster_id
                  ? "translateY(-4px)"
                  : "none",
              borderColor:
                hoveredCluster === cluster.cluster_id
                  ? "rgba(147, 197, 253, 0.4)"
                  : "rgba(255, 255, 255, 0.18)",
            }}
            onMouseEnter={() => setHoveredCluster(cluster.cluster_id)}
            onMouseLeave={() => setHoveredCluster(null)}
            onClick={() => onViewCluster(cluster)}
          >
            <div
              style={{
                width: "100%",
                aspectRatio: "1",
                background: "rgba(0, 0, 0, 0.3)",
                borderRadius: "16px",
                marginBottom: "16px",
                overflow: "hidden",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {cluster.thumbnail ? (
                <img
                  src={`${BASE_URL}/thumbnails/${cluster.thumbnail}`}
                  alt="Face"
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              ) : (
                <Users size={48} color="rgba(255,255,255,0.3)" />
              )}
            </div>
            <input
              type="text"
              value={cluster.name}
              onChange={(e) => {
                e.stopPropagation();
                onRenameCluster(cluster.cluster_id, e.target.value);
              }}
              onClick={(e) => e.stopPropagation()}
              className="glass-input"
              style={{
                width: "100%",
                padding: "12px",
                borderRadius: "12px",
                textAlign: "center",
                boxSizing: "border-box",
              }}
              placeholder="Enter name..."
            />
            <p
              style={{
                textAlign: "center",
                fontSize: "13px",
                color: "rgba(255,255,255,0.5)",
                margin: "12px 0 0",
              }}
            >
              {cluster.photos.length} photos
            </p>
          </div>
        ))}
    </div>

    <div style={{ display: "flex", justifyContent: "center" }}>
      <button
        onClick={onFinish}
        disabled={processing}
        className="glass-button"
        style={{
          padding: "16px 32px",
          borderRadius: "16px",
          border: "none",
          cursor: "pointer",
          display: "inline-flex",
          alignItems: "center",
          gap: "12px",
          fontSize: "16px",
          fontWeight: "500",
          color: "#ffffff",
          background: "rgba(74, 222, 128, 0.15)",
          borderColor: "rgba(74, 222, 128, 0.3)",
        }}
      >
        <Sparkles size={20} /> Finish Organization
      </button>
    </div>
  </div>
);

const OrganizingStep = () => (
  <div
    className="glass-card"
    style={{ borderRadius: "24px", padding: "60px", textAlign: "center" }}
  >
    <FolderOpen
      size={60}
      color="#93c5fd"
      strokeWidth={1.5}
      style={{ margin: "0 auto 24px", animation: "pulse 2s infinite" }}
    />
    <h2 style={{ fontSize: "28px", marginBottom: "12px" }}>Preparing AI...</h2>
    <p style={{ color: "rgba(255,255,255,0.6)" }}>
      Indexing your memories for chat
    </p>
  </div>
);

// ─── Main OrganizeMode ─────────────────────────────────────────────────────────

/**
 * OrganizeMode
 * Props:
 *   currentStep, photos, clusters, stats, processing, error
 *   hoveredCluster, setHoveredCluster
 *   handlers: handleFileUpload, processPhotos, organizePhotos,
 *             updateClusterName, viewClusterPhotos
 *   setAppMode
 */
const OrganizeMode = ({
  currentStep,
  photos,
  clusters,
  stats,
  processing,
  error,
  hoveredCluster,
  setHoveredCluster,
  handleFileUpload,
  processPhotos,
  organizePhotos,
  updateClusterName,
  viewClusterPhotos,
  setAppMode,
}) => {
  const safeClusters = Array.isArray(clusters) ? clusters : [];

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
      {/* Back button */}
      <button
        onClick={() => setAppMode("home")}
        className="glass-button"
        style={{
          marginBottom: "20px",
          padding: "10px 16px",
          borderRadius: "12px",
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          fontSize: "14px",
          color: "rgba(255,255,255,0.6)",
        }}
      >
        <ChevronLeft size={16} /> Back to Home
      </button>

      {/* Page title */}
      {/* <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "12px",
            marginBottom: "12px",
            animation: "float 3s ease-in-out infinite",
          }}
        >
          <Camera size={36} strokeWidth={1.5} style={{ color: "#93c5fd" }} />
          <h1
            style={{
              fontSize: "clamp(24px, 6vw, 36px)",
              fontWeight: "600",
              margin: 0,
              background: "linear-gradient(135deg, #ffffff 0%, #93c5fd 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Photo Organizer
          </h1>
        </div>
        <p style={{ fontSize: "14px", color: "rgba(255,255,255,0.6)" }}>
          AI-powered face recognition & organization
        </p>
      </div> */}
      {/* Page title */}
      <div
        style={{
          textAlign: "center",
          marginBottom: "40px",
          position: "relative",
        }}
      >
        {/* Ambient glow behind title */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "320px",
            height: "80px",
            background:
              "radial-gradient(ellipse, rgba(147, 197, 253, 0.15) 0%, transparent 70%)",
            pointerEvents: "none",
            filter: "blur(20px)",
          }}
        />

        {/* AI badge pill */}
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            padding: "5px 14px",
            borderRadius: "999px",
            background: "rgba(147, 197, 253, 0.08)",
            border: "1px solid rgba(147, 197, 253, 0.25)",
            marginBottom: "16px",
            fontSize: "11px",
            fontWeight: "600",
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            color: "#93c5fd",
          }}
        >
          <span
            style={{
              width: "6px",
              height: "6px",
              borderRadius: "50%",
              background: "#93c5fd",
              boxShadow: "0 0 6px #93c5fd",
              animation: "pulse 2s ease-in-out infinite",
            }}
          />
          Lumeo
        </div>

        {/* Main title row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "16px",
            marginBottom: "14px",
            animation: "float 3s ease-in-out infinite",
          }}
        >
          {/* Icon container with layered rings */}
          <div style={{ position: "relative", flexShrink: 0 }}>
            <div
              style={{
                position: "absolute",
                inset: "-8px",
                borderRadius: "50%",
                border: "1px solid rgba(147, 197, 253, 0.15)",
                animation: "spin 8s linear infinite",
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: "-4px",
                borderRadius: "50%",
                border: "1px dashed rgba(147, 197, 253, 0.2)",
                animation: "spin 5s linear infinite reverse",
              }}
            />
            <div
              style={{
                width: "52px",
                height: "52px",
                borderRadius: "16px",
                background:
                  "linear-gradient(135deg, rgba(147, 197, 253, 0.15), rgba(147, 197, 253, 0.05))",
                border: "1px solid rgba(147, 197, 253, 0.3)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow:
                  "0 0 24px rgba(147, 197, 253, 0.1), inset 0 1px 0 rgba(255,255,255,0.1)",
              }}
            >
              <Camera
                size={26}
                strokeWidth={1.5}
                style={{ color: "#93c5fd" }}
              />
            </div>
          </div>

          {/* Title text */}
          <div>
            <h1
              style={{
                fontSize: "clamp(28px, 6vw, 42px)",
                fontWeight: "700",
                margin: 0,
                lineHeight: 1.1,
                letterSpacing: "-0.02em",
                background:
                  "linear-gradient(135deg, #ffffff 0%, #bfdbfe 50%, #93c5fd 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Photo Organizer
            </h1>
          </div>
        </div>

        {/* Subtitle with decorative lines */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "12px",
          }}
        >
          <div
            style={{
              width: "40px",
              height: "1px",
              background: "rgba(255,255,255,0.1)",
            }}
          />
          <p
            style={{
              fontSize: "13px",
              color: "rgba(255,255,255,0.45)",
              margin: 0,
              letterSpacing: "0.04em",
            }}
          >
            Face recognition & intelligent organization
          </p>
          <div
            style={{
              width: "40px",
              height: "1px",
              background: "rgba(255,255,255,0.1)",
            }}
          />
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div
          className="glass-card"
          style={{
            padding: "16px",
            borderRadius: "16px",
            marginBottom: "24px",
            borderColor: "rgba(239, 68, 68, 0.3)",
            background: "rgba(239, 68, 68, 0.1)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <AlertCircle size={20} color="#ef4444" />
            <span style={{ fontSize: "14px", color: "#fca5a5" }}>{error}</span>
          </div>
        </div>
      )}

      <StepBar currentStep={currentStep} />

      {/* Step panels */}
      {currentStep === "upload" && (
        <UploadStep processing={processing} onUpload={handleFileUpload} />
      )}
      {currentStep === "process" && (
        <ProcessStep
          photosCount={photos.length}
          processing={processing}
          onProcess={processPhotos}
        />
      )}
      {currentStep === "processing" && <ProcessingStep />}
      {currentStep === "label" && (
        <LabelStep
          clusters={safeClusters}
          stats={stats}
          processing={processing}
          hoveredCluster={hoveredCluster}
          setHoveredCluster={setHoveredCluster}
          onViewCluster={viewClusterPhotos}
          onRenameCluster={updateClusterName}
          onFinish={organizePhotos}
        />
      )}
      {currentStep === "organize" && <OrganizingStep />}
    </div>
  );
};

export default OrganizeMode;
