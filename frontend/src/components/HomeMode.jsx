import {
  Camera,
  MessageSquare,
  Upload,
  ChevronRight,
  Sparkles,
  Users,
} from "lucide-react";

/**
 * HomeMode
 * The landing screen shown on app load.
 *
 * Props:
 *   hasPhotos  – boolean
 *   setAppMode – (mode: 'chat' | 'organize') => void
 *   resetApp   – () => void
 */
const HomeMode = ({ hasPhotos, setAppMode, resetApp }) => (
  <div style={{ maxWidth: "1000px", margin: "0 auto" }}>
    {/* Header */}
    <div
      style={{ textAlign: "center", marginBottom: "48px", marginTop: "40px" }}
    >
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "12px",
          marginBottom: "12px",
          animation: "float 3s ease-in-out infinite",
        }}
      >
        <Camera size={48} strokeWidth={1.5} style={{ color: "#93c5fd" }} />
        <h1
          style={{
            fontSize: "clamp(32px, 8vw, 48px)",
            fontWeight: "600",
            margin: 0,
            background: "linear-gradient(135deg, #ffffff 0%, #93c5fd 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Lumeo
        </h1>
      </div>
      <p style={{ fontSize: "16px", color: "rgba(255,255,255,0.6)" }}>
        AI-Powered Photo Memory Assistant
      </p>
    </div>

    {/* Mode cards */}
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
        gap: "24px",
        marginBottom: "32px",
      }}
    >
      {/* Chat card */}
      <div
        onClick={() =>
          hasPhotos
            ? setAppMode("chat")
            : alert("Please organize photos first!")
        }
        className="glass-card"
        style={{
          borderRadius: "24px",
          padding: "40px",
          cursor: hasPhotos ? "pointer" : "not-allowed",
          opacity: hasPhotos ? 1 : 0.5,
          transition: "all 0.3s",
        }}
        onMouseEnter={(e) =>
          hasPhotos && (e.currentTarget.style.transform = "translateY(-4px)")
        }
        onMouseLeave={(e) =>
          (e.currentTarget.style.transform = "translateY(0)")
        }
      >
        <div
          style={{
            width: "72px",
            height: "72px",
            borderRadius: "20px",
            background: "rgba(147, 197, 253, 0.15)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "24px",
            border: "1px solid rgba(147, 197, 253, 0.3)",
          }}
        >
          <MessageSquare size={36} color="#93c5fd" strokeWidth={1.5} />
        </div>
        <h2
          style={{ fontSize: "24px", marginBottom: "12px", fontWeight: "600" }}
        >
          Chat with Photos
        </h2>
        <p
          style={{
            color: "rgba(255,255,255,0.6)",
            fontSize: "14px",
            lineHeight: "1.6",
            marginBottom: "20px",
          }}
        >
          Search your photo library with natural language. Ask questions like
          "Show me beach photos" or "When was I last with Mom?"
        </p>
        {hasPhotos ? (
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              color: "#93c5fd",
              fontSize: "13px",
              fontWeight: "500",
            }}
          >
            Start Chatting <ChevronRight size={16} />
          </div>
        ) : (
          <div
            style={{
              fontSize: "12px",
              color: "rgba(255,255,255,0.4)",
              fontStyle: "italic",
            }}
          >
            No photos organized yet
          </div>
        )}
      </div>

      {/* Organize card */}
      <div
        onClick={() => setAppMode("organize")}
        className="glass-card"
        style={{
          borderRadius: "24px",
          padding: "40px",
          cursor: "pointer",
          transition: "all 0.3s",
        }}
        onMouseEnter={(e) =>
          (e.currentTarget.style.transform = "translateY(-4px)")
        }
        onMouseLeave={(e) =>
          (e.currentTarget.style.transform = "translateY(0)")
        }
      >
        <div
          style={{
            width: "72px",
            height: "72px",
            borderRadius: "20px",
            background: "rgba(74, 222, 128, 0.15)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "24px",
            border: "1px solid rgba(74, 222, 128, 0.3)",
          }}
        >
          <Upload size={36} color="#4ade80" strokeWidth={1.5} />
        </div>
        <h2
          style={{ fontSize: "24px", marginBottom: "12px", fontWeight: "600" }}
        >
          Organize Photos
        </h2>
        <p
          style={{
            color: "rgba(255,255,255,0.6)",
            fontSize: "14px",
            lineHeight: "1.6",
            marginBottom: "20px",
          }}
        >
          Upload new photos and let AI detect faces, group people, and organize
          your memories automatically.
        </p>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            color: "#4ade80",
            fontSize: "13px",
            fontWeight: "500",
          }}
        >
          Get Started <ChevronRight size={16} />
        </div>
      </div>
    </div>
    {/* Manage Clusters card */}
    <div
      onClick={() => setAppMode("clusters")}
      className="glass-card"
      style={{
        borderRadius: "24px",
        padding: "40px",
        cursor: "pointer",
        transition: "all 0.3s",
      }}
      onMouseEnter={(e) =>
        (e.currentTarget.style.transform = "translateY(-4px)")
      }
      onMouseLeave={(e) => (e.currentTarget.style.transform = "translateY(0)")}
    >
      <div
        style={{
          width: "72px",
          height: "72px",
          borderRadius: "20px",
          background: "rgba(251, 191, 36, 0.15)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "24px",
          border: "1px solid rgba(251, 191, 36, 0.3)",
        }}
      >
        <Users size={36} color="#fbbf24" strokeWidth={1.5} />
      </div>
      <h2 style={{ fontSize: "24px", marginBottom: "12px", fontWeight: "600" }}>
        Manage Clusters
      </h2>
      <p
        style={{
          color: "rgba(255,255,255,0.6)",
          fontSize: "14px",
          lineHeight: "1.6",
          marginBottom: "20px",
        }}
      >
        Rename people, merge duplicate face groups, or remove misclassified
        photos from any cluster.
      </p>
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          color: "#fbbf24",
          fontSize: "13px",
          fontWeight: "500",
        }}
      >
        Open Manager <ChevronRight size={16} />
      </div>
    </div>

    {/* Status banner */}
    {hasPhotos && (
      <div
        className="glass-card"
        style={{
          borderRadius: "20px",
          padding: "24px",
          display: "flex",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <Sparkles size={20} color="#93c5fd" />
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: "500", marginBottom: "4px" }}>
            You have organized photos!
          </div>
          <div style={{ fontSize: "13px", color: "rgba(255,255,255,0.5)" }}>
            Ready to chat and explore your memories
          </div>
        </div>
        <button
          onClick={resetApp}
          className="glass-button"
          style={{
            padding: "8px 16px",
            borderRadius: "10px",
            fontSize: "13px",
            color: "rgba(255,255,255,0.6)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            background: "rgba(239, 68, 68, 0.1)",
          }}
        >
          Reset All Data
        </button>
      </div>
    )}
  </div>
);

export default HomeMode;
