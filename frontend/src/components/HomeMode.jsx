import { useState, useEffect } from "react";
import {
  Camera,
  MessageSquare,
  Upload,
  ChevronRight,
  Sparkles,
  Users,
  Brain,
  Zap,
} from "lucide-react";

const HomeMode = ({ hasPhotos, setAppMode, resetApp }) => {
  const [mounted, setMounted] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 80);
    return () => clearTimeout(t);
  }, []);

  const cards = [
    {
      id: "chat",
      label: "Chat",
      title: "Talk to Your Photos",
      subtitle: "Natural language search",
      description:
        'Ask anything — "Who was at my birthday?" or "Show me all beach sunsets" — and get instant answers.',
      cta: hasPhotos ? "Open Chat" : "No photos yet",
      icon: MessageSquare,
      accentColor: "#93c5fd",
      accentRgb: "147, 197, 253",
      glow: "#1e3a5f",
      locked: !hasPhotos,
      onClick: () =>
        hasPhotos ? setAppMode("chat") : alert("Please organize photos first!"),
    },
    {
      id: "organize",
      label: "Organize",
      title: "Upload & Organize",
      subtitle: "AI face recognition",
      description:
        "Drop in your photos. AI groups people, detects faces, and builds your searchable memory index.",
      cta: "Get Started",
      icon: Upload,
      accentColor: "#4ade80",
      accentRgb: "74, 222, 128",
      glow: "#0f2d1a",
      locked: false,
      onClick: () => setAppMode("organize"),
    },
    {
      id: "clusters",
      label: "People",
      title: "Manage People",
      subtitle: "Refine your clusters",
      description:
        "Rename faces, merge duplicates, or remove misclassified photos. Keep your memory graph accurate.",
      cta: "Open Manager",
      icon: Users,
      accentColor: "#fbbf24",
      accentRgb: "251, 191, 36",
      glow: "#2d1f00",
      locked: false,
      onClick: () => setAppMode("clusters"),
    },
  ];

  return (
    <div
      style={{
        maxWidth: "1100px",
        margin: "0 auto",
        padding: "0 4px",
        fontFamily:
          '"SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <div
        style={{
          textAlign: "center",
          paddingTop: "56px",
          paddingBottom: "52px",
          opacity: mounted ? 1 : 0,
          transform: mounted ? "translateY(0)" : "translateY(18px)",
          transition: "opacity 0.6s ease, transform 0.6s ease",
        }}
      >
        {/* Main title */}
        <h1
          style={{
            fontSize: "clamp(48px, 9vw, 80px)",
            fontWeight: "700",
            letterSpacing: "-0.03em",
            lineHeight: "1",
            margin: "0 0 18px",
            background:
              "linear-gradient(160deg, #ffffff 30%, rgba(147,197,253,0.8) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Lumeo
        </h1>

        {/* Tagline */}
        {/* <p
          style={{
            fontSize: "clamp(15px, 2vw, 18px)",
            color: "rgba(255,255,255,0.45)",
            fontWeight: "400",
            letterSpacing: "0.01em",
            maxWidth: "420px",
            margin: "0 auto",
            lineHeight: "1.6",
          }}
        >
          Your memories, organized and searchable — powered by local AI
        </p> */}
        <p
          style={{
            fontSize: "clamp(16px, 2vw, 18px)",
            color: "rgba(255,255,255,0.65)", 
            fontWeight: "400",
            letterSpacing: "0.02em",
            maxWidth: "480px",
            margin: "12px auto 0",
            lineHeight: "1.7",
          }}
        >
          Relive your moments, not the mess.
        </p>
      </div>

      {/* ── Card Grid ─────────────────────────────────────────────────── */}
      {/*  Layout:  [Chat — large left]  [Organize — top right]           */}
      {/*           [─── Clusters — full width bottom ───]                */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gridTemplateRows: "auto auto",
          gap: "16px",
          opacity: mounted ? 1 : 0,
          transform: mounted ? "translateY(0)" : "translateY(24px)",
          transition: "opacity 0.7s ease 0.15s, transform 0.7s ease 0.15s",
        }}
      >
        {/* ── Chat (tall, left) ──────────────────────────────────────── */}
        <div
          onClick={cards[0].onClick}
          onMouseEnter={() => setHoveredCard("chat")}
          onMouseLeave={() => setHoveredCard(null)}
          style={{
            gridRow: "1 / 2",
            gridColumn: "1 / 2",
            borderRadius: "28px",
            padding: "40px",
            cursor: cards[0].locked ? "not-allowed" : "pointer",
            opacity: cards[0].locked ? 0.45 : 1,
            background: "rgba(255,255,255,0.06)",
            border: `1px solid ${
              hoveredCard === "chat" && !cards[0].locked
                ? "rgba(147,197,253,0.35)"
                : "rgba(255,255,255,0.1)"
            }`,
            backdropFilter: "blur(20px)",
            boxShadow:
              hoveredCard === "chat" && !cards[0].locked
                ? "0 0 40px rgba(147,197,253,0.08), 0 20px 60px rgba(0,0,0,0.4)"
                : "0 8px 32px rgba(0,0,0,0.3)",
            transform:
              hoveredCard === "chat" && !cards[0].locked
                ? "translateY(-6px)"
                : "translateY(0)",
            transition: "all 0.35s cubic-bezier(0.4,0,0.2,1)",
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Background glow blob */}
          <div
            style={{
              position: "absolute",
              top: "-40px",
              right: "-40px",
              width: "180px",
              height: "180px",
              borderRadius: "50%",
              background: "rgba(147,197,253,0.06)",
              filter: "blur(40px)",
              pointerEvents: "none",
            }}
          />

          {/* Icon */}
          <div
            style={{
              width: "56px",
              height: "56px",
              borderRadius: "16px",
              background: "rgba(147,197,253,0.12)",
              border: "1px solid rgba(147,197,253,0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: "28px",
            }}
          >
            <MessageSquare size={28} color="#93c5fd" strokeWidth={1.5} />
          </div>

          {/* Label */}
          <div
            style={{
              fontSize: "11px",
              fontWeight: "600",
              letterSpacing: "0.1em",
              color: "rgba(147,197,253,0.6)",
              textTransform: "uppercase",
              marginBottom: "10px",
            }}
          >
            Chat Interface
          </div>

          <h2
            style={{
              fontSize: "clamp(22px, 3vw, 28px)",
              fontWeight: "650",
              letterSpacing: "-0.02em",
              margin: "0 0 14px",
              color: "#fff",
            }}
          >
            Talk to Your Photos
          </h2>

          <p
            style={{
              color: "rgba(255,255,255,0.5)",
              fontSize: "14px",
              lineHeight: "1.7",
              margin: "0 0 32px",
              maxWidth: "320px",
            }}
          >
            Ask anything in plain English. Find moments, people, and places
            instantly without digging through folders.
          </p>

          {/* Feature chips */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "8px",
              marginBottom: "32px",
            }}
          >
            {["Semantic search", "Person filter", "Conversation memory"].map(
              (f) => (
                <span
                  key={f}
                  style={{
                    padding: "5px 12px",
                    borderRadius: "20px",
                    fontSize: "12px",
                    background: "rgba(147,197,253,0.08)",
                    border: "1px solid rgba(147,197,253,0.18)",
                    color: "rgba(147,197,253,0.7)",
                  }}
                >
                  {f}
                </span>
              ),
            )}
          </div>

          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              color: cards[0].locked ? "rgba(255,255,255,0.3)" : "#93c5fd",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            {cards[0].locked ? "Organize photos first" : "Open Chat"}
            {!cards[0].locked && <ChevronRight size={16} />}
          </div>
        </div>

        {/* ── Organize (top right) ──────────────────────────────────── */}
        <div
          onClick={cards[1].onClick}
          onMouseEnter={() => setHoveredCard("organize")}
          onMouseLeave={() => setHoveredCard(null)}
          style={{
            gridRow: "1 / 2",
            gridColumn: "2 / 3",
            borderRadius: "28px",
            padding: "36px",
            cursor: "pointer",
            background: "rgba(255,255,255,0.06)",
            border: `1px solid ${
              hoveredCard === "organize"
                ? "rgba(74,222,128,0.35)"
                : "rgba(255,255,255,0.1)"
            }`,
            backdropFilter: "blur(20px)",
            boxShadow:
              hoveredCard === "organize"
                ? "0 0 40px rgba(74,222,128,0.07), 0 20px 60px rgba(0,0,0,0.4)"
                : "0 8px 32px rgba(0,0,0,0.3)",
            transform:
              hoveredCard === "organize" ? "translateY(-6px)" : "translateY(0)",
            transition: "all 0.35s cubic-bezier(0.4,0,0.2,1)",
            position: "relative",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              position: "absolute",
              bottom: "-30px",
              left: "-30px",
              width: "140px",
              height: "140px",
              borderRadius: "50%",
              background: "rgba(74,222,128,0.05)",
              filter: "blur(30px)",
              pointerEvents: "none",
            }}
          />

          <div
            style={{
              width: "56px",
              height: "56px",
              borderRadius: "16px",
              background: "rgba(74,222,128,0.12)",
              border: "1px solid rgba(74,222,128,0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: "28px",
            }}
          >
            <Upload size={28} color="#4ade80" strokeWidth={1.5} />
          </div>

          <div
            style={{
              fontSize: "11px",
              fontWeight: "600",
              letterSpacing: "0.1em",
              color: "rgba(74,222,128,0.6)",
              textTransform: "uppercase",
              marginBottom: "10px",
            }}
          >
            Upload & Process
          </div>

          <h2
            style={{
              fontSize: "clamp(20px, 2.5vw, 26px)",
              fontWeight: "650",
              letterSpacing: "-0.02em",
              margin: "0 0 12px",
              color: "#fff",
            }}
          >
            Organize Photos
          </h2>

          <p
            style={{
              color: "rgba(255,255,255,0.5)",
              fontSize: "14px",
              lineHeight: "1.7",
              flex: 1,
              marginBottom: "28px",
            }}
          >
            Upload a batch of photos and let AI detect faces, group people, and
            build your searchable memory index automatically.
          </p>

          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              color: "#4ade80",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            Get Started <ChevronRight size={16} />
          </div>
        </div>

        {/* ── Clusters — full width ─────────────────────────────────── */}
        <div
          onClick={cards[2].onClick}
          onMouseEnter={() => setHoveredCard("clusters")}
          onMouseLeave={() => setHoveredCard(null)}
          style={{
            gridRow: "2 / 3",
            gridColumn: "1 / 3",
            borderRadius: "28px",
            padding: "36px 40px",
            cursor: "pointer",
            background: "rgba(255,255,255,0.05)",
            border: `1px solid ${
              hoveredCard === "clusters"
                ? "rgba(251,191,36,0.3)"
                : "rgba(255,255,255,0.09)"
            }`,
            backdropFilter: "blur(20px)",
            boxShadow:
              hoveredCard === "clusters"
                ? "0 0 40px rgba(251,191,36,0.06), 0 12px 40px rgba(0,0,0,0.3)"
                : "0 8px 32px rgba(0,0,0,0.25)",
            transform:
              hoveredCard === "clusters" ? "translateY(-4px)" : "translateY(0)",
            transition: "all 0.35s cubic-bezier(0.4,0,0.2,1)",
            display: "flex",
            alignItems: "center",
            gap: "32px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "50%",
              right: "60px",
              transform: "translateY(-50%)",
              width: "200px",
              height: "200px",
              borderRadius: "50%",
              background: "rgba(251,191,36,0.04)",
              filter: "blur(50px)",
              pointerEvents: "none",
            }}
          />

          {/* Icon */}
          <div
            style={{
              width: "60px",
              height: "60px",
              borderRadius: "18px",
              background: "rgba(251,191,36,0.1)",
              border: "1px solid rgba(251,191,36,0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Users size={30} color="#fbbf24" strokeWidth={1.5} />
          </div>

          {/* Text block */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: "11px",
                fontWeight: "600",
                letterSpacing: "0.1em",
                color: "rgba(251,191,36,0.55)",
                textTransform: "uppercase",
                marginBottom: "6px",
              }}
            >
              People Manager
            </div>
            <h2
              style={{
                fontSize: "clamp(18px, 2vw, 22px)",
                fontWeight: "650",
                letterSpacing: "-0.02em",
                margin: "0 0 6px",
                color: "#fff",
              }}
            >
              Manage Clusters
            </h2>
            <p
              style={{
                color: "rgba(255,255,255,0.45)",
                fontSize: "14px",
                lineHeight: "1.6",
                margin: 0,
              }}
            >
              Rename people, merge duplicate face groups, or remove
              misclassified photos from any cluster.
            </p>
          </div>

          {/* Right CTA */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              color: "#fbbf24",
              fontSize: "14px",
              fontWeight: "500",
              flexShrink: 0,
            }}
          >
            Open Manager <ChevronRight size={16} />
          </div>
        </div>
      </div>

      {/* ── Status / Reset footer ─────────────────────────────────────── */}
      {hasPhotos && (
        <div
          style={{
            marginTop: "16px",
            padding: "18px 28px",
            borderRadius: "20px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(16px)",
            display: "flex",
            alignItems: "center",
            gap: "14px",
            opacity: mounted ? 1 : 0,
            transition: "opacity 0.6s ease 0.35s",
          }}
        >
          <Sparkles size={16} color="rgba(147,197,253,0.7)" />
          <div style={{ flex: 1 }}>
            <span
              style={{
                fontSize: "13px",
                color: "rgba(255,255,255,0.55)",
                letterSpacing: "0.01em",
              }}
            >
              Photos organized and indexed — ready to chat
            </span>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              resetApp();
            }}
            style={{
              padding: "7px 16px",
              borderRadius: "10px",
              fontSize: "12px",
              fontWeight: "500",
              color: "rgba(239,68,68,0.7)",
              border: "1px solid rgba(239,68,68,0.2)",
              background: "rgba(239,68,68,0.07)",
              cursor: "pointer",
              letterSpacing: "0.01em",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(239,68,68,0.15)";
              e.currentTarget.style.color = "rgba(239,68,68,0.9)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(239,68,68,0.07)";
              e.currentTarget.style.color = "rgba(239,68,68,0.7)";
            }}
          >
            Reset All Data
          </button>
        </div>
      )}

      {/* ── Bottom spacer ─────────────────────────────────────────────── */}
      <div style={{ height: "48px" }} />
    </div>
  );
};

export default HomeMode;
