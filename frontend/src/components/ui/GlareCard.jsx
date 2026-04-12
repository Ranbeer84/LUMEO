import { useRef } from "react";

/**
 * GlareCard — holographic tilt + foil shimmer card
 * Pure inline-styles, zero Tailwind dependency.
 *
 * Props:
 *   children     – card content
 *   width        – CSS string (default "100%")
 *   aspectRatio  – CSS string (default undefined → height driven by content)
 *   borderRadius – CSS string (default "24px")
 *   background   – CSS string (default "rgba(14,14,26,0.85)")
 *   style        – extra styles merged onto the outer wrapper
 *   innerStyle   – extra styles merged onto the inner content div
 */
export const GlareCard = ({
  children,
  width = "100%",
  aspectRatio,
  borderRadius = "24px",
  background = "rgba(14,14,26,0.85)",
  style = {},
  innerStyle = {},
}) => {
  const isPointerInside = useRef(false);
  const refElement = useRef(null);
  const state = useRef({
    glare:      { x: 50, y: 50 },
    background: { x: 50, y: 50 },
    rotate:     { x: 0,  y: 0  },
  });

  // ── CSS-variable helpers ───────────────────────────────────────────────────
  const set = (k, v) => refElement.current?.style.setProperty(k, v);
  const del = (k)    => refElement.current?.style.removeProperty(k);

  const applyState = () => {
    const { background: bg, rotate, glare } = state.current;
    set("--m-x",  `${glare.x}%`);
    set("--m-y",  `${glare.y}%`);
    set("--r-x",  `${rotate.x}deg`);
    set("--r-y",  `${rotate.y}deg`);
    set("--bg-x", `${bg.x}%`);
    set("--bg-y", `${bg.y}%`);
  };

  // ── Shimmer / foil layer backgrounds ─────────────────────────────────────
  const foilSvg = `url("data:image/svg+xml,%3Csvg width='26' height='26' viewBox='0 0 26 26' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M2.99994 3.419C2.99994 3.419 21.6142 7.43646 22.7921 12.153C23.97 16.8695 3.41838 23.0306 3.41838 23.0306' stroke='white' stroke-width='5' stroke-miterlimit='3.86874' stroke-linecap='round' style='mix-blend-mode:darken'/%3E%3C/svg%3E")`;

  const shimmerStyle = {
    "--step":    "5%",
    "--pattern": `${foilSvg} center/100% no-repeat`,
    "--rainbow": `repeating-linear-gradient(
      0deg,
      rgb(255,119,115)      calc(var(--step) * 1),
      rgba(255,237,95,1)    calc(var(--step) * 2),
      rgba(168,255,95,1)    calc(var(--step) * 3),
      rgba(131,255,247,1)   calc(var(--step) * 4),
      rgba(120,148,255,1)   calc(var(--step) * 5),
      rgb(216,117,255)      calc(var(--step) * 6),
      rgb(255,119,115)      calc(var(--step) * 7)
    ) 0% var(--bg-y)/200% 700% no-repeat`,
    "--diagonal": `repeating-linear-gradient(
      128deg,
      #0e152e 0%,
      hsl(180,10%,60%) 3.8%,
      hsl(180,10%,60%) 4.5%,
      hsl(180,10%,60%) 5.2%,
      #0e152e 10%,
      #0e152e 12%
    ) var(--bg-x) var(--bg-y)/300% no-repeat`,
    "--shade": `radial-gradient(
      farthest-corner circle at var(--m-x) var(--m-y),
      rgba(255,255,255,0.1)  12%,
      rgba(255,255,255,0.15) 20%,
      rgba(255,255,255,0.25) 120%
    ) var(--bg-x) var(--bg-y)/300% no-repeat`,
    backgroundBlendMode: "hue, hue, hue, overlay",
  };

  // ── Container CSS vars (initial state) ────────────────────────────────────
  const containerVars = {
    "--m-x":        "50%",
    "--m-y":        "50%",
    "--r-x":        "0deg",
    "--r-y":        "0deg",
    "--bg-x":       "50%",
    "--bg-y":       "50%",
    "--duration":   "300ms",
    "--easing":     "ease",
    "--opacity":    "0",
    "--radius":     borderRadius,
  };

  return (
    <div
      ref={refElement}
      style={{
        ...containerVars,
        position:   "relative",
        isolation:  "isolate",
        perspective: "600px",
        transition: "transform var(--duration) var(--easing)",
        willChange: "transform",
        width,
        ...(aspectRatio ? { aspectRatio } : {}),
        ...style,
      }}
      onPointerMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const pos  = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        const pct  = { x: (100 / rect.width) * pos.x, y: (100 / rect.height) * pos.y };
        const delta = { x: pct.x - 50, y: pct.y - 50 };
        const FACTOR = 0.4;

        const s = state.current;
        s.background.x = 50 + pct.x / 4 - 12.5;
        s.background.y = 50 + pct.y / 3 - 16.67;
        s.rotate.x     = -(delta.x / 3.5) * FACTOR;
        s.rotate.y     =  (delta.y / 2)   * FACTOR;
        s.glare.x      = pct.x;
        s.glare.y      = pct.y;
        applyState();
      }}
      onPointerEnter={() => {
        isPointerInside.current = true;
        setTimeout(() => {
          if (isPointerInside.current) set("--duration", "0s");
        }, 300);
        set("--opacity", "0.6");
      }}
      onPointerLeave={() => {
        isPointerInside.current = false;
        del("--duration");
        set("--r-x",   "0deg");
        set("--r-y",   "0deg");
        set("--opacity", "0");
      }}
    >
      {/* ── 3-D tilt shell ─────────────────────────────────────────────── */}
      <div
        style={{
          height:          "100%",
          display:         "grid",
          willChange:      "transform",
          transformOrigin: "center",
          transition:      "transform var(--duration) var(--easing)",
          transform:       "rotateY(var(--r-x)) rotateX(var(--r-y))",
          borderRadius:    "var(--radius)",
          border:          "1px solid rgba(255,255,255,0.12)",
          overflow:        "hidden",
        }}
      >
        {/* Layer 1 — content */}
        <div
          style={{
            gridArea:   "1/1",
            mixBlendMode: "soft-light",
            clipPath:   "inset(0 0 0 0 round var(--radius))",
          }}
        >
          <div
            style={{
              height:     "100%",
              width:      "100%",
              background,
              ...innerStyle,
            }}
          >
            {children}
          </div>
        </div>

        {/* Layer 2 — white radial glare */}
        <div
          style={{
            gridArea:   "1/1",
            mixBlendMode: "soft-light",
            clipPath:   "inset(0 0 1px 0 round var(--radius))",
            opacity:    "var(--opacity)",
            transition: "opacity var(--duration) var(--easing), background var(--duration) var(--easing)",
            willChange: "background",
            background: `radial-gradient(
              farthest-corner circle at var(--m-x) var(--m-y),
              rgba(255,255,255,0.8)  10%,
              rgba(255,255,255,0.65) 20%,
              rgba(255,255,255,0)    90%
            )`,
          }}
        />

        {/* Layer 3 — rainbow foil shimmer */}
        <div
          style={{
            ...shimmerStyle,
            gridArea:    "1/1",
            mixBlendMode: "color-dodge",
            opacity:     "var(--opacity)",
            willChange:  "background",
            transition:  "opacity var(--duration) var(--easing)",
            clipPath:    "inset(0 0 1px 0 round var(--radius))",
            background:  "var(--pattern), var(--rainbow), var(--diagonal), var(--shade)",
          }}
        />
      </div>
    </div>
  );
};

export default GlareCard;