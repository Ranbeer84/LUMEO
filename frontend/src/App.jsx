import { useState, useEffect } from "react";

import { API_URL } from "./constants/config";
import "./styles/globals.css";

import HomeMode from "./components/HomeMode";
import OrganizeMode from "./components/OrganizeMode";
import ChatPanel from "./components/ChatPanel";
import PhotoCanvas from "./components/PhotoCanvas";
import ClusterModal from "./components/modals/ClusterModal";
import PhotoModal from "./components/modals/PhotoModal";
import ClusterManager from "./components/ClusterManager";
import ObjectClusters from "./components/ObjectClusters";
import { SmokeBackground } from "./components/ui/SmokeBackground";

function App() {
  const [appMode, setAppMode] = useState("home");
  const [hasPhotos, setHasPhotos] = useState(false);
  const [photos, setPhotos] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [stats, setStats] = useState({ total_faces: 0 });
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState("upload");
  const [error, setError] = useState("");
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [viewingCluster, setViewingCluster] = useState(null);
  const [clusterPhotos, setClusterPhotos] = useState([]);
  const [selectedPhoto, setSelectedPhoto] = useState(null);
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      content:
        "Hi! Ask me anything about your photos like 'Show me beach photos' or 'Find photos from last summer'.",
      timestamp: Date.now(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [retrievedPhotos, setRetrievedPhotos] = useState([]);
  const [viewMode, setViewMode] = useState("grid");
  const [objectsOnly, setObjectsOnly] = useState(false);

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    const formData = new FormData();
    files.forEach((f) => formData.append("photos", f));
    try {
      setProcessing(true);
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setPhotos(data.photos);
      setCurrentStep("process");
      setError("");
    } catch (err) {
      setError("Failed to upload: " + err.message);
    } finally {
      setProcessing(false);
    }
  };

  const processPhotos = async () => {
    setProcessing(true);
    setCurrentStep("processing");
    try {
      const res = await fetch(`${API_URL}/process`, { method: "POST" });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setCurrentStep("process");
        return;
      }
      setClusters(data.clusters || []);
      setStats({ total_faces: data.total_faces || 0 });
      setCurrentStep("label");
    } catch (err) {
      setError("Failed to process: " + err.message);
      setCurrentStep("process");
    } finally {
      setProcessing(false);
    }
  };

  const loadClusters = async () => {
    try {
      const res = await fetch(`${API_URL}/clusters`);
      const data = await res.json();
      setClusters(data.clusters || []);
    } catch (err) {
      setError("Failed to load: " + err.message);
      setClusters([]);
    }
  };

  const viewClusterPhotos = async (cluster) => {
    try {
      const res = await fetch(
        `${API_URL}/cluster/${cluster.cluster_id}/photos`,
      );
      const data = await res.json();
      setClusterPhotos(data.photos);
      setViewingCluster(cluster);
    } catch (err) {
      setError("Failed to load photos: " + err.message);
    }
  };

  const updateClusterName = async (clusterId, newName) => {
    try {
      await fetch(`${API_URL}/cluster/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cluster_id: clusterId, name: newName }),
      });
      setClusters((prev) =>
        prev.map((c) =>
          c.cluster_id === clusterId ? { ...c, name: newName } : c,
        ),
      );
      if (viewingCluster?.cluster_id === clusterId) {
        setViewingCluster((prev) => ({ ...prev, name: newName }));
      }
    } catch (err) {
      setError("Failed to rename: " + err.message);
    }
  };

  const organizePhotos = async () => {
    setCurrentStep("organize");
    setProcessing(true);
    try {
      await fetch(`${API_URL}/organize`, { method: "POST" });
      setCurrentStep("complete");
      setHasPhotos(true);
      setTimeout(() => setAppMode("home"), 100);
    } catch (err) {
      setError("Failed to organize: " + err.message);
      setCurrentStep("label");
    } finally {
      setProcessing(false);
    }
  };

  const resetApp = async () => {
    if (!window.confirm("Reset all data?")) return;
    try {
      await fetch(`${API_URL}/reset`, { method: "POST" });
      setPhotos([]);
      setClusters([]);
      setCurrentStep("upload");
      setError("");
      setViewingCluster(null);
      setClusterPhotos([]);
      setSelectedPhoto(null);
      setMessages([
        {
          id: 1,
          role: "assistant",
          content: "Ready to start fresh!",
          timestamp: Date.now(),
        },
      ]);
      setRetrievedPhotos([]);
      setAppMode("home");
      setHasPhotos(false);
    } catch (err) {
      setError("Failed to reset: " + err.message);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isTyping) return;
    const userMsg = {
      id: Date.now(),
      role: "user",
      content: inputMessage.trim(),
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputMessage("");
    setIsTyping(true);
    setRetrievedPhotos([]);
    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg.content,
          conversation_id: "default",
          top_k: 8,
          stream: true,
        }),
      });
      const contentType = res.headers.get("content-type");
      if (contentType?.includes("application/json")) {
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (data.retrieved_photos) setRetrievedPhotos(data.retrieved_photos);
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            role: "assistant",
            content: data.response,
            timestamp: Date.now(),
          },
        ]);
      } else if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let assistantMsg = {
          id: Date.now() + 1,
          role: "assistant",
          content: "",
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const lines = decoder.decode(value, { stream: true }).split("\n");
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const json = line.slice(6);
              if (json.trim() === "[DONE]") break;
              const data = JSON.parse(json);
              if (data.type === "photos") {
                setRetrievedPhotos(data.photos || []);
              } else if (data.type === "token") {
                assistantMsg = {
                  ...assistantMsg,
                  content: assistantMsg.content + data.content,
                };
                setMessages((prev) => {
                  const next = [...prev];
                  next[next.length - 1] = assistantMsg;
                  return next;
                });
              }
            } catch {
              /* malformed chunk – skip */
            }
          }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: "Sorry, I encountered an error connecting to the server.",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  useEffect(() => {
    if (currentStep === "label") loadClusters();
  }, [currentStep]);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_URL}/clusters`);
        const data = await res.json();
        if (data.clusters?.length > 0) setHasPhotos(true);
      } catch {
        /* no existing photos */
      }
    };
    check();
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div
      style={{
        minHeight: "100vh",
        position: "relative",
        color: "#ffffff",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
      }}
    >
      {/* ── Smoke Background (fixed, behind everything) ───────────────────── */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 0,
          background: "linear-gradient(180deg, #000000 0%, #0d0d1a 100%)",
        }}
      >
        <SmokeBackground smokeColor="#3b1f6e" />
      </div>

      {/* ── All content (sits above the canvas) ──────────────────────────── */}
      <div
        style={{
          position: "relative",
          zIndex: 1,
          padding: "20px",
          boxSizing: "border-box",
        }}
      >
        {appMode === "home" && (
          <HomeMode
            hasPhotos={hasPhotos}
            setAppMode={setAppMode}
            resetApp={resetApp}
          />
        )}

        {appMode === "organize" && (
          <OrganizeMode
            currentStep={currentStep}
            photos={photos}
            clusters={clusters}
            stats={stats}
            processing={processing}
            error={error}
            hoveredCluster={hoveredCluster}
            setHoveredCluster={setHoveredCluster}
            handleFileUpload={handleFileUpload}
            processPhotos={processPhotos}
            organizePhotos={organizePhotos}
            updateClusterName={updateClusterName}
            viewClusterPhotos={viewClusterPhotos}
            setAppMode={setAppMode}
          />
        )}

        {appMode === "clusters" && <ClusterManager setAppMode={setAppMode} />}

        {/* ── Objects mode ─────────────────────────────────────────────── */}
        {appMode === "objects" && (
          <div style={{ maxWidth: "1400px", margin: "0 auto" }}>
            {/* Back button */}
            <button
              onClick={() => setAppMode("home")}
              style={{
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.15)",
                color: "#ffffff",
                padding: "8px 18px",
                borderRadius: "10px",
                cursor: "pointer",
                fontSize: "14px",
                marginBottom: "20px",
                backdropFilter: "blur(8px)",
              }}
            >
              ← Back
            </button>

            {/* The actual grid — ObjectClusters fetches its own data */}
            <ObjectClusters objectsOnly={objectsOnly} />
          </div>
        )}
        {/* ─────────────────────────────────────────────────────────────── */}

        {appMode === "chat" && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "400px 1fr",
              gap: "20px",
              height: "calc(100vh - 40px)",
              maxWidth: "1800px",
              margin: "0 auto",
            }}
          >
            <ChatPanel
              messages={messages}
              isTyping={isTyping}
              inputMessage={inputMessage}
              setInputMessage={setInputMessage}
              retrievedCount={retrievedPhotos.length}
              onSend={handleSendMessage}
              onBack={() => setAppMode("home")}
            />
            <PhotoCanvas
              retrievedPhotos={retrievedPhotos}
              viewMode={viewMode}
              setViewMode={setViewMode}
              onPhotoClick={setSelectedPhoto}
            />
          </div>
        )}

        <ClusterModal
          viewingCluster={viewingCluster}
          clusterPhotos={clusterPhotos}
          onClose={() => setViewingCluster(null)}
          onPhotoClick={setSelectedPhoto}
          updateClusterName={updateClusterName}
        />
        <PhotoModal
          selectedPhoto={selectedPhoto}
          onClose={() => setSelectedPhoto(null)}
        />
      </div>
    </div>
  );
}

export default App;
