import { useState, useEffect, useCallback } from "react";
import {
  ChevronLeft,
  Users,
  Merge,
  AlertTriangle,
  Search,
  Layers,
  X,
} from "lucide-react";
import { BASE_URL } from "../constants/config";
import {
  accent,
  ConfirmDialog,
  ClusterCard,
  ClusterDetail,
  StatsBar,
} from "./ClusterComponents";

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

  // ── Data fetching ───────────────────────────────────────────────────────────
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

  // ── Handlers ────────────────────────────────────────────────────────────────
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

  const cancelMerge = () => {
    setMergeMode(false);
    setMergeSource(null);
  };

  const filtered = clusters.filter((c) =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ maxWidth: "1440px", margin: "0 auto", padding: "4px 0" }}>
      {/* Top bar */}
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
              onClick={cancelMerge}
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

      {/* Error banner */}
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

      {/* Stats + search row */}
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

      {/* Main content */}
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
        <div
          style={{
            display: "grid",
            gridTemplateColumns: selectedCluster ? "1fr 360px" : "1fr",
            gap: "20px",
            alignItems: "start",
          }}
        >
          {/* Cluster cards grid */}
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

          {/* Sticky detail panel */}
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

      {/* Merge confirm dialog */}
      {confirmMerge && (
        <ConfirmDialog
          message={`Merge "${confirmMerge.source.name}" INTO "${confirmMerge.target.name}"? Source cluster will be deleted; all its photos move to the target.`}
          onConfirm={executeMerge}
          onCancel={() => {
            setConfirmMerge(null);
            cancelMerge();
          }}
        />
      )}
    </div>
  );
};

export default ClusterManager;
