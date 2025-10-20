import React, { useState, useEffect } from 'react';
import { Upload, Users, FolderOpen, Camera, CheckCircle, AlertCircle, Loader, X, ChevronLeft, ChevronRight } from 'lucide-react';

const API_URL = 'http://localhost:5002/api';
const BASE_URL = 'http://localhost:5002';

function App() {
  const [photos, setPhotos] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState('upload');
  const [error, setError] = useState('');
  const [stats, setStats] = useState({ total_faces: 0 });
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [viewingCluster, setViewingCluster] = useState(null);
  const [clusterPhotos, setClusterPhotos] = useState([]);
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  const safeClusters = Array.isArray(clusters) ? clusters : [];

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach(file => formData.append('photos', file));

    try {
      setProcessing(true);
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setPhotos(data.photos);
      setCurrentStep('process');
      setError('');
    } catch (err) {
      setError('Failed to upload: ' + err.message);
    } finally {
      setProcessing(false);
    }
  };

  const processPhotos = async () => {
    setProcessing(true);
    setCurrentStep('processing');
    try {
      const response = await fetch(`${API_URL}/process`, { method: 'POST' });
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setCurrentStep('process');
        return;
      }
      
      setClusters(data.clusters || []);
      setStats({ total_faces: data.total_faces || 0 });
      setCurrentStep('label');
    } catch (err) {
      setError('Failed to process: ' + err.message);
      setCurrentStep('process');
    } finally {
      setProcessing(false);
    }
  };

  const loadClusters = async () => {
    try {
      const response = await fetch(`${API_URL}/clusters`);
      const data = await response.json();
      setClusters(data.clusters || []);
    } catch (err) {
      setError('Failed to load: ' + err.message);
      setClusters([]);
    }
  };

  const viewClusterPhotos = async (cluster) => {
    try {
      const response = await fetch(`${API_URL}/cluster/${cluster.cluster_id}/photos`);
      const data = await response.json();
      setClusterPhotos(data.photos);
      setViewingCluster(cluster);
    } catch (err) {
      setError('Failed to load photos: ' + err.message);
    }
  };

  const updateClusterName = async (clusterId, newName) => {
    try {
      await fetch(`${API_URL}/cluster/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: clusterId, name: newName })
      });
      setClusters(safeClusters.map(c => 
        c.cluster_id === clusterId ? { ...c, name: newName } : c
      ));
      if (viewingCluster && viewingCluster.cluster_id === clusterId) {
        setViewingCluster({ ...viewingCluster, name: newName });
      }
    } catch (err) {
      setError('Failed to rename: ' + err.message);
    }
  };

  const organizePhotos = async () => {
    setCurrentStep('organize');
    setProcessing(true);
    try {
      const response = await fetch(`${API_URL}/organize`, { method: 'POST' });
      const data = await response.json();
      setCurrentStep('complete');
    } catch (err) {
      setError('Failed to organize: ' + err.message);
      setCurrentStep('label');
    } finally {
      setProcessing(false);
    }
  };

  const resetApp = async () => {
    if (window.confirm('Reset all data?')) {
      try {
        await fetch(`${API_URL}/reset`, { method: 'POST' });
        setPhotos([]);
        setClusters([]);
        setCurrentStep('upload');
        setError('');
        setViewingCluster(null);
        setClusterPhotos([]);
      } catch (err) {
        setError('Failed to reset: ' + err.message);
      }
    }
  };

  useEffect(() => {
    if (currentStep === 'label') loadClusters();
  }, [currentStep]);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #000000 0%, #1a1a2e 50%, #16213e 100%)',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
      color: '#ffffff'
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        @keyframes shimmer {
          0% { background-position: -200% center; }
          100% { background-position: 200% center; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        
        .glass-card {
          background: rgba(255, 255, 255, 0.08);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.18);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .glass-button {
          background: rgba(255, 255, 255, 0.12);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-button:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.18);
          transform: translateY(-2px);
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }
        
        .glass-button:active:not(:disabled) {
          transform: translateY(0px);
        }
        
        .glass-input {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: #ffffff;
        }
        
        .glass-input:focus {
          outline: none;
          border-color: rgba(147, 197, 253, 0.5);
          box-shadow: 0 0 0 3px rgba(147, 197, 253, 0.1);
        }
        
        * {
          -webkit-tap-highlight-color: transparent;
        }
      `}</style>

      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '12px',
            animation: 'float 3s ease-in-out infinite'
          }}>
            <Camera size={36} strokeWidth={1.5} style={{ color: '#93c5fd' }} />
            <h1 style={{
              fontSize: 'clamp(24px, 6vw, 36px)',
              fontWeight: '600',
              margin: 0,
              background: 'linear-gradient(135deg, #ffffff 0%, #93c5fd 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em'
            }}>
              Photo Organizer
            </h1>
          </div>
          <p style={{ 
            fontSize: '14px', 
            color: 'rgba(255,255,255,0.6)',
            fontWeight: '400',
            margin: 0
          }}>
            AI-powered face recognition
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="glass-card" style={{
            padding: '16px',
            borderRadius: '16px',
            marginBottom: '24px',
            borderColor: 'rgba(239, 68, 68, 0.3)',
            background: 'rgba(239, 68, 68, 0.1)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <AlertCircle size={20} color="#ef4444" />
              <span style={{ fontSize: '14px', color: '#fca5a5' }}>{error}</span>
            </div>
          </div>
        )}

        {/* Progress Steps */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          marginBottom: '32px',
          gap: '12px',
          overflowX: 'auto',
          padding: '8px 0'
        }}>
          {['upload', 'process', 'label', 'complete'].map((step, idx) => {
            const isActive = currentStep === step;
            const stepIndex = ['upload', 'process', 'label', 'complete'].indexOf(currentStep);
            const isCompleted = stepIndex > idx;
            return (
              <React.Fragment key={step}>
                <div className="glass-button" style={{
                  padding: '10px 20px',
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '13px',
                  fontWeight: '500',
                  whiteSpace: 'nowrap',
                  background: isActive ? 'rgba(147, 197, 253, 0.2)' : isCompleted ? 'rgba(74, 222, 128, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                  borderColor: isActive ? 'rgba(147, 197, 253, 0.4)' : isCompleted ? 'rgba(74, 222, 128, 0.3)' : 'rgba(255, 255, 255, 0.15)',
                  transform: isActive ? 'scale(1.05)' : 'scale(1)',
                  transition: 'all 0.3s'
                }}>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '50%',
                    border: '2px solid',
                    borderColor: isActive || isCompleted ? 'currentColor' : 'rgba(255,255,255,0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '11px',
                    color: isActive ? '#93c5fd' : isCompleted ? '#4ade80' : 'rgba(255,255,255,0.5)'
                  }}>{idx + 1}</div>
                  <span style={{ 
                    textTransform: 'capitalize',
                    color: isActive ? '#93c5fd' : isCompleted ? '#4ade80' : 'rgba(255,255,255,0.7)'
                  }}>{step}</span>
                </div>
                {idx < 3 && (
                  <div style={{
                    width: '24px',
                    height: '2px',
                    background: isCompleted ? 'rgba(74, 222, 128, 0.3)' : 'rgba(255, 255, 255, 0.1)',
                    borderRadius: '2px'
                  }} />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Upload Section */}
        {currentStep === 'upload' && (
          <div className="glass-card" style={{
            borderRadius: '24px',
            padding: 'clamp(32px, 8vw, 60px)',
            textAlign: 'center'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              margin: '0 auto 24px',
              borderRadius: '20px',
              background: 'rgba(147, 197, 253, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(147, 197, 253, 0.2)'
            }}>
              <Upload size={40} color="#93c5fd" strokeWidth={1.5} />
            </div>
            <h2 style={{ 
              fontSize: 'clamp(22px, 5vw, 28px)', 
              fontWeight: '600', 
              marginBottom: '12px',
              letterSpacing: '-0.01em'
            }}>
              Upload Photos
            </h2>
            <p style={{ 
              color: 'rgba(255,255,255,0.6)', 
              marginBottom: '32px',
              fontSize: '14px'
            }}>
              Select multiple photos to organize
            </p>
            <label>
              <input 
                type="file" 
                multiple 
                accept="image/*" 
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
                disabled={processing} 
              />
              <div className="glass-button" style={{
                padding: '16px 32px',
                borderRadius: '16px',
                cursor: processing ? 'not-allowed' : 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '12px',
                fontSize: '16px',
                fontWeight: '500',
                opacity: processing ? 0.5 : 1
              }}>
                <Upload size={20} />
                {processing ? 'Uploading...' : 'Choose Photos'}
              </div>
            </label>
          </div>
        )}

        {/* Process Section */}
        {currentStep === 'process' && (
          <div className="glass-card" style={{
            borderRadius: '24px',
            padding: 'clamp(32px, 8vw, 60px)',
            textAlign: 'center'
          }}>
            <h2 style={{ 
              fontSize: 'clamp(22px, 5vw, 28px)', 
              fontWeight: '600', 
              marginBottom: '12px'
            }}>
              Ready to Process
            </h2>
            <p style={{ 
              color: 'rgba(255,255,255,0.6)', 
              marginBottom: '32px',
              fontSize: '14px'
            }}>
              {photos.length} photos uploaded
            </p>
            <button 
              onClick={processPhotos} 
              disabled={processing} 
              className="glass-button"
              style={{
                padding: '16px 32px',
                borderRadius: '16px',
                border: 'none',
                cursor: processing ? 'not-allowed' : 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '12px',
                fontSize: '16px',
                fontWeight: '500',
                color: '#ffffff',
                opacity: processing ? 0.5 : 1
              }}
            >
              <Users size={20} />
              Start Face Detection
            </button>
          </div>
        )}

        {/* Processing Section */}
        {currentStep === 'processing' && (
          <div className="glass-card" style={{
            borderRadius: '24px',
            padding: 'clamp(32px, 8vw, 60px)',
            textAlign: 'center'
          }}>
            <Loader 
              size={60} 
              color="#93c5fd" 
              strokeWidth={1.5}
              style={{ 
                margin: '0 auto 24px', 
                animation: 'spin 1s linear infinite',
                display: 'block'
              }} 
            />
            <h2 style={{ 
              fontSize: 'clamp(22px, 5vw, 28px)', 
              fontWeight: '600', 
              marginBottom: '12px'
            }}>
              Processing...
            </h2>
            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px' }}>
              Detecting and grouping faces
            </p>
          </div>
        )}

        {/* Label Section */}
        {currentStep === 'label' && (
          <div className="glass-card" style={{
            borderRadius: '24px',
            padding: 'clamp(20px, 5vw, 40px)'
          }}>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <h2 style={{ 
                fontSize: 'clamp(22px, 5vw, 28px)', 
                fontWeight: '600', 
                marginBottom: '12px'
              }}>
                Label People
              </h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px' }}>
                Found {safeClusters.length} {safeClusters.length === 1 ? 'person' : 'people'} ({stats.total_faces} faces)
              </p>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(min(280px, 100%), 1fr))',
              gap: '16px',
              marginBottom: '32px'
            }}>
              {safeClusters.filter(c => c.photos && c.photos.length > 0).map(cluster => (
                <div
                  key={cluster.cluster_id}
                  className="glass-card"
                  style={{
                    borderRadius: '20px',
                    padding: '16px',
                    cursor: 'pointer',
                    transition: 'all 0.3s',
                    transform: hoveredCluster === cluster.cluster_id ? 'translateY(-4px) scale(1.02)' : 'none',
                    borderColor: hoveredCluster === cluster.cluster_id ? 'rgba(147, 197, 253, 0.4)' : 'rgba(255, 255, 255, 0.18)'
                  }}
                  onMouseEnter={() => setHoveredCluster(cluster.cluster_id)}
                  onMouseLeave={() => setHoveredCluster(null)}
                  onTouchStart={() => setHoveredCluster(cluster.cluster_id)}
                  onClick={() => viewClusterPhotos(cluster)}
                >
                  <div style={{
                    width: '100%',
                    aspectRatio: '1',
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '16px',
                    marginBottom: '16px',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    {cluster.thumbnail ? (
                      <img
                        src={`${BASE_URL}/thumbnails/${cluster.thumbnail}`}
                        alt="Face"
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    ) : (
                      <Users size={48} color="rgba(255,255,255,0.3)" strokeWidth={1.5} />
                    )}
                  </div>
                  <input
                    type="text"
                    value={cluster.name}
                    onChange={(e) => {
                      e.stopPropagation();
                      updateClusterName(cluster.cluster_id, e.target.value);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    className="glass-input"
                    style={{
                      width: '100%',
                      padding: '12px',
                      borderRadius: '12px',
                      fontSize: '15px',
                      fontWeight: '500',
                      textAlign: 'center',
                      boxSizing: 'border-box'
                    }}
                    placeholder="Enter name..."
                  />
                  <p style={{ 
                    textAlign: 'center', 
                    fontSize: '13px', 
                    color: 'rgba(255,255,255,0.5)', 
                    margin: '12px 0 0',
                    fontWeight: '400'
                  }}>
                    {cluster.photos.length} {cluster.photos.length === 1 ? 'photo' : 'photos'} • {cluster.face_count} {cluster.face_count === 1 ? 'face' : 'faces'}
                  </p>
                </div>
              ))}
            </div>

            <div style={{ 
              display: 'flex', 
              justifyContent: 'center', 
              gap: '12px', 
              flexWrap: 'wrap' 
            }}>
              <button
                onClick={organizePhotos}
                disabled={processing}
                className="glass-button"
                style={{
                  padding: '16px 32px',
                  borderRadius: '16px',
                  border: 'none',
                  cursor: processing ? 'not-allowed' : 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '12px',
                  fontSize: '16px',
                  fontWeight: '500',
                  color: '#ffffff',
                  background: 'rgba(74, 222, 128, 0.15)',
                  borderColor: 'rgba(74, 222, 128, 0.3)',
                  opacity: processing ? 0.5 : 1
                }}
              >
                <FolderOpen size={20} />
                Organize Folders
              </button>
              <button
                onClick={resetApp}
                className="glass-button"
                style={{
                  padding: '16px 32px',
                  borderRadius: '16px',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '12px',
                  fontSize: '16px',
                  fontWeight: '500',
                  color: '#ffffff'
                }}
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Organize Section */}
        {currentStep === 'organize' && (
          <div className="glass-card" style={{
            borderRadius: '24px',
            padding: 'clamp(32px, 8vw, 60px)',
            textAlign: 'center'
          }}>
            <FolderOpen 
              size={60} 
              color="#93c5fd" 
              strokeWidth={1.5}
              style={{ 
                margin: '0 auto 24px', 
                animation: 'pulse 2s infinite',
                display: 'block'
              }} 
            />
            <h2 style={{ 
              fontSize: 'clamp(22px, 5vw, 28px)', 
              fontWeight: '600', 
              marginBottom: '12px'
            }}>
              Organizing...
            </h2>
            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px' }}>
              Creating folders for each person
            </p>
          </div>
        )}

        {/* Complete Section */}
        {currentStep === 'complete' && (
          <div className="glass-card" style={{
            borderRadius: '24px',
            padding: 'clamp(32px, 8vw, 60px)',
            textAlign: 'center'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              margin: '0 auto 24px',
              borderRadius: '20px',
              background: 'rgba(74, 222, 128, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(74, 222, 128, 0.3)'
            }}>
              <CheckCircle size={40} color="#4ade80" strokeWidth={1.5} />
            </div>
            <h2 style={{ 
              fontSize: 'clamp(22px, 5vw, 28px)', 
              fontWeight: '600', 
              marginBottom: '12px'
            }}>
              Complete! 🎉
            </h2>
            <p style={{ 
              color: 'rgba(255,255,255,0.6)', 
              marginBottom: '32px',
              fontSize: '14px'
            }}>
              Photos organized successfully
            </p>

            <div className="glass-card" style={{
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '32px',
              textAlign: 'left',
              maxWidth: '500px',
              margin: '0 auto 32px'
            }}>
              <h3 style={{ 
                fontWeight: '600', 
                marginBottom: '16px',
                fontSize: '16px'
              }}>
                Folders Created:
              </h3>
              {safeClusters.filter(c => c.photos && c.photos.length > 0).map(cluster => (
                <div key={cluster.cluster_id} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '12px 0',
                  borderBottom: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <FolderOpen size={18} color="#93c5fd" strokeWidth={1.5} />
                  <span style={{ fontWeight: '500', flex: 1, fontSize: '14px' }}>{cluster.name}</span>
                  <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '13px' }}>
                    {cluster.photos.length}
                  </span>
                </div>
              ))}
            </div>

            <button 
              onClick={resetApp} 
              className="glass-button"
              style={{
                padding: '16px 32px',
                borderRadius: '16px',
                border: 'none',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '12px',
                fontSize: '16px',
                fontWeight: '500',
                color: '#ffffff'
              }}
            >
              Organize More
            </button>
          </div>
        )}
      </div>

      {/* Cluster Photos Modal */}
      {viewingCluster && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
            overflowY: 'auto'
          }} 
          onClick={() => setViewingCluster(null)}
        >
          <div 
            className="glass-card"
            style={{
              borderRadius: '24px',
              padding: 'clamp(24px, 5vw, 40px)',
              maxWidth: '1000px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              position: 'relative'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setViewingCluster(null)}
              className="glass-button"
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                border: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                padding: 0
              }}
            >
              <X size={20} />
            </button>

            <div style={{ textAlign: 'center', marginBottom: '24px', paddingRight: '40px' }}>
              <h2 style={{ 
                fontSize: 'clamp(20px, 5vw, 28px)', 
                marginBottom: '8px',
                fontWeight: '600'
              }}>
                {viewingCluster.name}
              </h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px' }}>
                {clusterPhotos.length} {clusterPhotos.length === 1 ? 'photo' : 'photos'} • {viewingCluster.face_count} {viewingCluster.face_count === 1 ? 'face' : 'faces'}
              </p>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(min(150px, 100%), 1fr))',
              gap: '12px'
            }}>
              {clusterPhotos.map((photo, idx) => (
                <div
                  key={photo.photo_id}
                  style={{
                    position: 'relative',
                    aspectRatio: '1',
                    borderRadius: '12px',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    transition: 'transform 0.2s',
                    background: 'rgba(0, 0, 0, 0.3)',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}
                  onClick={() => setSelectedPhoto({ ...photo, index: idx })}
                  onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                  onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                >
                  <img
                    src={`${BASE_URL}/uploads/${photo.path}`}
                    alt={photo.filename}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover'
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Photo Viewer Modal */}
      {selectedPhoto && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.95)',
            backdropFilter: 'blur(40px)',
            WebkitBackdropFilter: 'blur(40px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '20px'
          }} 
          onClick={() => setSelectedPhoto(null)}
        >
          <button
            onClick={() => setSelectedPhoto(null)}
            className="glass-button"
            style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              width: '50px',
              height: '50px',
              borderRadius: '50%',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              zIndex: 2001,
              padding: 0
            }}
          >
            <X size={24} />
          </button>

          {selectedPhoto.index > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelectedPhoto({ ...clusterPhotos[selectedPhoto.index - 1], index: selectedPhoto.index - 1 });
              }}
              className="glass-button"
              style={{
                position: 'absolute',
                left: '20px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '50px',
                height: '50px',
                borderRadius: '50%',
                border: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                zIndex: 2001,
                padding: 0
              }}
            >
              <ChevronLeft size={24} />
            </button>
          )}

          {selectedPhoto.index < clusterPhotos.length - 1 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelectedPhoto({ ...clusterPhotos[selectedPhoto.index + 1], index: selectedPhoto.index + 1 });
              }}
              className="glass-button"
              style={{
                position: 'absolute',
                right: '20px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '50px',
                height: '50px',
                borderRadius: '50%',
                border: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                zIndex: 2001,
                padding: 0
              }}
            >
              <ChevronRight size={24} />
            </button>
          )}

          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <img
              src={`${BASE_URL}/uploads/${selectedPhoto.path}`}
              alt={selectedPhoto.filename}
              style={{
                maxWidth: '100%',
                maxHeight: '90vh',
                borderRadius: '16px',
                boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;