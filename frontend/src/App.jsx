import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Users, FolderOpen, Camera, CheckCircle, AlertCircle, 
  Loader, X, ChevronLeft, ChevronRight, Send, Image as ImageIcon, 
  MessageSquare, Sparkles, Calendar, Clock, Smile, MapPin, 
  LayoutGrid, List
} from 'lucide-react';

const API_URL = 'http://localhost:5002/api';
const BASE_URL = 'http://localhost:5002';

// ============================================================================
// HELPER COMPONENTS
// ============================================================================

const PhotoCard = ({ photo, onClick, compact = false }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        background: 'rgba(255, 255, 255, 0.08)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.15)',
        borderRadius: compact ? '12px' : '16px',
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        transform: isHovered ? 'translateY(-4px) scale(1.02)' : 'none',
        boxShadow: isHovered 
          ? '0 12px 24px rgba(0, 0, 0, 0.4)' 
          : '0 4px 12px rgba(0, 0, 0, 0.2)',
      }}
    >
      <div style={{
        width: '100%',
        aspectRatio: '1',
        background: 'rgba(0, 0, 0, 0.3)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <img
          // src={`${BASE_URL}/uploads/${photo.path || photo.filename}`}
          src={photo.path ? `${BASE_URL}/${photo.path}` : `${BASE_URL}/uploads/${photo.filename}`}
          alt={photo.caption || 'Photo'}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }}
          loading="lazy"
        />
        
        {photo.similarity && (
          <div style={{
            position: 'absolute',
            top: '8px',
            right: '8px',
            background: 'rgba(147, 197, 253, 0.9)',
            backdropFilter: 'blur(10px)',
            padding: '4px 10px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: '600',
            color: '#000'
          }}>
            {Math.round(photo.similarity * 100)}%
          </div>
        )}
      </div>

      {!compact && (
        <div style={{ padding: '12px' }}>
          {photo.caption && (
            <div style={{
              fontSize: '13px',
              fontWeight: '500',
              marginBottom: '8px',
              color: 'rgba(255, 255, 255, 0.9)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical'
            }}>
              {photo.caption}
            </div>
          )}

          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '6px',
            fontSize: '11px',
            color: 'rgba(255, 255, 255, 0.6)',
            marginBottom: '8px'
          }}>
            {photo.date_taken && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Calendar size={11} />
                {new Date(photo.date_taken).toLocaleDateString()}
              </div>
            )}
            {photo.dominant_emotion && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Smile size={11} />
                {photo.dominant_emotion}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const TimelineView = ({ photos, onPhotoClick }) => {
  const [groupedPhotos, setGroupedPhotos] = useState({});

  useEffect(() => {
    const groups = {};
    photos.forEach(photo => {
      if (photo.date_taken) {
        const date = new Date(photo.date_taken);
        const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        const label = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
        
        if (!groups[key]) {
          groups[key] = { label, photos: [], emotions: {} };
        }
        groups[key].photos.push(photo);
        
        if (photo.dominant_emotion) {
          groups[key].emotions[photo.dominant_emotion] = 
            (groups[key].emotions[photo.dominant_emotion] || 0) + 1;
        }
      }
    });

    const sorted = Object.entries(groups)
      .sort(([a], [b]) => b.localeCompare(a))
      .reduce((acc, [key, value]) => { acc[key] = value; return acc; }, {});

    setGroupedPhotos(sorted);
  }, [photos]);

  return (
    <div className="chat-scroll" style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      {Object.entries(groupedPhotos).map(([key, group]) => (
        <div key={key} style={{ marginBottom: '32px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '16px',
            paddingBottom: '12px',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
          }}>
            <Calendar size={16} color="#93c5fd" />
            <span style={{ fontSize: '16px', fontWeight: '600' }}>{group.label}</span>
            <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)' }}>({group.photos.length})</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' }}>
            {group.photos.map((photo, idx) => (
              <PhotoCard key={idx} photo={photo} onClick={() => onPhotoClick(photo)} compact />
            ))}
          </div>
        </div>
      ))}
      {Object.keys(groupedPhotos).length === 0 && (
        <div style={{ textAlign: 'center', padding: '60px 20px', color: 'rgba(255, 255, 255, 0.4)' }}>
          <p>No dated photos found in this search.</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// MAIN APP COMPONENT
// ============================================================================

function App() {
  // --- State ---
  const [photos, setPhotos] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState('upload');
  const [error, setError] = useState('');
  const [stats, setStats] = useState({ total_faces: 0 });
  const [appMode, setAppMode] = useState('home'); // 'home', 'organize', 'chat'
  const [hasPhotos, setHasPhotos] = useState(false);
  
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [viewingCluster, setViewingCluster] = useState(null);
  const [clusterPhotos, setClusterPhotos] = useState([]);
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', content: "Hi! Ask me anything about your photos like 'Show me beach photos' or 'Find photos from last summer'.", timestamp: Date.now() }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [retrievedPhotos, setRetrievedPhotos] = useState([]);
  const [viewMode, setViewMode] = useState('grid');
  const messagesEndRef = useRef(null);

  const safeClusters = Array.isArray(clusters) ? clusters : [];

  // --- Upload & Organize Handlers ---
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    const formData = new FormData();
    files.forEach(file => formData.append('photos', file));
    try {
      setProcessing(true);
      const response = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
      const data = await response.json();
      setPhotos(data.photos);
      setCurrentStep('process');
      setError('');
    } catch (err) { setError('Failed to upload: ' + err.message); } finally { setProcessing(false); }
  };

  const processPhotos = async () => {
    setProcessing(true);
    setCurrentStep('processing');
    try {
      const response = await fetch(`${API_URL}/process`, { method: 'POST' });
      const data = await response.json();
      if (data.error) { setError(data.error); setCurrentStep('process'); return; }
      setClusters(data.clusters || []);
      setStats({ total_faces: data.total_faces || 0 });
      setCurrentStep('label');
    } catch (err) { setError('Failed to process: ' + err.message); setCurrentStep('process'); } finally { setProcessing(false); }
  };

  const loadClusters = async () => {
    try {
      const response = await fetch(`${API_URL}/clusters`);
      const data = await response.json();
      setClusters(data.clusters || []);
    } catch (err) { setError('Failed to load: ' + err.message); setClusters([]); }
  };

  const viewClusterPhotos = async (cluster) => {
    try {
      const response = await fetch(`${API_URL}/cluster/${cluster.cluster_id}/photos`);
      const data = await response.json();
      setClusterPhotos(data.photos);
      setViewingCluster(cluster);
    } catch (err) { setError('Failed to load photos: ' + err.message); }
  };

  const updateClusterName = async (clusterId, newName) => {
    try {
      await fetch(`${API_URL}/cluster/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: clusterId, name: newName })
      });
      setClusters(safeClusters.map(c => c.cluster_id === clusterId ? { ...c, name: newName } : c));
      if (viewingCluster && viewingCluster.cluster_id === clusterId) {
        setViewingCluster({ ...viewingCluster, name: newName });
      }
    } catch (err) { setError('Failed to rename: ' + err.message); }
  };

  const organizePhotos = async () => {
    setCurrentStep('organize');
    setProcessing(true);
    try {
      await fetch(`${API_URL}/organize`, { method: 'POST' });
      setCurrentStep('complete');
      setHasPhotos(true);
      setTimeout(() => setAppMode('home'), 100);
    } catch (err) { setError('Failed to organize: ' + err.message); setCurrentStep('label'); } finally { setProcessing(false); }
  };

  const resetApp = async () => {
    if (window.confirm('Reset all data?')) {
      try {
        await fetch(`${API_URL}/reset`, { method: 'POST' });
        setPhotos([]); setClusters([]); setCurrentStep('upload'); setError('');
        setViewingCluster(null); setClusterPhotos([]); setSelectedPhoto(null);
        setMessages([{ id: 1, role: 'assistant', content: "Ready to start fresh!", timestamp: Date.now() }]);
        setRetrievedPhotos([]);
        setAppMode('home');
        setHasPhotos(false);
      } catch (err) { setError('Failed to reset: ' + err.message); }
    }
  };

  // --- Chat Handler ---
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isTyping) return;

    const userMsg = { id: Date.now(), role: 'user', content: inputMessage.trim(), timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsTyping(true);
    setRetrievedPhotos([]); 

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMsg.content, 
          conversation_id: 'default', 
          top_k: 8,
          stream: true 
        })
      });

      const contentType = response.headers.get('content-type');

      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        
        if (data.error) throw new Error(data.error);

        if (data.retrieved_photos) {
          setRetrievedPhotos(data.retrieved_photos);
        }

        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          role: 'assistant',
          content: data.response,
          timestamp: Date.now()
        }]);
      } 
      else if (response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMsg = { id: Date.now() + 1, role: 'assistant', content: '', timestamp: Date.now() };
        
        setMessages(prev => [...prev, assistantMsg]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.trim() === '') continue;

            if (line.startsWith('data: ')) {
              try {
                const jsonStr = line.slice(6);
                if (jsonStr.trim() === '[DONE]') break; 

                const data = JSON.parse(jsonStr);

                if (data.type === 'photos') {
                  setRetrievedPhotos(data.photos || []);
                } 
                else if (data.type === 'token') {
                  assistantMsg.content += data.content;
                  
                  setMessages(prev => {
                    const newMsgs = [...prev];
                    newMsgs[newMsgs.length - 1] = { ...assistantMsg };
                    return newMsgs;
                  });
                } 
                else if (data.type === 'error') {
                  console.error('Stream error:', data.error);
                }
              } catch (e) {
                console.warn('Error parsing stream chunk:', e);
              }
            }
          }
        }
      }

    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        role: 'assistant', 
        content: "Sorry, I encountered an error connecting to the server." 
      }]);
    } finally { 
      setIsTyping(false); 
    }
  };

  // --- Effects ---
  useEffect(() => {
    if (currentStep === 'label') loadClusters();
  }, [currentStep]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  useEffect(() => {
    const checkExistingPhotos = async () => {
      try {
        const response = await fetch(`${API_URL}/clusters`);
        const data = await response.json();
        if (data.clusters && data.clusters.length > 0) {
          setHasPhotos(true);
        }
      } catch (err) {
        console.log('No existing photos found');
      }
    };
    checkExistingPhotos();
  }, []);

  // --- Render ---
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #000000 0%, #1a1a2e 50%, #16213e 100%)',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
      color: '#ffffff',
      boxSizing: 'border-box'
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        @keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        
        .glass-card {
          background: rgba(255, 255, 255, 0.08);
          backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.18);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        .glass-button {
          background: rgba(255, 255, 255, 0.12);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-button:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.18);
          transform: translateY(-2px);
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }
        .glass-input {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: #ffffff;
        }
        .glass-input:focus { outline: none; border-color: rgba(147, 197, 253, 0.5); box-shadow: 0 0 0 3px rgba(147, 197, 253, 0.1); }
        
        .message-bubble { max-width: 85%; padding: 12px 16px; border-radius: 18px; margin-bottom: 12px; font-size: 14px; line-height: 1.5; animation: fadeIn 0.3s ease; }
        .user-msg { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
        .ai-msg { background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.1); align-self: flex-start; border-bottom-left-radius: 4px; }
        .typing-dot { width: 8px; height: 8px; border-radius: 50%; background: rgba(147, 197, 253, 0.6); animation: bounce 1.4s infinite ease-in-out; }
        .typing-dot:nth-child(2) { animation-delay: 0.16s; }
        .typing-dot:nth-child(3) { animation-delay: 0.32s; }
        .chat-scroll::-webkit-scrollbar { width: 6px; }
        .chat-scroll::-webkit-scrollbar-track { background: transparent; }
        .chat-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
      `}</style>

      {/* ========== HOME PAGE ========== */}
      {appMode === 'home' && (
        <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '48px', marginTop: '40px' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', marginBottom: '12px', animation: 'float 3s ease-in-out infinite' }}>
              <Camera size={48} strokeWidth={1.5} style={{ color: '#93c5fd' }} />
              <h1 style={{ fontSize: 'clamp(32px, 8vw, 48px)', fontWeight: '600', margin: 0, background: 'linear-gradient(135deg, #ffffff 0%, #93c5fd 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Lumeo
              </h1>
            </div>
            <p style={{ fontSize: '16px', color: 'rgba(255,255,255,0.6)' }}>AI-Powered Photo Memory Assistant</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '32px' }}>
            {/* Chat Mode Card */}
            <div 
              onClick={() => hasPhotos ? setAppMode('chat') : alert('Please organize photos first!')}
              className="glass-card"
              style={{
                borderRadius: '24px',
                padding: '40px',
                cursor: hasPhotos ? 'pointer' : 'not-allowed',
                opacity: hasPhotos ? 1 : 0.5,
                transition: 'all 0.3s'
              }}
              onMouseEnter={(e) => hasPhotos && (e.currentTarget.style.transform = 'translateY(-4px)')}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
            >
              <div style={{ 
                width: '72px', 
                height: '72px', 
                borderRadius: '20px', 
                background: 'rgba(147, 197, 253, 0.15)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                marginBottom: '24px',
                border: '1px solid rgba(147, 197, 253, 0.3)'
              }}>
                <MessageSquare size={36} color="#93c5fd" strokeWidth={1.5} />
              </div>
              <h2 style={{ fontSize: '24px', marginBottom: '12px', fontWeight: '600' }}>Chat with Photos</h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px', lineHeight: '1.6', marginBottom: '20px' }}>
                Search your photo library with natural language. Ask questions like "Show me beach photos" or "When was I last with Mom?"
              </p>
              {hasPhotos && (
                <div style={{ 
                  display: 'inline-flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  color: '#93c5fd',
                  fontSize: '13px',
                  fontWeight: '500'
                }}>
                  Start Chatting <ChevronRight size={16} />
                </div>
              )}
              {!hasPhotos && (
                <div style={{ 
                  fontSize: '12px',
                  color: 'rgba(255,255,255,0.4)',
                  fontStyle: 'italic'
                }}>
                  No photos organized yet
                </div>
              )}
            </div>

            {/* Organize Mode Card */}
            <div 
              onClick={() => setAppMode('organize')}
              className="glass-card"
              style={{
                borderRadius: '24px',
                padding: '40px',
                cursor: 'pointer',
                transition: 'all 0.3s'
              }}
              onMouseEnter={(e) => (e.currentTarget.style.transform = 'translateY(-4px)')}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
            >
              <div style={{ 
                width: '72px', 
                height: '72px', 
                borderRadius: '20px', 
                background: 'rgba(74, 222, 128, 0.15)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                marginBottom: '24px',
                border: '1px solid rgba(74, 222, 128, 0.3)'
              }}>
                <Upload size={36} color="#4ade80" strokeWidth={1.5} />
              </div>
              <h2 style={{ fontSize: '24px', marginBottom: '12px', fontWeight: '600' }}>Organize Photos</h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px', lineHeight: '1.6', marginBottom: '20px' }}>
                Upload new photos and let AI detect faces, group people, and organize your memories automatically.
              </p>
              <div style={{ 
                display: 'inline-flex', 
                alignItems: 'center', 
                gap: '8px',
                color: '#4ade80',
                fontSize: '13px',
                fontWeight: '500'
              }}>
                Get Started <ChevronRight size={16} />
              </div>
            </div>
          </div>

          {hasPhotos && (
            <div className="glass-card" style={{ borderRadius: '20px', padding: '24px', display: 'flex', alignItems: 'center', gap: '16px' }}>
              <Sparkles size={20} color="#93c5fd" />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: '500', marginBottom: '4px' }}>You have organized photos!</div>
                <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>
                  Ready to chat and explore your memories
                </div>
              </div>
              <button 
                onClick={resetApp}
                className="glass-button"
                style={{
                  padding: '8px 16px',
                  borderRadius: '10px',
                  fontSize: '13px',
                  color: 'rgba(255,255,255,0.6)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  background: 'rgba(239, 68, 68, 0.1)'
                }}
              >
                Reset All Data
              </button>
            </div>
          )}
        </div>
      )}

      {/* ========== ORGANIZE MODE ========== */}
      {appMode === 'organize' && (
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <button 
            onClick={() => setAppMode('home')}
            className="glass-button"
            style={{
              marginBottom: '20px',
              padding: '10px 16px',
              borderRadius: '12px',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '14px'
            }}
          >
            <ChevronLeft size={16} /> Back to Home
          </button>

          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', marginBottom: '12px', animation: 'float 3s ease-in-out infinite' }}>
              <Camera size={36} strokeWidth={1.5} style={{ color: '#93c5fd' }} />
              <h1 style={{ fontSize: 'clamp(24px, 6vw, 36px)', fontWeight: '600', margin: 0, background: 'linear-gradient(135deg, #ffffff 0%, #93c5fd 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Photo Organizer
              </h1>
            </div>
            <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)' }}>AI-powered face recognition & organization</p>
          </div>

          {error && (
            <div className="glass-card" style={{ padding: '16px', borderRadius: '16px', marginBottom: '24px', borderColor: 'rgba(239, 68, 68, 0.3)', background: 'rgba(239, 68, 68, 0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <AlertCircle size={20} color="#ef4444" />
                <span style={{ fontSize: '14px', color: '#fca5a5' }}>{error}</span>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '32px', gap: '12px', overflowX: 'auto' }}>
            {['upload', 'process', 'label', 'complete'].map((step, idx) => {
              const isActive = currentStep === step || (step === 'process' && currentStep === 'processing') || (step === 'complete' && currentStep === 'organize');
              const stepIndex = ['upload', 'process', 'label', 'complete'].indexOf(currentStep === 'processing' ? 'process' : currentStep === 'organize' ? 'complete' :
              
              currentStep);
              const isCompleted = stepIndex > idx;
              
              return (
                <React.Fragment key={step}>
                  <div className="glass-button" style={{ 
                    padding: '10px 20px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: '500',
                    background: isActive ? 'rgba(147, 197, 253, 0.2)' : isCompleted ? 'rgba(74, 222, 128, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                    borderColor: isActive ? 'rgba(147, 197, 253, 0.4)' : isCompleted ? 'rgba(74, 222, 128, 0.3)' : 'rgba(255, 255, 255, 0.15)',
                    color: isActive ? '#93c5fd' : isCompleted ? '#4ade80' : 'rgba(255,255,255,0.5)'
                  }}>
                    <div style={{ width: '20px', height: '20px', borderRadius: '50%', border: '2px solid', borderColor: 'currentColor', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px' }}>{idx + 1}</div>
                    <span style={{ textTransform: 'capitalize' }}>{step}</span>
                  </div>
                  {idx < 3 && <div style={{ width: '24px', height: '2px', background: isCompleted ? 'rgba(74, 222, 128, 0.3)' : 'rgba(255, 255, 255, 0.1)', marginTop: '18px' }} />}
                </React.Fragment>
              );
            })}
          </div>

          {/* --- Organize Mode Steps Content --- */}
          
          {currentStep === 'upload' && (
            <div className="glass-card" style={{ borderRadius: '24px', padding: '60px', textAlign: 'center' }}>
              <div style={{ width: '80px', height: '80px', margin: '0 auto 24px', borderRadius: '20px', background: 'rgba(147, 197, 253, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(147, 197, 253, 0.2)' }}>
                <Upload size={40} color="#93c5fd" strokeWidth={1.5} />
              </div>
              <h2 style={{ fontSize: '28px', marginBottom: '12px' }}>Upload Photos</h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '32px' }}>Select multiple photos to organize</p>
              <label>
                <input type="file" multiple accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} disabled={processing} />
                <div className="glass-button" style={{ padding: '16px 32px', borderRadius: '16px', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '12px', fontSize: '16px', fontWeight: '500' }}>
                  <Upload size={20} />
                  {processing ? 'Uploading...' : 'Choose Photos'}
                </div>
              </label>
            </div>
          )}

          {currentStep === 'process' && (
            <div className="glass-card" style={{ borderRadius: '24px', padding: '60px', textAlign: 'center' }}>
              <h2 style={{ fontSize: '28px', marginBottom: '12px' }}>Ready to Process</h2>
              <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '32px' }}>{photos.length} photos uploaded</p>
              <button onClick={processPhotos} disabled={processing} className="glass-button" style={{ padding: '16px 32px', borderRadius: '16px', border: 'none', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '12px', fontSize: '16px', fontWeight: '500', color: '#ffffff' }}>
                <Users size={20} /> Start Face Detection
              </button>
            </div>
          )}

          {currentStep === 'processing' && (
            <div className="glass-card" style={{ borderRadius: '24px', padding: '60px', textAlign: 'center' }}>
              <Loader size={60} color="#93c5fd" strokeWidth={1.5} style={{ margin: '0 auto 24px', animation: 'spin 1s linear infinite' }} />
              <h2 style={{ fontSize: '28px', marginBottom: '12px' }}>Processing...</h2>
              <p style={{ color: 'rgba(255,255,255,0.6)' }}>Detecting and grouping faces</p>
            </div>
          )}

          {currentStep === 'label' && (
            <div className="glass-card" style={{ borderRadius: '24px', padding: '40px' }}>
              <div style={{ textAlign: 'center', marginBottom: '32px' }}>
                <h2 style={{ fontSize: '28px', marginBottom: '12px' }}>Label People</h2>
                <p style={{ color: 'rgba(255,255,255,0.6)' }}>Found {safeClusters.length} people ({stats.total_faces} faces)</p>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px', marginBottom: '32px' }}>
                {safeClusters.filter(c => c.photos && c.photos.length > 0).map(cluster => (
                  <div key={cluster.cluster_id} className="glass-card"
                    style={{ borderRadius: '20px', padding: '16px', cursor: 'pointer', transition: 'all 0.3s', transform: hoveredCluster === cluster.cluster_id ? 'translateY(-4px)' : 'none', borderColor: hoveredCluster === cluster.cluster_id ? 'rgba(147, 197, 253, 0.4)' : 'rgba(255, 255, 255, 0.18)' }}
                    onMouseEnter={() => setHoveredCluster(cluster.cluster_id)} onMouseLeave={() => setHoveredCluster(null)}
                    onClick={() => viewClusterPhotos(cluster)}
                  >
                    <div style={{ width: '100%', aspectRatio: '1', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '16px', marginBottom: '16px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {cluster.thumbnail ? <img src={`${BASE_URL}/thumbnails/${cluster.thumbnail}`} alt="Face" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <Users size={48} color="rgba(255,255,255,0.3)" />}
                    </div>
                    <input type="text" value={cluster.name} onChange={(e) => { e.stopPropagation(); updateClusterName(cluster.cluster_id, e.target.value); }} onClick={(e) => e.stopPropagation()} className="glass-input" style={{ width: '100%', padding: '12px', borderRadius: '12px', textAlign: 'center', boxSizing: 'border-box' }} placeholder="Enter name..." />
                    <p style={{ textAlign: 'center', fontSize: '13px', color: 'rgba(255,255,255,0.5)', margin: '12px 0 0' }}>{cluster.photos.length} photos</p>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
                <button onClick={organizePhotos} disabled={processing} className="glass-button" style={{ padding: '16px 32px', borderRadius: '16px', border: 'none', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '12px', fontSize: '16px', fontWeight: '500', color: '#ffffff', background: 'rgba(74, 222, 128, 0.15)', borderColor: 'rgba(74, 222, 128, 0.3)' }}>
                  <Sparkles size={20} /> Finish Organization
                </button>
              </div>
            </div>
          )}

          {currentStep === 'organize' && (
            <div className="glass-card" style={{ borderRadius: '24px', padding: '60px', textAlign: 'center' }}>
              <FolderOpen size={60} color="#93c5fd" strokeWidth={1.5} style={{ margin: '0 auto 24px', animation: 'pulse 2s infinite' }} />
              <h2 style={{ fontSize: '28px', marginBottom: '12px' }}>Preparing AI...</h2>
              <p style={{ color: 'rgba(255,255,255,0.6)' }}>Indexing your memories for chat</p>
            </div>
          )}
        </div>
      )}

      {/* ========== CHAT MODE ========== */}
      {appMode === 'chat' && (
        <div style={{ display: 'grid', gridTemplateColumns: '400px 1fr', gap: '20px', height: 'calc(100vh - 40px)', maxWidth: '1800px', margin: '0 auto' }}>
          
          {/* LEFT PANEL: Chat Interface */}
          <div className="glass-card" style={{ borderRadius: '24px', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {/* Chat Header */}
            <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '12px' }}>
               <button onClick={() => setAppMode('home')} className="glass-button" style={{ width: 32, height: 32, borderRadius: '8px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                 <ChevronLeft size={16} />
               </button>
              <div style={{ width: 40, height: 40, borderRadius: '12px', background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(37, 99, 235, 0.2))', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                <Sparkles size={20} color="#60a5fa" />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: '16px' }}>Lumeo Assistant</div>
                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.5)' }}>{retrievedPhotos.length > 0 ? `${retrievedPhotos.length} photos found` : 'Online'}</div>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="chat-scroll" style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column' }}>
              {messages.map((msg) => (
                <div key={msg.id} className={`message-bubble ${msg.role === 'user' ? 'user-msg' : 'ai-msg'}`}>
                  <div>{msg.content}</div>
                  <div style={{ fontSize: '10px', opacity: 0.6, marginTop: '4px', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))}
              {isTyping && <div style={{ display: 'flex', gap: '4px', padding: '12px', background: 'rgba(255,255,255,0.1)', borderRadius: '18px', width: 'fit-content' }}><div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" /></div>}
              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input */}
            <form onSubmit={handleSendMessage} style={{ padding: '16px', borderTop: '1px solid rgba(255,255,255,0.1)', display: 'flex', gap: '10px' }}>
              <input type="text" value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} placeholder="Search memories..." disabled={isTyping} className="glass-input" style={{ flex: 1, padding: '12px 16px', borderRadius: '12px' }} />
              <button type="submit" disabled={!inputMessage.trim() || isTyping} className="glass-button" style={{ borderRadius: '12px', width: '48px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Send size={18} />
              </button>
            </form>
          </div>

          {/* RIGHT PANEL: Visual Canvas */}
          <div className="glass-card" style={{ borderRadius: '24px', overflow: 'hidden', position: 'relative' }}>
            {retrievedPhotos.length > 0 && (
              <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ flex: 1, fontSize: '14px', fontWeight: '500' }}>Search Results</div>
                <div style={{ display: 'flex', background: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '4px', gap: '4px' }}>
                  <button onClick={() => setViewMode('grid')} style={{ background: viewMode === 'grid' ? 'rgba(59, 130, 246, 0.3)' : 'transparent', border: 'none', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'white' }}><LayoutGrid size={14} /> Grid</button>
                  <button onClick={() => setViewMode('timeline')} style={{ background: viewMode === 'timeline' ? 'rgba(59, 130, 246, 0.3)' : 'transparent', border: 'none', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'white' }}><List size={14} /> Timeline</button>
                </div>
              </div>
            )}

            {retrievedPhotos.length === 0 ? (
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center', opacity: 0.4 }}>
                <div style={{ width: 80, height: 80, borderRadius: '24px', background: 'rgba(255,255,255,0.05)', margin: '0 auto 24px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><ImageIcon size={40} /></div>
                <h3 style={{ margin: '0 0 8px 0', fontWeight: 500 }}>Visual Context</h3>
                <p style={{ margin: 0, fontSize: '14px' }}>Photos will appear here based on your chat</p>
              </div>
            ) : viewMode === 'grid' ? (
              <div className="chat-scroll" style={{ padding: '20px', overflowY: 'auto', height: 'calc(100% - 65px)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
                  {retrievedPhotos.map((photo, idx) => (
                    <PhotoCard key={idx} photo={photo} onClick={() => setSelectedPhoto(photo)} />
                  ))}
                </div>
              </div>
            ) : (
              <TimelineView photos={retrievedPhotos} onPhotoClick={setSelectedPhoto} />
            )}
          </div>
        </div>
      )}

      {/* ========== MODALS ========== */}
      
      {/* Cluster/Person Detail Modal */}
      {viewingCluster && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(20px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '20px' }} onClick={() => setViewingCluster(null)}>
          <div className="glass-card" style={{ borderRadius: '24px', padding: '40px', maxWidth: '1000px', width: '100%', maxHeight: '90vh', overflowY: 'auto', position: 'relative' }} onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setViewingCluster(null)} className="glass-button" style={{ position: 'absolute', top: '16px', right: '16px', width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0 }}><X size={20} /></button>
            <h2 style={{ fontSize: '28px', marginBottom: '8px', fontWeight: '600' }}>{viewingCluster.name}</h2>
            <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '24px' }}>{clusterPhotos.length} photos</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '12px' }}>
              {clusterPhotos.map((photo, idx) => (
                <div key={photo.photo_id} style={{ aspectRatio: '1', borderRadius: '12px', overflow: 'hidden', cursor: 'pointer', border: '1px solid rgba(255, 255, 255, 0.1)' }} onClick={() => setSelectedPhoto({ ...photo, index: idx })}>
                  {/* <img src={`${BASE_URL}/uploads/${photo.path}`} alt={photo.filename} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> */}
                  <img src={photo.path ? `${BASE_URL}/${photo.path}` : `${BASE_URL}/uploads/${photo.filename}`} alt={photo.filename} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Full Screen Photo Viewer */}
      {selectedPhoto && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.95)', backdropFilter: 'blur(40px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2000, padding: '20px' }} onClick={() => setSelectedPhoto(null)}>
          <button onClick={() => setSelectedPhoto(null)} className="glass-button" style={{ position: 'absolute', top: '20px', right: '20px', width: '50px', height: '50px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0 }}><X size={24} /></button>
          <div onClick={(e) => e.stopPropagation()} style={{ textAlign: 'center', maxWidth: '90vw' }}>
            <img src={`${BASE_URL}/uploads/${selectedPhoto.path || selectedPhoto.filename}`} alt="Full view" style={{ maxWidth: '100%', maxHeight: '80vh', borderRadius: '16px', boxShadow: '0 20px 60px rgba(0,0,0,0.5)', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
            {selectedPhoto.caption && (
              <div className="glass-card" style={{ marginTop: '20px', padding: '16px 24px', borderRadius: '16px', display: 'inline-block' }}>
                <div style={{ fontSize: '14px', marginBottom: '8px' }}>{selectedPhoto.caption}</div>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', display: 'flex', gap: '12px' }}>
                  {selectedPhoto.date_taken && <span>📅 {new Date(selectedPhoto.date_taken).toLocaleDateString()}</span>}
                  {selectedPhoto.dominant_emotion && <span>😊 {selectedPhoto.dominant_emotion}</span>}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;