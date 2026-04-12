import { useRef, useEffect } from "react";
import { ChevronLeft, Sparkles, Send } from "lucide-react";

/**
 * ChatPanel (left column in Chat mode)
 *
 * Props:
 *   messages        – array of { id, role, content, timestamp }
 *   isTyping        – boolean
 *   inputMessage    – string (controlled)
 *   setInputMessage – (value) => void
 *   retrievedCount  – number of photos currently shown on canvas
 *   onSend          – form submit handler (e) => void
 *   onBack          – () => void  (back to home)
 */
const ChatPanel = ({
  messages,
  isTyping,
  inputMessage,
  setInputMessage,
  retrievedCount,
  onSend,
  onBack,
}) => {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  return (
    <div
      className="glass-card"
      style={{
        borderRadius: "24px",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "16px 20px",
          borderBottom: "1px solid rgba(255,255,255,0.1)",
          display: "flex",
          alignItems: "center",
          gap: "12px",
        }}
      >
        <button
          onClick={onBack}
          className="glass-button"
          style={{
            width: 32,
            height: 32,
            borderRadius: "8px",
            padding: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <ChevronLeft size={16} />
        </button>

        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: "12px",
            background: "linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(37, 99, 235, 0.2))",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: "1px solid rgba(59, 130, 246, 0.3)",
          }}
        >
          <Sparkles size={20} color="#60a5fa" />
        </div>

        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600, fontSize: "16px" }}>Lumeo Assistant</div>
          <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.5)" }}>
            {retrievedCount > 0 ? `${retrievedCount} photos found` : "Online"}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div
        className="chat-scroll"
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "20px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`message-bubble ${msg.role === "user" ? "user-msg" : "ai-msg"}`}
          >
            <div>{msg.content}</div>
            <div
              style={{
                fontSize: "10px",
                opacity: 0.6,
                marginTop: "4px",
                textAlign: msg.role === "user" ? "right" : "left",
              }}
            >
              {msg.timestamp
                ? new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : ""}
            </div>
          </div>
        ))}

        {isTyping && (
          <div
            style={{
              display: "flex",
              gap: "4px",
              padding: "12px",
              background: "rgba(255,255,255,0.1)",
              borderRadius: "18px",
              width: "fit-content",
            }}
          >
            <div className="typing-dot" />
            <div className="typing-dot" />
            <div className="typing-dot" />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={onSend}
        style={{
          padding: "16px",
          borderTop: "1px solid rgba(255,255,255,0.1)",
          display: "flex",
          gap: "10px",
        }}
      >
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Search memories..."
          disabled={isTyping}
          className="glass-input"
          style={{ flex: 1, padding: "12px 16px", borderRadius: "12px" }}
        />
        <button
          type="submit"
          disabled={!inputMessage.trim() || isTyping}
          className="glass-button"
          style={{
            borderRadius: "12px",
            width: "48px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};

export default ChatPanel;