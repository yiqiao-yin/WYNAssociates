import { useState, useRef, useEffect, useCallback } from 'react';
import { triggerSpotlight } from './spotlight';

const API_URL = 'https://u0bx8gasvk.execute-api.us-east-1.amazonaws.com/prod/chat';
const API_KEY = '3PFy6zibam4um8XGeavg3axSpEyP3FQs4JRS99Pu';

/* ---- SVG icons ---- */
const BotIcon = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="10" rx="3" />
    <circle cx="8.5" cy="16" r="1.5" fill="currentColor" />
    <circle cx="15.5" cy="16" r="1.5" fill="currentColor" />
    <path d="M12 2v4" />
    <path d="M8 6h8" />
    <path d="M9 11V8a3 3 0 0 1 6 0v3" />
  </svg>
);

const CloseIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const SpeakerIcon = ({ playing }) => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: playing ? 1 : 0.5, cursor: 'pointer' }}>
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill={playing ? 'currentColor' : 'none'} />
    {playing ? (
      <>
        <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
      </>
    ) : (
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
    )}
  </svg>
);

export default function ChatbotWidget({ onTabChange, activeTab, onOpenChange }) {
  const [open, setOpen] = useState(false);

  const toggleOpen = (val) => {
    setOpen(val);
    onOpenChange?.(val);
  };
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [speakingIdx, setSpeakingIdx] = useState(null);
  const messagesEndRef = useRef(null);
  const synthRef = useRef(window.speechSynthesis);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Stop TTS when panel closes
  useEffect(() => {
    if (!open) {
      synthRef.current.cancel();
      setSpeakingIdx(null);
    }
  }, [open]);

  const speak = useCallback((text, idx) => {
    const synth = synthRef.current;
    if (speakingIdx === idx) {
      synth.cancel();
      setSpeakingIdx(null);
      return;
    }
    synth.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1;
    utter.pitch = 1;
    utter.onend = () => setSpeakingIdx(null);
    utter.onerror = () => setSpeakingIdx(null);
    setSpeakingIdx(idx);
    synth.speak(utter);
  }, [speakingIdx]);

  const executeActions = (actions) => {
    if (!actions || !Array.isArray(actions)) return;

    let delay = 0;
    for (const action of actions) {
      if (action.type === 'navigate' && action.tab) {
        setTimeout(() => triggerSpotlight(`nav-${action.tab}`, 2000), delay);
        delay += 400;
        setTimeout(() => onTabChange(action.tab), delay);
        delay += 600;
      } else if (action.type === 'navigate_subtab' && action.subtab) {
        setTimeout(() => {
          window.dispatchEvent(
            new CustomEvent('wyn-subtab-navigate', {
              detail: { tab: action.tab, subtab: action.subtab },
            })
          );
        }, delay);
        delay += 500;
      } else if (action.type === 'switch_ai_view' && action.view) {
        setTimeout(() => {
          window.dispatchEvent(
            new CustomEvent('wyn-ai-view-navigate', {
              detail: { view: action.view },
            })
          );
        }, delay);
        delay += 500;
      } else if (action.type === 'spotlight' && action.target) {
        setTimeout(() => triggerSpotlight(action.target), delay);
        delay += 300;
      }
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: 'user', content: text };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': API_KEY,
        },
        body: JSON.stringify({ messages: updated }),
      });

      const raw = await res.json();
      // Recursively unwrap until we get { reply: "<plain text>", actions: [...] }
      const unwrap = (d) => {
        if (typeof d === 'string') {
          try { return unwrap(JSON.parse(d)); } catch { return { reply: d, actions: [] }; }
        }
        if (d && typeof d === 'object') {
          // API Gateway wrapper: { statusCode, body: "..." }
          if (typeof d.body === 'string') {
            try { return unwrap(JSON.parse(d.body)); } catch { /* fall through */ }
          }
          // reply itself might be a JSON string with reply/actions
          if (typeof d.reply === 'string' && d.reply.trim().startsWith('{')) {
            try {
              const inner = JSON.parse(d.reply);
              if (inner && typeof inner.reply === 'string') return unwrap(inner);
            } catch { /* reply is plain text */ }
          }
          if (typeof d.reply === 'string') return d;
        }
        return { reply: String(d), actions: [] };
      };
      const data = unwrap(raw);
      const reply = data.reply || 'Sorry, I could not process that.';
      const actions = data.actions || [];

      setMessages((prev) => [...prev, { role: 'assistant', content: reply }]);
      executeActions(actions);

      // Auto-speak the reply
      const synth = synthRef.current;
      synth.cancel();
      const utter = new SpeechSynthesisUtterance(reply);
      utter.rate = 1;
      utter.pitch = 1;
      const newIdx = updated.length; // index of the new assistant msg
      utter.onend = () => setSpeakingIdx(null);
      utter.onerror = () => setSpeakingIdx(null);
      setSpeakingIdx(newIdx);
      synth.speak(utter);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Overlay for mobile */}
      {open && <div className="chatbot-overlay" onClick={() => toggleOpen(false)} />}

      {!open && (
        <button id="chatbot-toggle" onClick={() => toggleOpen(true)} aria-label="Open WYN Assistant">
          <BotIcon />
        </button>
      )}

      <div id="chatbot-panel" className={open ? 'open' : ''}>
        <div className="chatbot-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BotIcon />
            <strong>WYN Assistant</strong>
          </div>
          <button className="chatbot-close-btn" onClick={() => toggleOpen(false)} aria-label="Close WYN Assistant">
            <CloseIcon />
          </button>
        </div>
        <div className="chatbot-messages">
          {messages.length === 0 && (
            <div className="chatbot-msg-assistant">
              Hi! I'm the WYN Assistant. Ask me anything about this site — I can navigate you to
              any section and highlight what you're looking for.
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={msg.role === 'user' ? 'chatbot-msg-user' : 'chatbot-msg-assistant'}
            >
              {msg.content}
              {msg.role === 'assistant' && (
                <button
                  className="tts-btn"
                  onClick={() => speak(msg.content, i)}
                  title={speakingIdx === i ? 'Stop' : 'Listen'}
                >
                  <SpeakerIcon playing={speakingIdx === i} />
                </button>
              )}
            </div>
          ))}
          {loading && <div className="chatbot-msg-assistant" style={{ opacity: 0.5 }}>Thinking...</div>}
          <div ref={messagesEndRef} />
        </div>
        <div className="chatbot-input-row">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading}>
            Send
          </button>
        </div>
      </div>
    </>
  );
}
