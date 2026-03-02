import { useState, useRef, useEffect } from 'react';
import { triggerSpotlight } from './spotlight';

const API_URL = 'https://u0bx8gasvk.execute-api.us-east-1.amazonaws.com/prod/chat';
const API_KEY = '3PFy6zibam4um8XGeavg3axSpEyP3FQs4JRS99Pu';

export default function ChatbotWidget({ onTabChange, activeTab }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const executeActions = (actions) => {
    if (!actions || !Array.isArray(actions)) return;

    let delay = 0;
    for (const action of actions) {
      if (action.type === 'navigate' && action.tab) {
        // Spotlight the nav button first for the visual "click" effect
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

      const data = await res.json();
      const reply = data.reply || 'Sorry, I could not process that.';
      const actions = data.actions || [];

      setMessages((prev) => [...prev, { role: 'assistant', content: reply }]);
      executeActions(actions);
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
      <button id="chatbot-toggle" onClick={() => setOpen(!open)}>
        {open ? '✕' : '💬'}
      </button>
      {open && (
        <div id="chatbot-panel">
          <div className="chatbot-header">
            <strong>WYN Assistant</strong>
            <span style={{ fontSize: '11px', opacity: 0.6 }}>Memory clears on refresh</span>
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
                className={
                  msg.role === 'user' ? 'chatbot-msg-user' : 'chatbot-msg-assistant'
                }
              >
                {msg.content}
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
      )}
    </>
  );
}
