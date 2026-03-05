import { useEffect, useState } from 'react';
import TickerTag from '../tradingview/TickerTag';
import { aiWatchlistData } from '../../data/aiWatchlistData';
import AINetworkGraph from './AINetworkGraph';

const LAYER_SPOTLIGHT_MAP = {
  'Energy Layer': 'watchlist-ai-energy',
  'Chips Layer': 'watchlist-ai-chips',
  'Infrastructure': 'watchlist-ai-infrastructure',
  'Models': 'watchlist-ai-models',
  'Application': 'watchlist-ai-application',
};

export default function AIWatchlistSubTab() {
  const [view, setView] = useState('graph'); // 'graph' or 'list'

  // Listen for chatbot-driven view switches
  useEffect(() => {
    const handler = (e) => {
      const { view: targetView } = e.detail || {};
      if (targetView === 'graph' || targetView === 'list') {
        setView(targetView);
      }
    };
    window.addEventListener('wyn-ai-view-navigate', handler);
    return () => window.removeEventListener('wyn-ai-view-navigate', handler);
  }, []);

  useEffect(() => {
    if (view !== 'list') return;
    const existing = document.querySelector('script[src*="tv-ticker-tag"]');
    if (!existing) {
      const script = document.createElement('script');
      script.type = 'module';
      script.src = 'https://widgets.tradingview-widget.com/w/en/tv-ticker-tag.js';
      document.head.appendChild(script);
    }
  }, [view]);

  return (
    <>
      {/* View toggle */}
      <div style={{ marginBottom: 12, display: 'flex', gap: 6 }} data-spotlight="watchlist-ai-view-toggle">
        <button
          className={view === 'graph' ? 'active' : ''}
          onClick={() => setView('graph')}
          data-spotlight="watchlist-ai-3d-btn"
          style={{
            background: view === 'graph' ? '#555' : 'none',
            border: '1px solid #555', color: view === 'graph' ? 'white' : '#aaa',
            padding: '6px 14px', borderRadius: 3, cursor: 'pointer',
            fontFamily: '"Times New Roman", serif', fontSize: 14,
          }}
        >
          3D Supply Chain
        </button>
        <button
          className={view === 'list' ? 'active' : ''}
          onClick={() => setView('list')}
          data-spotlight="watchlist-ai-list-btn"
          style={{
            background: view === 'list' ? '#555' : 'none',
            border: '1px solid #555', color: view === 'list' ? 'white' : '#aaa',
            padding: '6px 14px', borderRadius: 3, cursor: 'pointer',
            fontFamily: '"Times New Roman", serif', fontSize: 14,
          }}
        >
          Ticker List
        </button>
      </div>

      {view === 'graph' ? (
        <div data-spotlight="watchlist-ai-3d-graph">
          <AINetworkGraph />
        </div>
      ) : (
        Object.entries(aiWatchlistData).map(([layerName, symbols]) => (
          <div key={layerName} data-spotlight={LAYER_SPOTLIGHT_MAP[layerName]}>
            <h3>{layerName}</h3>
            {symbols.map((symbol) => (
              <TickerTag key={symbol} symbol={symbol} />
            ))}
          </div>
        ))
      )}
    </>
  );
}
