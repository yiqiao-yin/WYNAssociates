import { useState, useEffect } from 'react';
import StocksSubTab from '../watchlist/StocksSubTab';
import AIWatchlistSubTab from '../watchlist/AIWatchlistSubTab';

export default function WatchlistTab() {
  const [activeSubTab, setActiveSubTab] = useState('stocks');

  useEffect(() => {
    const handler = (e) => {
      const { subtab } = e.detail || {};
      if (subtab) setActiveSubTab(subtab);
    };
    window.addEventListener('wyn-subtab-navigate', handler);
    return () => window.removeEventListener('wyn-subtab-navigate', handler);
  }, []);

  return (
    <>
      <h2 data-spotlight="watchlist-title">Watchlist</h2>

      <div className="sub-navbar">
        <button
          data-spotlight="watchlist-stocks-btn"
          className={activeSubTab === 'stocks' ? 'active' : ''}
          onClick={() => setActiveSubTab('stocks')}
        >
          Stocks
        </button>
        <button
          data-spotlight="watchlist-ai-btn"
          className={activeSubTab === 'ai' ? 'active' : ''}
          onClick={() => setActiveSubTab('ai')}
        >
          AI Watchlist
        </button>
      </div>

      <div className={`sub-tab-content${activeSubTab === 'stocks' ? ' active' : ''}`}>
        <StocksSubTab />
      </div>
      <div className={`sub-tab-content${activeSubTab === 'ai' ? ' active' : ''}`}>
        <AIWatchlistSubTab />
      </div>
    </>
  );
}
