import { useEffect } from 'react';
import TickerTag from '../tradingview/TickerTag';
import { aiWatchlistData } from '../../data/aiWatchlistData';

const LAYER_SPOTLIGHT_MAP = {
  'Energy Layer': 'watchlist-ai-energy',
  'Chips Layer': 'watchlist-ai-chips',
  'Infrastructure': 'watchlist-ai-infrastructure',
  'Models': 'watchlist-ai-models',
  'Application': 'watchlist-ai-application',
};

export default function AIWatchlistSubTab() {
  useEffect(() => {
    // Load the TV ticker tag ES module once
    const existing = document.querySelector('script[src*="tv-ticker-tag"]');
    if (!existing) {
      const script = document.createElement('script');
      script.type = 'module';
      script.src = 'https://widgets.tradingview-widget.com/w/en/tv-ticker-tag.js';
      document.head.appendChild(script);
    }
  }, []);

  return (
    <>
      {Object.entries(aiWatchlistData).map(([layerName, symbols]) => (
        <div key={layerName} data-spotlight={LAYER_SPOTLIGHT_MAP[layerName]}>
          <h3>{layerName}</h3>
          {symbols.map((symbol) => (
            <TickerTag key={symbol} symbol={symbol} />
          ))}
        </div>
      ))}
    </>
  );
}
