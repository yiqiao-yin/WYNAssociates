import MiniSymbolOverview from '../tradingview/MiniSymbolOverview';
import StockScreener from '../tradingview/StockScreener';
import { watchlistSymbols } from '../../data/watchlistSymbols';

export default function StocksSubTab() {
  return (
    <>
      <p>Your watchlist will be shown here.</p>
      <div className="grid-container" data-spotlight="watchlist-stocks-grid">
        {watchlistSymbols.map((symbol) => (
          <div className="widget-box" key={symbol}>
            <MiniSymbolOverview symbol={symbol} />
          </div>
        ))}
      </div>

      <details data-spotlight="watchlist-stock-screener">
        <summary>Expand/Collapse</summary>
        <StockScreener />
      </details>
    </>
  );
}
