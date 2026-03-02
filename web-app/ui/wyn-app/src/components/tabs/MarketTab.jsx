import StockHeatmap from '../tradingview/StockHeatmap';
import CryptoHeatmap from '../tradingview/CryptoHeatmap';

export default function MarketTab() {
  return (
    <>
      <h2 data-spotlight="market-title">Market</h2>
      <p>Market information and updates will appear here.</p>

      <h2 data-spotlight="market-heatmaps"><b>Heatmaps</b></h2>
      <p>The heatmaps contain large volume of data and may not be present during trading hours.</p>
      <details>
        <summary>Expand/Collapse</summary>
        <div>
          <div className="row_heatmap">
            <div className="column" data-spotlight="market-stock-heatmap">
              <h2><b>Stock Heatmap</b></h2>
              <StockHeatmap />
            </div>
            <div className="column" data-spotlight="market-crypto-heatmap">
              <h2><b>Crypto Coins Heatmap</b></h2>
              <CryptoHeatmap />
            </div>
          </div>
        </div>
      </details>
    </>
  );
}
