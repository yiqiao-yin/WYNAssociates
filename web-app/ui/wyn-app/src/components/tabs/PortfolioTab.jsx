import TickerTape from '../tradingview/TickerTape';
import AdvancedChart from '../tradingview/AdvancedChart';

export default function PortfolioTab() {
  return (
    <>
      <h2 data-spotlight="portfolio-title">Portfolio</h2>
      <p>Your portfolio overview and stats go here.</p>
      <div data-spotlight="portfolio-tape">
        <TickerTape />
      </div>
      <div data-spotlight="portfolio-chart">
        <AdvancedChart />
      </div>
    </>
  );
}
