import TradingViewWidget from './TradingViewWidget';
import { portfolioSymbols } from '../../data/portfolioSymbols';

export default function TickerTape() {
  return (
    <TradingViewWidget
      scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js"
      config={{
        symbols: portfolioSymbols,
        showSymbolLogo: true,
        isTransparent: false,
        displayMode: "adaptive",
        colorTheme: "dark",
        locale: "en",
      }}
    />
  );
}
