import TradingViewWidget from './TradingViewWidget';

export default function StockScreener() {
  return (
    <TradingViewWidget
      scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-screener.js"
      config={{
        width: "100%",
        height: "850",
        defaultColumn: "overview",
        defaultScreen: "most_capitalized",
        market: "america",
        showToolbar: true,
        colorTheme: "dark",
        locale: "en",
        isTransparent: true,
      }}
    />
  );
}
