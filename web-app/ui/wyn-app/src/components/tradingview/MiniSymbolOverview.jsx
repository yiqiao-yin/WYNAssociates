import TradingViewWidget from './TradingViewWidget';

export default function MiniSymbolOverview({ symbol }) {
  return (
    <TradingViewWidget
      scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
      config={{
        symbol,
        width: "100%",
        height: 220,
        locale: "en",
        dateRange: "12M",
        colorTheme: "dark",
        isTransparent: false,
        autosize: true,
        largeChartUrl: "",
        chartOnly: false,
      }}
    />
  );
}
