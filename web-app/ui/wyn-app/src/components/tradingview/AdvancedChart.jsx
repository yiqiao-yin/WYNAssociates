import TradingViewWidget from './TradingViewWidget';

export default function AdvancedChart() {
  return (
    <TradingViewWidget
      scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
      config={{
        width: "100%",
        height: "810",
        symbol: "TSLA",
        interval: "W",
        timezone: "America/New_York",
        theme: "dark",
        style: "1",
        locale: "en",
        withdateranges: true,
        hide_side_toolbar: false,
        allow_symbol_change: true,
        studies: ["STD;MA%Ribbon", "STD;RSI"],
        show_popup_button: true,
        popup_width: "1000",
        popup_height: "650",
        support_host: "https://www.tradingview.com",
      }}
    />
  );
}
