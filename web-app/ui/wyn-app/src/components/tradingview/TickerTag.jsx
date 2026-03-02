import { useEffect, useRef } from 'react';

export default function TickerTag({ symbol }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.setAttribute('symbol', symbol);
    }
  }, [symbol]);

  return <tv-ticker-tag ref={ref} symbol={symbol} />;
}
