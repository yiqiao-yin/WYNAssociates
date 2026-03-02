import { useState } from 'react';

const TABS = ['Home', 'Market', 'Watchlist', 'Portfolio', 'Research', 'Teaching'];

export default function Navbar({ onTabChange, activeTab }) {
  const [menuOpen, setMenuOpen] = useState(false);

  const handleTabClick = (tab) => {
    onTabChange(tab.toLowerCase());
    setMenuOpen(false);
  };

  return (
    <div className={`navbar${menuOpen ? ' open' : ''}`}>
      <button className="hamburger" onClick={() => setMenuOpen(!menuOpen)}>
        <span />
        <span />
        <span />
      </button>
      <div className="nav-links">
        {TABS.map((tab) => (
          <button
            key={tab}
            data-spotlight={`nav-${tab.toLowerCase()}`}
            className={activeTab === tab.toLowerCase() ? 'active' : ''}
            onClick={() => handleTabClick(tab)}
          >
            {tab}
          </button>
        ))}
      </div>
    </div>
  );
}
