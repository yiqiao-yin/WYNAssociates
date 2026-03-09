import { useState } from 'react';
import Navbar from './components/Navbar';
import ChatbotWidget from './components/ChatbotWidget';
import HomeTab from './components/tabs/HomeTab';
import MarketTab from './components/tabs/MarketTab';
import WatchlistTab from './components/tabs/WatchlistTab';
import PortfolioTab from './components/tabs/PortfolioTab';
import LettersTab from './components/tabs/LettersTab';
import ResearchTab from './components/tabs/ResearchTab';
import TeachingTab from './components/tabs/TeachingTab';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <div className={`app-wrapper${chatOpen ? ' chat-open' : ''}`}>
      <Navbar onTabChange={setActiveTab} activeTab={activeTab} />

      <div id="home" className={`tab-content${activeTab === 'home' ? ' active' : ''}`}>
        <HomeTab />
      </div>
      <div id="market" className={`tab-content${activeTab === 'market' ? ' active' : ''}`}>
        <MarketTab />
      </div>
      <div id="watchlist" className={`tab-content${activeTab === 'watchlist' ? ' active' : ''}`}>
        <WatchlistTab />
      </div>
      <div id="portfolio" className={`tab-content${activeTab === 'portfolio' ? ' active' : ''}`}>
        <PortfolioTab />
      </div>
      <div id="letters" className={`tab-content${activeTab === 'letters' ? ' active' : ''}`}>
        <LettersTab />
      </div>
      <div id="research" className={`tab-content${activeTab === 'research' ? ' active' : ''}`}>
        <ResearchTab />
      </div>
      <div id="teaching" className={`tab-content${activeTab === 'teaching' ? ' active' : ''}`}>
        <TeachingTab />
      </div>

      <footer className="site-footer">
        <p>&copy; 2010–{new Date().getFullYear()} Yiqiao Yin. All rights reserved.</p>
      </footer>

      <ChatbotWidget onTabChange={setActiveTab} activeTab={activeTab} onOpenChange={setChatOpen} />
    </div>
  );
}

export default App;
