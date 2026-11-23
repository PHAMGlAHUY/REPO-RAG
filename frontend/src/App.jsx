import { useState } from 'react';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import ChatWindow from './components/chat/ChatWindow';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <main className="app__main">
        <Header onMenu={() => setSidebarOpen(true)} />
        <ChatWindow />
      </main>
    </div>
  );
}
