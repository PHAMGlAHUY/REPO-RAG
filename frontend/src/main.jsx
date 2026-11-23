import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ChatProvider } from './store/ChatProvider';
import './styles/tokens.css';
import './styles/global.css';
import './styles/layout.css';
import './styles/chat.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ChatProvider>
      <App />
    </ChatProvider>
  </React.StrictMode>
);
