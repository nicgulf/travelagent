import React from 'react';
import Sidebar from './components/Sidebar';
import ChatHeader from './components/ChatHeader';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="h-screen bg-gray-50 flex">
      {/* Sidebar - Hidden on mobile, visible on desktop */}
      <div className="hidden lg:block">
        <Sidebar />
      </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader />
        <div className="flex-1 min-h-0">
          <ChatInterface />
        </div>
      </div>
    </div>
  );
}

export default App;