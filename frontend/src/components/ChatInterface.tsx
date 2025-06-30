// src/components/ChatInterface.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader } from 'lucide-react';
import { chatService } from '../services/chatService';
import EnhancedFlightDisplay from './EnhancedFlightDisplay';
import { MonthSearchConfirmation } from './MonthSearchConfirmation';
import { MonthSearchProgress } from './MonthSearchProgress';

interface ChatMessage {
  id: string;
  message: string;
  type?: string;
  flights?: any[];
  data?: any;
  suggestions?: string[];
  timestamp: Date;
  sender: 'user' | 'bot';
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      message: "Hello! I'm your travel assistant. I can help you search for flights, find accommodations, and answer any travel-related questions. How can I assist you today?",
      timestamp: new Date(),
      sender: 'bot',
      suggestions: [
        'Search flights from Ahmedabad to Kochi',
        'Find flights for next week',
        'Show me flight options'
      ]
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  
  // ✅ NEW: Month search state
  const [showMonthWarning, setShowMonthWarning] = useState(false);
  const [isMonthSearching, setIsMonthSearching] = useState(false);
  const [monthSearchInfo, setMonthSearchInfo] = useState<any>(null);
  const [searchProgress, setSearchProgress] = useState({
    current: 0,
    total: 0,
    currentDate: '',
    flightsFound: 0
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // ✅ NEW: Month detection function
  const detectMonthSearch = (query: string): { isMonth: boolean; monthName?: string; estimatedDates?: number } => {
    const queryLower = query.toLowerCase();
    
    // Check for explicit month keywords first
    const monthKeywords = ['next month', 'this month', 'entire month', 'whole month', 'full month'];
    
    for (const keyword of monthKeywords) {
      if (queryLower.includes(keyword)) {
        return { 
          isMonth: true, 
          monthName: keyword.includes('next') ? 'Next Month' : 
                     keyword.includes('this') ? 'This Month' : 'Month',
          estimatedDates: 12 
        };
      }
    }
    
    // ✅ IMPROVED: More flexible month name detection
    const monthNames = ['january', 'february', 'march', 'april', 'may', 'june', 
                       'july', 'august', 'september', 'october', 'november', 'december'];
    
    for (const month of monthNames) {
      // Check for various month patterns
      const monthPatterns = [
        `flights in ${month}`,
        `flights for ${month}`, 
        `flights during ${month}`,
        `in ${month}`,
        `for ${month}`,           // ✅ This will catch "for July"
        `during ${month}`,
        `throughout ${month}`,
        `${month} flights`,
        `all of ${month}`,
        `entire ${month}`
      ];
      
      // Check if any pattern matches
      const hasMonthPattern = monthPatterns.some(pattern => queryLower.includes(pattern));
      
      // ✅ EXCLUDE specific date patterns
      const specificDatePatterns = [
        new RegExp(`\\d{1,2}\\s+${month}`),        // "15 july"
        new RegExp(`${month}\\s+\\d{1,2}`),        // "july 15"
        new RegExp(`\\d{1,2}(st|nd|rd|th)\\s+${month}`), // "15th july"
        new RegExp(`${month}\\s+\\d{1,2}(st|nd|rd|th)`)  // "july 15th"
      ];
      
      const hasSpecificDate = specificDatePatterns.some(pattern => pattern.test(queryLower));
      
      // Only return month search if it has month patterns AND no specific dates
      if (hasMonthPattern && !hasSpecificDate) {
        return { 
          isMonth: true, 
          monthName: month.charAt(0).toUpperCase() + month.slice(1),
          estimatedDates: 12 
        };
      }
    }
    
    return { isMonth: false };
  };

  // ✅ UPDATED: Modified handleSendMessage function
  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return;

    // ✅ NEW: Check for month search before sending
    const monthDetection = detectMonthSearch(inputValue);
    
    if (monthDetection.isMonth) {
      setMonthSearchInfo({
        monthName: monthDetection.monthName,
        estimatedDates: monthDetection.estimatedDates,
        originalQuery: inputValue
      });
      setShowMonthWarning(true);
      return; // Don't send the message yet
    }

    // Regular message sending logic
    await sendRegularMessage(inputValue);
  };

  // ✅ NEW: Extracted regular message logic
  const sendRegularMessage = async (message: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: message,
      timestamp: new Date(),
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await chatService.sendMessage(message);
      
      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: response.message,
        type: response.type,
        flights: response.flights,
        data: response.data,
        suggestions: response.suggestions,
        timestamp: new Date(),
        sender: 'bot'
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        sender: 'bot'
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // ✅ NEW: Month search handlers
  const handleMonthSearchConfirm = async () => {
    setShowMonthWarning(false);
    setIsMonthSearching(true);
    
    // Reset progress
    setSearchProgress({ current: 0, total: 12, currentDate: '', flightsFound: 0 });
    
    // Start the actual search
    await sendMonthSearch(monthSearchInfo.originalQuery);
    
    setIsMonthSearching(false);
  };

  const sendMonthSearch = async (query: string) => {
    // Add user message to chat
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: query,
      timestamp: new Date(),
      sender: 'user'
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    // Simulate progress updates (you can make this real by polling your backend)
    simulateProgress();
    
    try {
      // Send the actual search request
      const response = await chatService.sendMessage(query);
      
      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: response.message,
        type: response.type,
        flights: response.flights,
        data: response.data,
        suggestions: response.suggestions,
        timestamp: new Date(),
        sender: 'bot'
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error in month search:', error);
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: 'Sorry, I encountered an error during the month search. Please try again.',
        timestamp: new Date(),
        sender: 'bot'
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const simulateProgress = () => {
    let current = 0;
    const total = 12;
    
    const interval = setInterval(() => {
      current++;
      setSearchProgress(prev => ({
        current,
        total,
        currentDate: `2025-07-${String(current + 1).padStart(2, '0')}`,
        flightsFound: prev.flightsFound + Math.floor(Math.random() * 4) + 1
      }));
      
      if (current >= total) {
        clearInterval(interval);
      }
    }, 2500); // Update every 2.5 seconds
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const renderMessage = (message: ChatMessage) => {
    const isUser = message.sender === 'user';
    
    return (
      <div key={message.id} className={`mb-6 ${isUser ? 'flex justify-end' : 'flex justify-start'}`}>
        <div className={`max-w-4xl ${isUser ? 'order-2' : 'order-1'}`}>
          {/* Message bubble */}
          <div className={`
            rounded-lg px-4 py-3 mb-2
            ${isUser 
              ? 'bg-blue-500 text-white ml-auto max-w-md' 
              : 'bg-white border border-gray-200 shadow-sm'
            }
          `}>
            {isUser ? (
              <div className="whitespace-pre-wrap">{message.message}</div>
            ) : (
              <>
                {/* Bot message content */}
                {message.type === 'flight_results' && message.data?.flights ? (
                  <div>
                    {/* Brief text summary */}
                    <div className="mb-4 text-gray-800">
                      <div className="whitespace-pre-wrap">{message.message}</div>
                    </div>
                    
                    {/* Enhanced flight display */}
                    <EnhancedFlightDisplay 
                      flightData={message.data.flights} 
                      searchInfo={message.data.search_info} 
                    />
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap text-gray-800">{message.message}</div>
                )}
              </>
            )}
          </div>

          {/* Suggestions */}
          {!isUser && message.suggestions && message.suggestions.length > 0 && (
            <div className="mt-3 space-y-2">
              <p className="text-sm font-medium text-gray-600">Suggestions:</p>
              <div className="flex flex-wrap gap-2">
                {message.suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg px-3 py-1 transition-colors border border-blue-200"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Timestamp */}
          <div className={`text-xs text-gray-500 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen max-w-6xl mx-auto bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4 shadow-sm">
        <h1 className="text-xl font-semibold text-gray-800">Travel Assistant</h1>
        <p className="text-sm text-gray-600">Find flights, accommodations, and travel information</p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(renderMessage)}
        
        {loading && (
          <div className="flex justify-start mb-6">
            <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 shadow-sm">
              <div className="flex items-center gap-2 text-gray-600">
                <Loader className="animate-spin" size={16} />
                <span className="text-sm">Searching for flights...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4 shadow-lg">
        <div className="flex gap-2 max-w-4xl mx-auto">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about flights, hotels, or travel planning..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={1}
            style={{ minHeight: '40px', maxHeight: '120px' }}
          />
          <button
            onClick={handleSendMessage}
            disabled={loading || !inputValue.trim()}
            className="bg-blue-500 text-white p-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={20} />
          </button>
        </div>
      </div>

      {/* ✅ NEW: Month Search Warning Modal */}
      {showMonthWarning && monthSearchInfo && (
        <MonthSearchConfirmation
          onConfirm={handleMonthSearchConfirm}
          onCancel={() => setShowMonthWarning(false)}
          monthName={monthSearchInfo.monthName}
          estimatedDates={monthSearchInfo.estimatedDates}
        />
      )}

      {/* ✅ NEW: Month Search Progress */}
      <MonthSearchProgress
        currentDate={searchProgress.current}
        totalDates={searchProgress.total}
        monthName={monthSearchInfo?.monthName || 'Next Month'}
        currentSearchDate={searchProgress.currentDate}
        flightsFound={searchProgress.flightsFound}
        isVisible={isMonthSearching}
      />
    </div>
  );
};

export default ChatInterface;