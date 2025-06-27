import React from 'react';
import { Message } from '../types/chat';
import { Bot, User, Plane, MapPin } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isAgent = message.sender === 'agent';
  
  // Enhanced message rendering for different content types
  const renderMessageContent = () => {
    // If message has flight data, render it specially
    if (message.data?.flights && message.data.flights.length > 0) {
      return (
        <div className="space-y-3">
          <p className="text-sm leading-relaxed whitespace-pre-line">{message.content}</p>
          <FlightResultsPreview flights={message.data.flights.slice(0, 2)} />
          {message.data.flights.length > 2 && (
            <p className="text-xs text-gray-600">
              +{message.data.flights.length - 2} more flights available
            </p>
          )}
        </div>
      );
    }

    // If message has airport/airline info
    if (message.data?.type === 'airport_info' || message.data?.type === 'airline_info') {
      return (
        <div className="space-y-2">
          <p className="text-sm leading-relaxed whitespace-pre-line">{message.content}</p>
          {message.data.type === 'airport_info' && (
            <div className="flex items-center text-xs text-gray-600">
              <MapPin className="w-3 h-3 mr-1" />
              Airport Information
            </div>
          )}
        </div>
      );
    }

    // Default text message
    return <p className="text-sm leading-relaxed whitespace-pre-line">{message.content}</p>;
  };

  return (
    <div className={`flex items-start space-x-3 ${isAgent ? '' : 'flex-row-reverse space-x-reverse'} animate-fadeIn`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isAgent 
          ? 'bg-gradient-to-br from-sky-500 to-blue-600' 
          : 'bg-gradient-to-br from-gray-600 to-gray-700'
      }`}>
        {isAgent ? (
          <Bot className="w-4 h-4 text-white" />
        ) : (
          <User className="w-4 h-4 text-white" />
        )}
      </div>
      
      <div className={`max-w-xs lg:max-w-md xl:max-w-lg ${isAgent ? '' : 'flex flex-col items-end'}`}>
        <div className={`rounded-2xl px-4 py-3 shadow-sm ${
          isAgent 
            ? 'bg-white border border-gray-200 text-gray-800' 
            : 'bg-gradient-to-r from-sky-500 to-blue-600 text-white'
        }`}>
          {renderMessageContent()}
        </div>
        
        {/* Message timestamp */}
        <p className={`text-xs text-gray-500 mt-1 px-1 ${isAgent ? '' : 'text-right'}`}>
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>

        {/* Suggestions (only for agent messages) */}
        {isAgent && message.suggestions && message.suggestions.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {message.suggestions.map((suggestion, index) => (
              <button
                key={index}
                className="px-2 py-1 text-xs bg-sky-50 text-sky-600 rounded-full hover:bg-sky-100 transition-colors border border-sky-200"
                onClick={() => {
                  // You can emit an event or call a parent function here
                  console.log('Suggestion clicked:', suggestion);
                  // Example: onSuggestionClick?.(suggestion);
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Mini component to show flight preview in messages
const FlightResultsPreview: React.FC<{ flights: any[] }> = ({ flights }) => {
  return (
    <div className="space-y-2">
      {flights.map((flight, index) => (
        <div key={index} className="bg-gray-50 rounded-lg p-2 text-xs">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Plane className="w-3 h-3 text-sky-600" />
              <span className="font-medium">{flight.flight_number}</span>
              <span className="text-gray-600">
                {flight.departure_time} → {flight.arrival_time}
              </span>
            </div>
            <span className="font-bold text-green-600">{flight.price}</span>
          </div>
          <div className="text-gray-500 mt-1">
            {flight.departure_airport} → {flight.arrival_airport}
            {flight.is_direct ? ' (Direct)' : ` (${flight.stops} stop${flight.stops > 1 ? 's' : ''})`}
          </div>
        </div>
      ))}
    </div>
  );
};

export default MessageBubble;