import { useState, useCallback } from 'react';
import { Message } from '../types/chat';
import { chatService, ChatResponse } from '../services/chatService';

export const useChat = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! I'm your personal travel assistant. I'm here to help you plan your perfect trip, book flights, find accommodations, and answer any travel-related questions you might have. How can I assist you today?",
      sender: 'agent',
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    // Add user message immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Send to API
      const response: ChatResponse = await chatService.sendMessage({
        message: content,
        context: {
          previousMessages: messages.slice(-5), // Send last 5 messages for context
        },
      });

      // Add agent response
      const agentMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message,
        sender: 'agent',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, agentMessage]);

      // Handle suggestions if provided
      if (response.suggestions && response.suggestions.length > 0) {
        // You can handle suggestions here, maybe show them as quick actions
        console.log('Suggestions received:', response.suggestions);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
        sender: 'agent',
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [messages]);

  const handleQuickAction = useCallback(async (action: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response: ChatResponse = await chatService.getQuickActionResponse(action);
      
      // Add user message for the action
      const userMessage: Message = {
        id: Date.now().toString(),
        content: action,
        sender: 'user',
        timestamp: new Date(),
      };

      // Add agent response
      const agentMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message,
        sender: 'agent',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, userMessage, agentMessage]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    handleQuickAction,
  };
};