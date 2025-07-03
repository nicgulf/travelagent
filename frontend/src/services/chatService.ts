import { travelService } from './travelService';
import { ChatResponse, FlightInfo } from './types';


export const detectMonthSearch = (query: string): { isMonth: boolean; monthName?: string; estimatedDates?: number } => {
  const monthKeywords = ['next month', 'this month', 'entire month', 'whole month'];
  const monthNames = ['january', 'february', 'march', 'april', 'may', 'june', 
                     'july', 'august', 'september', 'october', 'november', 'december'];
  
  const queryLower = query.toLowerCase();
  
  // Check for month keywords
  for (const keyword of monthKeywords) {
    if (queryLower.includes(keyword)) {
      return { 
        isMonth: true, 
        monthName: keyword.includes('next') ? 'Next Month' : 'This Month',
        estimatedDates: 12 
      };
    }
  }
  
  // Check for specific month names
  for (const month of monthNames) {
    if (queryLower.includes(month)) {
      return { 
        isMonth: true, 
        monthName: month.charAt(0).toUpperCase() + month.slice(1),
        estimatedDates: 12 
      };
    }
  }
  
  return { isMonth: false };
};
export const chatService = {
  async sendMessage(input: any): Promise<ChatResponse> {
    try {
      let message: string;
      
      if (typeof input === 'string') {
        message = input;
      } else if (input && typeof input === 'object' && input.content) {
        message = input.content;
      } else if (input && typeof input === 'object' && input.message) {
        message = input.message;
      } else {
        message = String(input || '');
      }

      if (!message || typeof message !== 'string' || message.trim().length === 0) {
        return {
          message: "Please enter a valid message.",
          type: 'error'
        };
      }

      console.log('🔍 Processing message:', message);

      if (this.isFlightQuery(message)) {
        return await this.handleFlightQueryWithRetry(message);
      }
      
      return await this.handleGeneralQuery(message);
      
    } catch (error: unknown) {
      console.error('Chat service error:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      return {
        message: `I'm sorry, I encountered an error processing your request: ${errorMessage}. Please try again.`,
        type: 'error'
      };
    }
  },

  isFlightQuery(input: any): boolean {
    try {
      const message = typeof input === 'string' ? input : String(input || '');
      
      if (!message || message.trim().length === 0) {
        return false;
      }

      const flightKeywords = [
        'flight', 'fly', 'book', 'search', 'from', 'to', 'travel',
        'airport', 'departure', 'arrival', 'trip', 'journey', 'AMD', 'COK',
        'Ahmedabad', 'Kochi'
      ];
      
      const lowerMessage = message.toLowerCase();
      return flightKeywords.some(keyword => lowerMessage.includes(keyword));
    } catch (error: unknown) {
      console.error('Error in isFlightQuery:', error);
      return false;
    }
  },

  // NEW: Handle flight query with retry mechanism for 422 errors
  async handleFlightQueryWithRetry(message: string): Promise<ChatResponse> {
    console.log('🛫 Handling flight query with retry logic:', message);
    
    // Strategy 1: Try the main method
    try {
      const response = await travelService.searchFlightsNaturalLanguage(message);
      console.log('✅ Main method response:', response);
      
      if (response && response.status === 'success') {
        return this.formatFlightResults(response.data, message);
      }
      
      // If we get an error response, check if it's a 422
      if (response && response.status === 'error') {
        const errorMessage = response.error || '';
        
        // If it's a 422 error, try alternative approaches
        if (errorMessage.includes('422') || errorMessage.includes('API Error: 422')) {
          console.log('🔄 Detected 422 error, trying alternative methods...');
          return await this.handle422ErrorWithAlternatives(message);
        }
        
        // For other errors, return them as-is
        return {
          message: response.error || 'An error occurred while searching for flights.',
          suggestions: response.suggestions || [
            'Try rephrasing your query',
            'Check if the API server is running'
          ],
          type: 'error'
        };
      }
      
    } catch (error: unknown) {
      console.error('❌ Main method failed:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      // If the main method fails, try alternatives
      console.log('🔄 Main method failed, trying alternatives...');
      return await this.handle422ErrorWithAlternatives(message);
    }
    
    // Fallback
    return {
      message: 'Unable to process flight search. Please try again.',
      suggestions: ['Check your internet connection', 'Try a simpler query'],
      type: 'error'
    };
  },

  // NEW: Handle 422 errors with alternative strategies
  async handle422ErrorWithAlternatives(originalMessage: string): Promise<ChatResponse> {
    console.log('🔧 Handling 422 error with alternatives for:', originalMessage);
    
    // Strategy 2: Try alternative method
    try {
      const altResponse = await travelService.searchFlightsAlternative(originalMessage);
      console.log('🔄 Alternative method response:', altResponse);
      
      if (altResponse && altResponse.status === 'success') {
        return this.formatFlightResults(altResponse.data, originalMessage);
      }
    } catch (altError: unknown) {
      console.error('❌ Alternative method failed:', altError);
    }
    
    // Strategy 3: Try simplified queries
    const simplifiedQueries = this.generateSimplifiedQueries(originalMessage);
    
    for (const simplifiedQuery of simplifiedQueries) {
      try {
        console.log(`🔄 Trying simplified query: "${simplifiedQuery}"`);
        
        const response = await travelService.searchFlightsNaturalLanguage(simplifiedQuery);
        
        if (response && response.status === 'success') {
          console.log('✅ Simplified query succeeded!');
          return this.formatFlightResults(response.data, originalMessage);
        }
        
      } catch (error) {
        console.log(`❌ Simplified query "${simplifiedQuery}" failed:`, error);
        continue; // Try the next simplified query
      }
    }
    
    // Strategy 4: Debug and provide helpful error message
    try {
      const debugInfo = await travelService.debugServerConnection();
      console.log('🔬 Debug info:', debugInfo);
      
      if (debugInfo.error) {
        return {
          message: 'Currently unable to connect to the flight search service.',
          suggestions: [
            'Check that internet connection is stable',
            'Restart the page',
            'Try after a few minutes'
          ],
          type: 'error'
        };
      }
      
    } catch (debugError: unknown) {
      console.error('❌ Debug failed:', debugError);
    }
    
    // Final fallback with helpful 422-specific guidance
    return {
      message: 'I\'m having trouble with the flight search request format (Error 422). This usually means the API server needs a specific format.',
      suggestions: [
        'Try: "flights from Ahmedabad to Kochi"',
        'Try: "search AMD to COK"',
        'Check if the API server is properly configured',
        'Review server logs for detailed error information'
      ],
      type: 'error'
    };
  },

  // NEW: Generate simplified queries to try
  generateSimplifiedQueries(originalMessage: string): string[] {
    const simplified = [];
    
    // Extract city names or airport codes
    const message = originalMessage.toLowerCase();
    
    if (message.includes('ahmedabad') && message.includes('kochi')) {
      simplified.push('flights from Ahmedabad to Kochi');
      simplified.push('AMD to COK');
      simplified.push('search flights AMD COK');
    } else if (message.includes('amd') && message.includes('cok')) {
      simplified.push('AMD to COK');
      simplified.push('flights AMD COK');
    }
    
    // Generic patterns
    simplified.push('search flights');
    simplified.push('flights');
    
    return simplified.filter(q => q !== originalMessage); // Don't repeat the original
  },

  // UPDATED: Format flight results with better error handling
  formatFlightResults(data: any, originalQuery: string): ChatResponse {
    console.log('🎯 Formatting flight results with enhanced display:', data);
    
    if (!data) {
      return {
        message: 'No flight data received from the server.',
        suggestions: ['Try again', 'Check server connection'],
        type: 'error'
      };
    }
    
    if (data.error) {
      return {
        message: data.error,
        suggestions: data.suggestions || [
          'Please use valid city names like Mumbai, Delhi',
          'Try airport codes like BOM, DEL'
        ],
        type: 'error'
      };
    }
  
    const flights: any[] = data.flights || [];
    const searchInfo = data.search_info || {};
    
    if (flights.length === 0) {
      return {
        message: `No flights found from ${searchInfo.origin || 'your origin'} to ${searchInfo.destination || 'your destination'} on ${searchInfo.search_date || 'the selected date'}`,
        suggestions: [
          'Try a different date',
          'Check alternative nearby airports',
          'Verify city names are correct'
        ]
      };
    }
  
    // Sort flights by price
    flights.sort((a, b) => {
      const priceA = typeof a.price_numeric === 'number' ? a.price_numeric : parseFloat(a.price.replace(/[^\d.]/g, ''));
      const priceB = typeof b.price_numeric === 'number' ? b.price_numeric : parseFloat(b.price.replace(/[^\d.]/g, ''));
      return priceA - priceB;
    });
    
    const cheapest = flights[0];
    const directFlights = flights.filter(f => f.is_direct);
    const fastestFlight = flights.reduce((fastest, current) => {
      const currentDuration = this.parseDuration(current.duration);
      const fastestDuration = this.parseDuration(fastest.duration);
      return currentDuration < fastestDuration ? current : fastest;
    });
    
    // Enhanced airline name mapping
    const airlineNames: Record<string, string> = {
      'AI': 'Air India', '6E': 'IndiGo', 'SG': 'SpiceJet', 
      'UK': 'Vistara', 'EK': 'Emirates', 'QR': 'Qatar Airways',
      'EY': 'Etihad Airways', 'WY': 'Oman Air', 'KU': 'Kuwait Airways',
      'LH': 'Lufthansa', 'BA': 'British Airways', 'AF': 'Air France',
      'KL': 'KLM', 'TK': 'Turkish Airlines', 'SU': 'Aeroflot',
      'G8': 'Go Air', 'I5': 'AirAsia India'
    };
    
    const originCity = searchInfo.origin || 'AMD';
    const destCity = searchInfo.destination || 'COK';
    
    // Create a condensed summary for the chat (the enhanced display will show full details)
    let message = `✈️ **Found ${flights.length} flights from ${originCity} to ${destCity}**\n`;
    message += `📅 **Date:** ${searchInfo.search_date || 'Selected date'} • **Passengers:** ${searchInfo.passengers || 1}\n\n`;
    
    // Quick summary
    message += `💰 **Best Deals:**\n`;
    message += `• Cheapest: ${cheapest.price} (${airlineNames[cheapest.airline] || cheapest.airline})\n`;
    message += `• Fastest: ${this.formatDuration(fastestFlight.duration)} (${airlineNames[fastestFlight.airline] || fastestFlight.airline})\n`;
    
    if (directFlights.length > 0) {
      message += `• Direct flights: ${directFlights.length} available\n`;
    } else {
      message += `• All flights have connections (${flights[0].stops || 1} stop average)\n`;
    }
    
    // Price range
    const prices = flights.map(f => f.price_numeric);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    message += `• Price range: €${minPrice.toFixed(2)} - €${maxPrice.toFixed(2)}\n\n`;
    
    message += `📋 **Interactive flight details are displayed below with full connection information, sorting options, and booking capabilities.**`;
  
    // Return both the summary text AND the data for the enhanced display
    return {
      message,
      type: 'flight_results',
      flights: flights,
      data: {
        flights: flights, // This is what EnhancedFlightDisplay needs
        search_info: searchInfo, // This is what EnhancedFlightDisplay needs
        originalQuery: originalQuery,
        insights: {
          cheapest: cheapest,
          fastest: fastestFlight,
          bestDirect: directFlights[0] || null,
          priceRange: maxPrice - minPrice,
          averagePrice: prices.reduce((sum, price) => sum + price, 0) / prices.length,
          airlineCount: new Set(flights.map(f => f.airline)).size,
          totalFlights: flights.length
        }
      },
      suggestions: [
        'Show only direct flights',
        'Filter by specific airline',
        'Find return flights',
        'Check different dates',
        'Compare with nearby airports'
      ]
    };
  },
  
  // Helper methods (add these if they don't exist):
  parseDuration(duration: string): number {
    if (!duration) return 0;
    
    // Handle PT format (e.g., "PT9H45M")
    const ptMatch = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?/);
    if (ptMatch) {
      const hours = parseInt(ptMatch[1] || '0');
      const minutes = parseInt(ptMatch[2] || '0');
      return hours * 60 + minutes;
    }
    
    // Handle simple format (e.g., "2h 30m")
    const simpleMatch = duration.match(/(\d+)h?\s*(\d+)?m?/i);
    if (simpleMatch) {
      const hours = parseInt(simpleMatch[1] || '0');
      const minutes = parseInt(simpleMatch[2] || '0');
      return hours * 60 + minutes;
    }
    
    return 0;
  },
  
  formatDuration(duration: string): string {
    if (!duration) return 'N/A';
    
    const totalMinutes = this.parseDuration(duration);
    if (totalMinutes === 0) return duration; // Return original if couldn't parse
    
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `${hours}h ${minutes}m`;
  },

  async handleGeneralQuery(message: string): Promise<ChatResponse> {
    const suggestions = [
      'Search for flights from Ahmedabad to Kochi',
      'Find flights for next week',
      'Show me flight options'
    ];

    if (message.toLowerCase().includes('hello') || message.toLowerCase().includes('hi')) {
      return {
        message: "Hello! I'm your travel assistant. I can help you search for flights. What would you like to do?",
        suggestions
      };
    }

    return {
      message: "I can help you with flight searches. Try asking me to find flights between cities!",
      suggestions
    };
  }
};