// Enhanced travelService.ts - Using new flight search API with fallback
import { apiClient } from './api';

export const travelService = {
  // Parse natural language flight queries into structured parameters
  parseFlightQuery(query: string): any {
    const lowerQuery = query.toLowerCase();
    
    // Common city mappings
    const cityMappings: { [key: string]: string } = {
      'ahmedabad': 'Ahmedabad',
      'kochi': 'Kochi', 
      'mumbai': 'Mumbai',
      'delhi': 'Delhi',
      'bangalore': 'Bangalore',
      'chennai': 'Chennai',
      'kolkata': 'Kolkata',
      'hyderabad': 'Hyderabad'
    };
    
    // Extract origin and destination
    let origin = '';
    let destination = '';
    
    // Look for "from X to Y" pattern
    const fromToMatch = lowerQuery.match(/from\s+(\w+)\s+to\s+(\w+)/);
    if (fromToMatch) {
      origin = cityMappings[fromToMatch[1]] || fromToMatch[1];
      destination = cityMappings[fromToMatch[2]] || fromToMatch[2];
    } else {
      // Look for "X to Y" pattern
      const toMatch = lowerQuery.match(/(\w+)\s+to\s+(\w+)/);
      if (toMatch) {
        origin = cityMappings[toMatch[1]] || toMatch[1];
        destination = cityMappings[toMatch[2]] || toMatch[2];
      }
    }
    
    // Extract date
    let departure_date = '';
    if (lowerQuery.includes('tomorrow')) {
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      departure_date = tomorrow.toISOString().split('T')[0];
    } else if (lowerQuery.includes('today')) {
      departure_date = new Date().toISOString().split('T')[0];
    }
    
    // Extract class
    let travel_class = 'ECONOMY';
    if (lowerQuery.includes('business')) {
      travel_class = 'BUSINESS';
    } else if (lowerQuery.includes('first')) {
      travel_class = 'FIRST';
    }
    
    return {
      origin,
      destination,
      departure_date,
      travel_class,
      passengers: 1
    };
  },

  async searchFlightsNaturalLanguage(query: string): Promise<any> {
    try {
      console.log('🔍 Calling enhanced API with query:', query);
      
      // Validate input
      if (!query || typeof query !== 'string' || query.trim().length === 0) {
        throw new Error('Query cannot be empty');
      }
      
      // Try conversation endpoint first (for natural language processing)
      try {
        const conversationPayload = {
          message: query.trim(),
          session_id: 'session_' + Date.now(),
          user_id: 'user_' + Date.now()
        };
        
        console.log('📤 Trying conversation endpoint with payload:', conversationPayload);
        
        const conversationResponse = await fetch('http://localhost:8000/conversation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(conversationPayload)
        });
        
        console.log('📡 Conversation Response Status:', conversationResponse.status);
        
        if (conversationResponse.ok) {
          const conversationData = await conversationResponse.json();
          console.log('✅ Conversation endpoint succeeded:', conversationData);
          
          // If conversation endpoint worked and has flights, return the result
          if (conversationData.response && conversationData.response.flights && conversationData.response.flights.length > 0) {
            return {
              status: 'success',
              data: conversationData.response,
              source: 'conversation_endpoint'
            };
          }
        }
        
        console.log('🔄 Conversation endpoint failed or no flights, trying direct flight search...');
        
      } catch (conversationError) {
        console.log('❌ Conversation endpoint error:', conversationError);
        console.log('🔄 Falling back to direct flight search...');
      }
      
      // Fallback: Try to parse the query and use direct flight search with fallback enabled
      const flightSearchParams = this.parseFlightQuery(query);
      
      if (flightSearchParams.origin && flightSearchParams.destination) {
        console.log('📤 Using direct flight search with params:', flightSearchParams);
        
        const searchUrl = new URL('http://localhost:8000/flights/search');
        searchUrl.searchParams.append('origin', flightSearchParams.origin);
        searchUrl.searchParams.append('destination', flightSearchParams.destination);
        searchUrl.searchParams.append('departure_date', flightSearchParams.departure_date || '2025-07-04');
        searchUrl.searchParams.append('travel_class', flightSearchParams.travel_class || 'ECONOMY');
        searchUrl.searchParams.append('passengers', flightSearchParams.passengers?.toString() || '1');
        searchUrl.searchParams.append('use_fallback', 'true'); // Enable fallback for demo data
        
        const directResponse = await fetch(searchUrl.toString(), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        });
        
        console.log('📡 Direct Flight Search Status:', directResponse.status);
        
        if (directResponse.ok) {
          const directData = await directResponse.json();
          console.log('✅ Direct flight search succeeded:', directData);
          return directData;
        } else {
          const errorText = await directResponse.text();
          console.log('❌ Direct flight search failed:', errorText);
        }
      }
      
      // If all methods failed, return a helpful error
      return {
        status: 'error',
        error: 'Unable to process flight search request',
        suggestions: [
          'Try: "flights from Ahmedabad to Kochi"',
          'Try: "AMD to COK flights tomorrow"',
          'Check if the API server is running'
        ]
      };
      
    } catch (error) {
      console.error('💥 Unexpected error in searchFlightsNaturalLanguage:', error);
      
      return {
        status: 'error',
        error: `Unexpected error: ${error.message}`,
        suggestions: [
          'Check your network connection',
          'Ensure API server is running',
          'Try again in a moment'
        ]
      };
    }
  },

  // Alternative method for testing
  async searchFlightsAlternative(query: string): Promise<any> {
    console.log('🔄 Using alternative search method for:', query);
    return this.searchFlightsNaturalLanguage(query);
  },

  // Debug method to test server responsiveness
  async debugServerConnection(): Promise<any> {
    try {
      console.log('🔬 Testing server connection...');
      
      // Test health endpoint
      const healthTest = await fetch('http://localhost:8000/health');
      console.log('📡 Health endpoint status:', healthTest.status);
      
      if (healthTest.ok) {
        const healthData = await healthTest.json();
        console.log('📡 Health endpoint data:', healthData);
        return {
          status: 'success',
          health_check: healthData
        };
      }
      
      return {
        status: 'error',
        error: 'Health check failed'
      };
      
    } catch (error) {
      console.error('🔴 Server connection test failed:', error);
      return {
        error: error.message,
        suggestion: 'Server may not be running on port 8000'
      };
    }
  }
};
