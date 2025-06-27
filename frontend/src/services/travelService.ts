// Updated travelService.ts - Targeted fix for 422 error
import { apiClient } from './api';

export const travelService = {
  async searchFlightsNaturalLanguage(query: string): Promise<any> {
    try {
      console.log('🔍 Calling API with query:', query);
      
      // Validate input
      if (!query || typeof query !== 'string' || query.trim().length === 0) {
        throw new Error('Query cannot be empty');
      }
      
      const payload = {
        query: query.trim(),
        user_id: 'user_' + Date.now(),
        timestamp: new Date().toISOString()
      };
      
      console.log('📤 Sending payload:', payload);
      
      // Method 1: Try with direct fetch first (bypass axios issues)
      try {
        const response = await fetch('http://localhost:8000/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            // Add these headers to help with CORS and API compatibility
            'Origin': window.location.origin,
            'User-Agent': 'TravelBot/1.0'
          },
          body: JSON.stringify(payload)
        });
        
        console.log('📡 Fetch Response Status:', response.status);
        console.log('📡 Fetch Response OK:', response.ok);
        
        const responseText = await response.text();
        console.log('📡 Raw Response Text:', responseText);
        
        let responseData;
        try {
          responseData = JSON.parse(responseText);
          console.log('📡 Parsed Response Data:', responseData);
        } catch (parseError) {
          console.error('❌ JSON Parse Error:', parseError);
          return {
            status: 'error',
            error: 'Invalid JSON response from server',
            raw_response: responseText
          };
        }
        
        // Handle the response regardless of HTTP status
        // Sometimes APIs return 422 but still have valid error info in the response
        if (responseData) {
          // If the API returned structured error info, use it
          if (responseData.status === 'error') {
            console.log('⚠️ API returned structured error:', responseData.error);
            
            // Check if it's a specific 422 validation error
            if (responseData.error && responseData.error.includes('422')) {
              return {
                status: 'error',
                error: 'Request format error. Let me try a different approach.',
                suggestions: [
                  'The query format might need adjustment',
                  'Try: "flights from Ahmedabad to Kochi"',
                  'Try: "AMD to COK flights"'
                ],
                debug_info: responseData
              };
            }
            
            return responseData; // Return the structured error as-is
          }
          
          // If successful, return the data
          if (responseData.status === 'success') {
            console.log('✅ Successful response received via fetch');
            return responseData;
          }
          
          // Handle any other response format
          return responseData;
        }
        
        // If we got here, something unexpected happened
        return {
          status: 'error',
          error: `HTTP ${response.status}: ${response.statusText}`,
          raw_response: responseText
        };
        
      } catch (fetchError: unknown) {
        console.error('❌ Fetch method failed:', fetchError);
        const fetchErrorMessage = fetchError instanceof Error ? fetchError.message : String(fetchError);
        
        // Method 2: Fallback to axios/apiClient
        console.log('🔄 Trying with axios as fallback...');
        
        try {
          const axiosResponse = await apiClient.post('/query', payload);
          console.log('✅ Axios Response:', axiosResponse);
          return axiosResponse;
        } catch (axiosError: unknown) {
          console.error('❌ Axios also failed:', axiosError);
          const axiosErrorMessage = axiosError instanceof Error ? axiosError.message : String(axiosError);
          
          // Method 3: Try with simplified payload
          console.log('🔄 Trying with simplified payload...');
          
          const simplifiedPayload = { query: query.trim() };
          
          try {
            const simpleResponse = await fetch('http://localhost:8000/query', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify(simplifiedPayload)
            });
            
            const simpleResponseText = await simpleResponse.text();
            console.log('📡 Simple Response:', simpleResponseText);
            
            try {
              const simpleResponseData = JSON.parse(simpleResponseText);
              return simpleResponseData;
            } catch (e) {
              return {
                status: 'error',
                error: 'All methods failed to get valid response',
                details: {
                  fetch_error: fetchErrorMessage,
                  axios_error: axiosErrorMessage,
                  simple_response: simpleResponseText
                }
              };
            }
            
          } catch (simpleError: unknown) {
            console.error('❌ All methods failed:', simpleError);
            const simpleErrorMessage = simpleError instanceof Error ? simpleError.message : String(simpleError);
            
            return {
              status: 'error',
              error: 'Cannot connect to API server',
              suggestions: [
                'Check if Python API server is running on port 8000',
                'Verify the server is accessible at http://localhost:8000',
                'Check server logs for error details'
              ],
              debug_info: {
                fetch_error: fetchErrorMessage,
                axios_error: axiosErrorMessage,
                simple_error: simpleErrorMessage
              }
            };
          }
        }
      }
      
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

  // Alternative method with different approach to headers and payload
  async searchFlightsAlternative(query: string): Promise<any> {
    try {
      console.log('🔄 Trying alternative search method for:', query);
      
      // Try different payload formats that might work better
      const payloadFormats = [
        // Format 1: Minimal
        { query: query },
        
        // Format 2: With user info
        { 
          query: query,
          user_id: `user_${Date.now()}`
        },
        
        // Format 3: With additional context
        {
          query: query,
          user_id: `user_${Date.now()}`,
          source: 'web_client',
          format: 'json'
        },
        
        // Format 4: Match your successful test format
        {
          query: query,
          user_id: 'user_1751012377175',
          timestamp: '2025-06-27T08:19:31.375Z'
        }
      ];
      
      // Try each format until one works
      for (let i = 0; i < payloadFormats.length; i++) {
        const payload = payloadFormats[i];
        console.log(`🧪 Trying payload format ${i + 1}:`, payload);
        
        try {
          const response = await fetch('http://localhost:8000/query', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
          });
          
          console.log(`📡 Format ${i + 1} Response Status:`, response.status);
          
          if (response.ok) {
            const data = await response.json();
            console.log(`✅ Format ${i + 1} succeeded:`, data);
            return data;
          } else {
            const errorText = await response.text();
            console.log(`❌ Format ${i + 1} failed:`, response.status, errorText);
            
            // If this is the last format and still 422, return detailed error
            if (i === payloadFormats.length - 1) {
              return {
                status: 'error',
                error: `All payload formats failed. Last error: HTTP ${response.status}`,
                last_error_details: errorText,
                tried_formats: payloadFormats
              };
            }
          }
        } catch (formatError: unknown) {
          console.log(`❌ Format ${i + 1} exception:`, formatError);
          const formatErrorMessage = formatError instanceof Error ? formatError.message : String(formatError);
          
          if (i === payloadFormats.length - 1) {
            throw new Error(formatErrorMessage);
          }
        }
      }
      
    } catch (error) {
      console.error('❌ Alternative method failed:', error);
      return {
        status: 'error',
        error: `Alternative method failed: ${error.message}`
      };
    }
  },

  // Debug method to test server responsiveness
  async debugServerConnection(): Promise<any> {
    try {
      console.log('🔬 Testing server connection...');
      
      // Test 1: Basic GET to root
      const rootTest = await fetch('http://localhost:8000/');
      console.log('📡 Root endpoint status:', rootTest.status);
      
      if (rootTest.ok) {
        const rootData = await rootTest.json();
        console.log('📡 Root endpoint data:', rootData);
      }
      
      // Test 2: OPTIONS request to check CORS
      const optionsTest = await fetch('http://localhost:8000/query', {
        method: 'OPTIONS'
      });
      console.log('📡 OPTIONS status:', optionsTest.status);
      console.log('📡 CORS headers:', Object.fromEntries(optionsTest.headers.entries()));
      
      return {
        root_test: {
          status: rootTest.status,
          ok: rootTest.ok
        },
        options_test: {
          status: optionsTest.status,
          headers: Object.fromEntries(optionsTest.headers.entries())
        }
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