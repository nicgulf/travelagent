// ==========================================
// types.ts - All shared interfaces and types
// ==========================================

// Flight-related interfaces
export interface FlightSearchParams {
    origin: string;
    destination: string;
    departure_date?: string;
    return_date?: string;
    passengers?: number;
    travel_class?: 'ECONOMY' | 'BUSINESS' | 'FIRST';
  }
  
export interface FlightInfo {
    airline: string;
    flight_number: string;
    departure_date: string;
    departure_time: string;
    arrival_date: string;
    arrival_time: string;
    departure_airport: string;
    arrival_airport: string;
    departure_terminal?: string;
    arrival_terminal?: string;
    duration: string;
    price: string;
    price_numeric: number;
    currency: string;
    stops: number;
    is_direct: boolean;
    booking_class: string;
    route: string;
    aircraft?: string;
    operating_carrier?: string;
    connecting_flights?: ConnectingFlight[];
  }
  
export interface ConnectingFlight {
  segment: number;
  flight_number: string;
  departure: string;
  arrival: string;
  duration: string;
}

export interface FlightSearchResponse {
  search_info: {
    search_date: string;
    return_date?: string;
    origin: string;
    destination: string;
    passengers: number;
    travel_class: string;
  };
  flights: FlightInfo[];
  total_found: number;
  success: boolean;
  message?: string;
  error?: string;
}

// Natural Language Processing interfaces
export interface NaturalLanguageQuery {
  query: string;
  user_id?: string;
}

export interface QueryResponse {
  status: string;
  tool_used: string;
  data: any;
  message?: string;
}

// Airport and Airline interfaces
export interface AirportInfo {
  code: string;
  name: string;
  city: string;
  country: string;
  timezone: string;
}

export interface AirlineInfo {
  code: string;
  name: string;
  country: string;
}

// Chat-related interfaces
export interface ChatResponse {
  message: string;
  suggestions?: string[];
  data?: any;
  type?: 'text' | 'flight_results' | 'airport_info' | 'airline_info' | 'error';
  flights?: FlightInfo[];
}

export interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  data?: any;
  suggestions?: string[];
  flights?: FlightInfo[];
}

// API Response interfaces
export interface APIResponse<T = any> {
  status: 'success' | 'error';
  data?: T;
  message?: string;
  error?: string;
}

// Utility interfaces
export interface DateParseResponse {
  input: string;
  parsed: string;
  formatted: string;
}

export interface ToolInfo {
  name: string;
  description: string;
  parameters: string[];
}

export interface ToolsResponse {
  tools: ToolInfo[];
}

// Travel class enum for better type safety
export enum TravelClass {
  ECONOMY = 'ECONOMY',
  BUSINESS = 'BUSINESS',
  FIRST = 'FIRST'
}

// Message types enum
export enum MessageType {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system'
}

// Response types enum
export enum ResponseType {
  TEXT = 'text',
  FLIGHT_RESULTS = 'flight_results',
  AIRPORT_INFO = 'airport_info',
  AIRLINE_INFO = 'airline_info',
  ERROR = 'error'
}
  