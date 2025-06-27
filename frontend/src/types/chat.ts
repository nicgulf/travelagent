export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  timestamp: Date;
  // Add these new properties:
  data?: {
    flights?: FlightInfo[];
    type?: 'flight_results' | 'airport_info' | 'airline_info' | 'error';
    search_info?: {
      origin: string;
      destination: string;
      search_date: string;
      return_date?: string;
      passengers: number;
      travel_class: string;
    };
    total_found?: number;
    [key: string]: any; // Allow any additional properties
  };
  suggestions?: string[];
  flights?: FlightInfo[]; // Alternative: direct flights property
}

// Add FlightInfo interface if it doesn't exist
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
  route?: string;
  aircraft?: string;
  operating_carrier?: string;
}