// src/components/EnhancedFlightDisplay.tsx
import React, { useState, useMemo } from 'react';
import { 
  Plane, Clock, MapPin, Filter, ChevronDown, ChevronUp,
  Calendar, Euro, Zap, Target, ArrowRight, Info
} from 'lucide-react';

// Define the interfaces based on your API response structure
interface ConnectingFlight {
  segment: number;
  flight_number: string;
  departure: string;
  arrival: string;
  duration: string;
}

interface FlightInfo {
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
  booking_class: string;
  route: string;
  is_direct: boolean;
  aircraft?: string;
  operating_carrier: string;
  number_of_stops: number;
  connecting_flights?: ConnectingFlight[];
}

interface SearchInfo {
  search_date: string;
  return_date?: string;
  origin: string;
  destination: string;
  passengers: number;
  travel_class: string;
}

interface EnhancedFlightDisplayProps {
  flightData: FlightInfo[];
  searchInfo: SearchInfo;
}

const EnhancedFlightDisplay: React.FC<EnhancedFlightDisplayProps> = ({ flightData, searchInfo }) => {
  const [expandedFlights, setExpandedFlights] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<'price' | 'duration' | 'departure'>('price');
  const [showOnlyDirect, setShowOnlyDirect] = useState(false);

  // Airline mapping
  const airlineNames: Record<string, string> = {
    'AI': 'Air India', '6E': 'IndiGo', 'SG': 'SpiceJet', 
    'UK': 'Vistara', 'EK': 'Emirates', 'QR': 'Qatar Airways',
    'EY': 'Etihad Airways', 'WY': 'Oman Air', 'KU': 'Kuwait Airways',
    'LH': 'Lufthansa', 'BA': 'British Airways', 'AF': 'Air France',
    'KL': 'KLM', 'TK': 'Turkish Airlines', 'SU': 'Aeroflot'
  };

  // Process and sort flights
  const processedFlights = useMemo(() => {
    let flights = [...(flightData || [])];

    // Filter
    if (showOnlyDirect) {
      flights = flights.filter(f => f.is_direct);
    }

    // Sort
    flights.sort((a, b) => {
      switch (sortBy) {
        case 'price':
          return a.price_numeric - b.price_numeric;
        case 'duration':
          return parseDuration(a.duration) - parseDuration(b.duration);
        case 'departure':
          return a.departure_time.localeCompare(b.departure_time);
        default:
          return 0;
      }
    });

    return flights;
  }, [flightData, sortBy, showOnlyDirect]);

  const parseDuration = (duration: string): number => {
    const match = duration?.match(/PT(?:(\d+)H)?(?:(\d+)M)?/);
    if (!match) return 0;
    const hours = parseInt(match[1] || '0');
    const minutes = parseInt(match[2] || '0');
    return hours * 60 + minutes;
  };

  const formatDuration = (duration: string): string => {
    const totalMinutes = parseDuration(duration);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `${hours}h ${minutes}m`;
  };

  const toggleFlightExpansion = (flightId: string) => {
    const newExpanded = new Set(expandedFlights);
    if (newExpanded.has(flightId)) {
      newExpanded.delete(flightId);
    } else {
      newExpanded.add(flightId);
    }
    setExpandedFlights(newExpanded);
  };

  const formatConnectionTime = (departure: string, arrival: string): string => {
    const parseFlightTime = (timeStr: string) => {
      const airportMatch = timeStr.match(/^([A-Z]{3})/);
      const dateMatch = timeStr.match(/(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})/);
      
      return {
        airport: airportMatch?.[1] || '',
        date: dateMatch?.[1] ? new Date(dateMatch[1]) : null
      };
    };
    
    const dep = parseFlightTime(departure);
    const arr = parseFlightTime(arrival);
    
    if (!dep.date || !arr.date) {
      return 'Invalid time format';
    }
    
    const depTime = dep.date.toLocaleTimeString('en-US', { 
      hour: '2-digit', minute: '2-digit', hour12: false 
    });
    const arrTime = arr.date.toLocaleTimeString('en-US', { 
      hour: '2-digit', minute: '2-digit', hour12: false 
    });
    
    return `${dep.airport} ${depTime} → ${arr.airport} ${arrTime}`;
  };

  const cheapest = processedFlights[0];
  const fastest = processedFlights.reduce((fastest, current) => 
    parseDuration(current.duration) < parseDuration(fastest.duration) ? current : fastest
  );
  const formatCurrency = (amount: number): string => new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 0
  }).format(amount);

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <Plane className="text-blue-500" />
            {processedFlights.length} Flights Found
          </h1>
          <div className="text-sm text-gray-600 flex items-center gap-2">
            <Calendar size={16} />
            {searchInfo?.search_date}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <div className="flex items-center gap-2 text-green-700">
              <Euro size={16} />
              <span className="font-medium">Cheapest</span>
            </div>
            <div className="text-lg font-bold text-green-800">
              {formatCurrency(cheapest.price_numeric)}
            </div>
            <div className="text-sm text-green-600">
              {airlineNames[cheapest.airline] || cheapest.airline} {cheapest.flight_number}
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <div className="flex items-center gap-2 text-blue-700">
              <Zap size={16} />
              <span className="font-medium">Fastest</span>
            </div>
            <div className="text-lg font-bold text-blue-800">
              {formatDuration(fastest.duration)}
            </div>
            <div className="text-sm text-blue-600">
              {airlineNames[fastest.airline] || fastest.airline} {fastest.flight_number}
            </div>
          </div>

          <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
            <div className="flex items-center gap-2 text-orange-700">
              <Target size={16} />
              <span className="font-medium">Route</span>
            </div>
            <div className="text-lg font-bold text-orange-800">
              {searchInfo?.origin} → {searchInfo?.destination}
            </div>
            <div className="text-sm text-orange-600">
              {processedFlights.filter(f => f.is_direct).length} direct flights
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-gray-500" />
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value as 'price' | 'duration' | 'departure')}
              className="border rounded px-3 py-1 text-sm"
            >
              <option value="price">Sort by Price</option>
              <option value="duration">Sort by Duration</option>
              <option value="departure">Sort by Departure</option>
            </select>
          </div>

          <label className="flex items-center gap-2 text-sm">
            <input 
              type="checkbox" 
              checked={showOnlyDirect}
              onChange={(e) => setShowOnlyDirect(e.target.checked)}
              className="rounded"
            />
            Direct flights only
          </label>
        </div>
      </div>

      {/* Flight List */}
      <div className="space-y-4">
        {processedFlights.map((flight, index) => {
          const flightKey = `${flight.flight_number}-${index}`;
          const isExpanded = expandedFlights.has(flightKey);
          const airlineName = airlineNames[flight.airline] || flight.airline;
          
          return (
            <div key={flightKey} className="bg-white rounded-lg shadow-lg border border-gray-200">
              {/* Flight Header */}
              <div 
                className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleFlightExpansion(flightKey)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="text-2xl font-bold text-blue-600">
                      {flight.price}
                    </div>
                    <div>
                      <div className="font-semibold text-gray-800">
                        {flight.flight_number} • {airlineName}
                      </div>
                      <div className="text-sm text-gray-600 flex items-center gap-4">
                        <span className="flex items-center gap-1">
                          <Clock size={14} />
                          {formatDuration(flight.duration)}
                        </span>
                        <span className="flex items-center gap-1">
                          <MapPin size={14} />
                          {flight.is_direct ? 'Direct' : `${flight.stops} stop${flight.stops > 1 ? 's' : ''}`}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-semibold text-gray-800">
                      {flight.departure_time} → {flight.arrival_time}
                    </div>
                    <div className="text-sm text-gray-600">
                      {flight.departure_airport} → {flight.arrival_airport}
                    </div>
                  </div>

                  <div className="ml-4">
                    {isExpanded ? <ChevronUp /> : <ChevronDown />}
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="border-t border-gray-200 p-4 bg-gray-50">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Flight Details */}
                    <div>
                      <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Info size={16} />
                        Flight Details
                      </h4>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Aircraft:</span>
                          <span className="font-medium">{flight.aircraft || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Class:</span>
                          <span className="font-medium">{flight.booking_class}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Departure Terminal:</span>
                          <span className="font-medium">{flight.departure_terminal || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Arrival Terminal:</span>
                          <span className="font-medium">{flight.arrival_terminal || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Departure Date:</span>
                          <span className="font-medium">{flight.departure_date || 'N/A'}</span>
                        </div>
                       
                      </div>
                    </div>

                    {/* Connection Details */}
                    {flight.connecting_flights && flight.connecting_flights.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                          <ArrowRight size={16} />
                          Connection Details
                        </h4>
                        <div className="space-y-3">
                          {flight.connecting_flights.map((connection, connIndex) => (
                            <div key={connIndex} className="bg-white p-3 rounded border">
                              <div className="font-medium text-sm text-gray-800 mb-1">
                                Segment {connection.segment}: {connection.flight_number}
                              </div>
                              <div className="text-sm text-gray-600">
                                {formatConnectionTime(connection.departure, connection.arrival)}
                              </div>
                              <div className="text-xs text-gray-500">
                                Duration: {formatDuration(connection.duration)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Book Button */}
                  <div className="mt-6 pt-4 border-t border-gray-200">
                    <button className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors font-semibold">
                      Select This Flight - {flight.price}
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary Footer */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center text-gray-600">
          {/* ✅ Construct message from available data */}
          Showing {processedFlights.length} cheapest flights
          {/* Check if it's a month search by looking at flight dates */}
        {processedFlights.length > 0 && new Set(processedFlights.map(f => f.departure_date)).size > 1 
          ? ' across multiple dates' 
          : ''
        } •           
        Prices from {formatCurrency(Math.min(...processedFlights.map(f => f.price_numeric)))} to {formatCurrency(Math.max(...processedFlights.map(f => f.price_numeric)))}
          </div>
      </div>
    </div>
  );
};

export default EnhancedFlightDisplay;