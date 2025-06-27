#!/usr/bin/env python3

import asyncio
import aiohttp
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from amadeus import Client
from dotenv import load_dotenv
import os
import re

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any, Tuple

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.types as types

load_dotenv()

class SmartLocationResolver:
    """Smart location resolver using Amadeus API - NO MANUAL MAPPING"""
    
    def __init__(self, amadeus_client):
        self.amadeus = amadeus_client
        self._location_cache = {}  # Cache for performance
    
    async def resolve_location_to_airport(self, location_input: str) -> str:
        """
        Resolve any location (city name, airport name, etc.) to airport code using Amadeus API
        """
        if not location_input:
            raise ValueError("Location input is required")
        
        location_clean = location_input.strip()
        location_key = location_clean.lower()
        
        # Check if it's already a 3-letter airport code
        if len(location_clean) == 3 and location_clean.isalpha():
            return location_clean.upper()
        
        # Check cache first
        if location_key in self._location_cache:
            print(f"🎯 Using cached result for '{location_clean}': {self._location_cache[location_key]}")
            return self._location_cache[location_key]
        
        # Use Amadeus API to search for location
        try:
            print(f"🔍 Searching for location: '{location_clean}'")
            
            response = self.amadeus.reference_data.locations.get(
                keyword=location_clean,
                subType='AIRPORT,CITY'
            )
            
            if response.status_code == 200 and response.data:
                # Find the best match
                for location in response.data:
                    if hasattr(location, 'iataCode') and location.iataCode:
                        airport_code = location.iataCode
                        location_name = getattr(location, 'name', 'Unknown')
                        
                        print(f"✅ Resolved '{location_clean}' → '{airport_code}' ({location_name})")
                        
                        # Cache the result
                        self._location_cache[location_key] = airport_code
                        return airport_code
                
                # If no iataCode found, try to get from subType CITY
                print(f"🔄 No direct airport found, searching for city airports...")
                city_response = self.amadeus.reference_data.locations.get(
                    keyword=location_clean,
                    subType='CITY'
                )
                
                if city_response.status_code == 200 and city_response.data:
                    for city in city_response.data:
                        if hasattr(city, 'relationships') and city.relationships:
                            # City might have associated airports
                            city_name = getattr(city, 'name', location_clean)
                            print(f"🏙️ Found city: {city_name}, searching for airports...")
                            
                            # Search for airports in this city
                            airport_search = self.amadeus.reference_data.locations.get(
                                keyword=city_name + " airport",
                                subType='AIRPORT'
                            )
                            
                            if airport_search.status_code == 200 and airport_search.data:
                                for airport in airport_search.data:
                                    if hasattr(airport, 'iataCode') and airport.iataCode:
                                        airport_code = airport.iataCode
                                        airport_name = getattr(airport, 'name', 'Unknown')
                                        
                                        print(f"✅ Found airport '{airport_code}' ({airport_name}) for city '{location_clean}'")
                                        
                                        # Cache the result
                                        self._location_cache[location_key] = airport_code
                                        return airport_code
        
        except Exception as e:
            print(f"❌ Amadeus API error for '{location_clean}': {e}")
        
        # If API fails, try fuzzy matching with common patterns
        fuzzy_result = self._fuzzy_location_match(location_clean)
        if fuzzy_result:
            self._location_cache[location_key] = fuzzy_result
            return fuzzy_result
        
        raise ValueError(f"Could not resolve location: '{location_clean}'. Please try with a more specific city name or airport code.")
    
    def _fuzzy_location_match(self, location: str) -> Optional[str]:
        """
        Fallback fuzzy matching for common cases
        Only includes a minimal set of most common locations
        """
        location_lower = location.lower()
        
        # Only the most essential mappings as fallback
        essential_fallbacks = {
            # Major Indian cities (most commonly searched)
            'mumbai': 'BOM', 'delhi': 'DEL', 'bangalore': 'BLR', 
            'chennai': 'MAA', 'kolkata': 'CCU', 'hyderabad': 'HYD',
            'ahmedabad': 'AMD', 'kochi': 'COK', 'pune': 'PNQ',
            
            # Major international hubs (most commonly searched)
            'london': 'LHR', 'paris': 'CDG', 'new york': 'JFK',
            'dubai': 'DXB', 'singapore': 'SIN', 'tokyo': 'NRT'
        }
        
        if location_lower in essential_fallbacks:
            print(f"🎯 Fuzzy match: '{location}' → '{essential_fallbacks[location_lower]}'")
            return essential_fallbacks[location_lower]
        
        return None

class EnhancedQueryProcessor:
    """Enhanced query processor with dynamic location resolution"""
    
    def __init__(self, flight_service, location_resolver):
        self.flight_service = flight_service
        self.location_resolver = location_resolver
        self.tools = {
            "search_flights": flight_service.search_flights_tool,
            "get_airport_info": flight_service.get_airport_info_tool,
            "get_airline_info": flight_service.get_airline_info_tool
        }
    
    async def extract_locations_smart(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract and resolve locations from natural language using Amadeus API"""
        query_lower = query.lower()
        
        # Pattern 1: "from X to Y"
        from_to_patterns = [
            r'from\s+([^to]+?)\s+to\s+(.+?)(?:\s+on|\s+tomorrow|\s+next|\s+\d|$)',
            r'from\s+([^to]+?)\s+to\s+(.+)',
        ]
        
        for pattern in from_to_patterns:
            match = re.search(pattern, query_lower)
            if match:
                origin_text = match.group(1).strip()
                dest_text = match.group(2).strip()
                
                try:
                    print(f"🔍 Extracting: '{origin_text}' → '{dest_text}'")
                    
                    # Resolve both locations using Amadeus API
                    origin_code = await self.location_resolver.resolve_location_to_airport(origin_text)
                    dest_code = await self.location_resolver.resolve_location_to_airport(dest_text)
                    
                    print(f"✅ Resolved: {origin_text} → {origin_code}, {dest_text} → {dest_code}")
                    return origin_code, dest_code
                    
                except ValueError as e:
                    print(f"❌ Location resolution failed: {e}")
                    continue
        
        # Pattern 2: "X to Y" (without "from")
        to_patterns = [
            r'([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s+on|\s+tomorrow|\s+next|\s+\d|$)',
            r'([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)',
        ]
        
        for pattern in to_patterns:
            match = re.search(pattern, query_lower)
            if match:
                origin_text = match.group(1).strip()
                dest_text = match.group(2).strip()
                
                # Skip if they're too short or common words
                if len(origin_text) < 3 or len(dest_text) < 3:
                    continue
                if origin_text in ['flight', 'flights', 'book', 'search', 'find']:
                    continue
                    
                try:
                    print(f"🔍 Extracting (pattern 2): '{origin_text}' → '{dest_text}'")
                    
                    # Resolve both locations using Amadeus API
                    origin_code = await self.location_resolver.resolve_location_to_airport(origin_text)
                    dest_code = await self.location_resolver.resolve_location_to_airport(dest_text)
                    
                    print(f"✅ Resolved: {origin_text} → {origin_code}, {dest_text} → {dest_code}")
                    return origin_code, dest_code
                    
                except ValueError as e:
                    print(f"❌ Location resolution failed: {e}")
                    continue
        
        # Pattern 3: Direct airport codes in query
        airport_codes = re.findall(r'\b([A-Z]{3})\b', query.upper())
        if len(airport_codes) >= 2:
            print(f"✅ Found airport codes directly: {airport_codes[0]} → {airport_codes[1]}")
            return airport_codes[0], airport_codes[1]
        
        return None, None
    
    def extract_date_from_query(self, query: str) -> str:
        """Extract departure date from natural language"""
        date_patterns = [
            r'tomorrow',
            r'today',
            r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(0)
        
        return None
    
    async def determine_tool_and_params(self, query: str) -> Tuple[Optional[str], dict]:
        """Analyze query and determine which tool to use with dynamic location resolution"""
        query_lower = query.lower()
        
        # Flight search patterns
        if any(keyword in query_lower for keyword in ['flight', 'fly', 'book', 'search', 'from', 'to']):
            try:
                origin, destination = await self.extract_locations_smart(query)
                departure_date = self.extract_date_from_query(query)
                
                if not origin or not destination:
                    return None, {}
                
                params = {
                    'origin': origin,
                    'destination': destination
                }
                
                if departure_date:
                    params['departure_date'] = departure_date
                
                # Extract passengers
                passenger_match = re.search(r'(\d+)\s+(passenger|person|people)', query_lower)
                if passenger_match:
                    params['passengers'] = int(passenger_match.group(1))
                
                # Extract class
                if 'business' in query_lower:
                    params['travel_class'] = 'BUSINESS'
                elif 'first' in query_lower:
                    params['travel_class'] = 'FIRST'
                
                return 'search_flights', params
                
            except Exception as e:
                print(f"❌ Error processing flight query: {e}")
                return None, {}
        
        # Airport info patterns
        elif any(keyword in query_lower for keyword in ['airport', 'what is', 'info about']):
            # Try to extract location and resolve to airport code
            words = query.split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    try:
                        airport_code = await self.location_resolver.resolve_location_to_airport(word)
                        return 'get_airport_info', {'airport_code': airport_code}
                    except:
                        continue
        
        # Airline info patterns  
        elif any(keyword in query_lower for keyword in ['airline', 'carrier']):
            codes = re.findall(r'\b([A-Z]{2})\b', query.upper())
            if codes:
                return 'get_airline_info', {'airline_code': codes[0]}
        
        return None, {}

class FlightSearchMCPServer:
    def __init__(self):
        """Initialize the MCP server for flight searches"""
        self.amadeus_api_key = os.getenv("AMADEUS_API_KEY")
        self.amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")
        self.access_token = None
        
        # Initialize Amadeus client
        self.amadeus = Client(
            client_id=self.amadeus_api_key,
            client_secret=self.amadeus_api_secret
        )
        # Initialize smart location resolver
        self.location_resolver = SmartLocationResolver(self.amadeus)


    def get_smart_default_date(self) -> str:
        """Get a smart default date (next week)"""
        default_date = datetime.now() + timedelta(days=7)
        return default_date.strftime("%Y-%m-%d")
    
    def parse_flexible_date(self, date_input: str) -> str:
        """Parse various date formats and relative dates"""
        if not date_input:
            return self.get_smart_default_date()
        
        date_input = date_input.lower().strip()
        today = datetime.now()
        
        # Handle relative dates
        relative_dates = {
            'today': today,
            'tomorrow': today + timedelta(days=1),
            'day after tomorrow': today + timedelta(days=2),
            'next week': today + timedelta(days=7),
            'next month': today + timedelta(days=30),
        }
        
        for phrase, date_obj in relative_dates.items():
            if phrase in date_input:
                return date_obj.strftime("%Y-%m-%d")
        
        # Handle "next [day]" patterns
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in weekdays.items():
            if f'next {day_name}' in date_input:
                days_ahead = day_num - today.weekday()
                if days_ahead <= 0:  # Target day has passed this week
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d")
        
        # Handle month names (e.g., "August", "December 15")
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in date_input:
                current_year = today.year
                # If month has passed this year, assume next year
                if month_num < today.month:
                    current_year += 1
                
                # Extract day if provided (e.g., "December 15")
                day_match = re.search(r'\b(\d{1,2})\b', date_input)
                day = int(day_match.group(1)) if day_match else 15  # Default to mid-month
                
                try:
                    target_date = datetime(current_year, month_num, day)
                    return target_date.strftime("%Y-%m-%d")
                except ValueError:
                    # Invalid day for month, use last valid day
                    target_date = datetime(current_year, month_num, 28)
                    return target_date.strftime("%Y-%m-%d")
        
        # Try to parse as standard date formats
        date_formats = [
            "%Y-%m-%d",      # 2024-08-15
            "%d-%m-%Y",      # 15-08-2024
            "%m/%d/%Y",      # 08/15/2024
            "%d/%m/%Y",      # 15/08/2024
            "%B %d, %Y",     # August 15, 2024
            "%d %B %Y",      # 15 August 2024
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_input, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If all else fails, return default date
        print(f"Warning: Could not parse date '{date_input}', using default")
        return self.get_smart_default_date()

    async def get_access_token(self) -> str:
        """Get Amadeus access token"""
        if self.access_token:
            return self.access_token
            
        auth_endpoint = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.amadeus_api_key,
            "client_secret": self.amadeus_api_secret
        }
        
        try:
            response = requests.post(auth_endpoint, headers=headers, data=data)
            response.raise_for_status()
            self.access_token = response.json()['access_token']
            return self.access_token
        except Exception as e:
            raise Exception(f"Authentication error: {e}")

    async def search_flights_tool(self, arguments: dict) -> Dict:
        """Enhanced flight search with dynamic location resolution"""
        print('🔍 Searching flight details with dynamic location resolution...')
        try:
            origin = arguments.get("origin")
            destination = arguments.get("destination") 
            departure_date_input = arguments.get("departure_date")
            passengers = arguments.get("passengers", 1)
            travel_class = arguments.get("travel_class", "ECONOMY")
            return_date_input = arguments.get("return_date")
            
            # Validate required parameters
            if not all([origin, destination]):
                raise ValueError("Missing required parameters: origin, destination")
            
            print(f"✅ Using resolved airports: {origin} → {destination}")
            
            # Parse dates
            departure_date = self.parse_flexible_date(departure_date_input)
            return_date = self.parse_flexible_date(return_date_input) if return_date_input else None
            
            print(f"🛫 Searching flights from {origin} to {destination} on {departure_date}")
            
            # Execute Amadeus flight search
            search_params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination, 
                "departureDate": departure_date,
                "adults": passengers,
                "max": 10
            }
            
            if return_date:
                search_params["returnDate"] = return_date
            if travel_class and travel_class.upper() != "ECONOMY":
                search_params["travelClass"] = travel_class.upper()
            
            response = self.amadeus.shopping.flight_offers_search.get(**search_params)
            
            if response.status_code == 200:
                flights = self.parse_flight_results(response.data, origin, destination)
                return {
                    "search_info": {
                        "search_date": departure_date,
                        "return_date": return_date,
                        "origin": origin,
                        "destination": destination,
                        "passengers": passengers,
                        "travel_class": travel_class
                    },
                    "flights": flights,
                    "total_found": len(flights),
                    "message": f"Found {len(flights)} flights from {origin} to {destination}"
                }
            else:
                raise Exception(f"Flight search failed with status {response.status_code}")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Flight search error: {error_msg}")
            
            return {
                "error": f"Flight search failed: {error_msg}",
                "search_info": {
                    "origin": arguments.get("origin"),
                    "destination": arguments.get("destination"),
                    "search_date": departure_date if 'departure_date' in locals() else None
                },
                "flights": [],
                "total_found": 0,
                "suggestions": [
                    "Try with specific city names",
                    "Check spelling of locations",
                    "Use airport codes if known"
                ]
            }
    
    def parse_flight_results(self, flight_data, origin, destination) -> List[Dict]:
        """Parse Amadeus API response with enhanced information"""
        flights = []
        print(flight_data, 'flight data')
        
        for offer in flight_data[:5]:  # Up to 5 results
            try:
                itinerary = offer['itineraries'][0]  # Get first itinerary
                segment = itinerary['segments'][0]   # Get first segment
                
                # Parse departure and arrival times
                departure_dt = datetime.fromisoformat(segment['departure']['at'].replace('Z', '+00:00') if segment['departure']['at'].endswith('Z') else segment['departure']['at'])
                arrival_dt = datetime.fromisoformat(segment['arrival']['at'].replace('Z', '+00:00') if segment['arrival']['at'].endswith('Z') else segment['arrival']['at'])
                
                # Calculate if arrival is next day
                arrival_day_diff = (arrival_dt.date() - departure_dt.date()).days
                arrival_display = arrival_dt.strftime("%H:%M")
                if arrival_day_diff > 0:
                    arrival_display += f"+{arrival_day_diff}"
                
                # Get cabin class from traveler pricing if available
                cabin_class = 'ECONOMY'  # default
                if 'travelerPricings' in offer and len(offer['travelerPricings']) > 0:
                    fare_details = offer['travelerPricings'][0].get('fareDetailsBySegment', [])
                    if fare_details and 'cabin' in fare_details[0]:
                        cabin_class = fare_details[0]['cabin']
                
                flight = {
                    "airline": segment['carrierCode'],
                    "flight_number": f"{segment['carrierCode']}{segment['number']}",
                    "departure_date": departure_dt.strftime("%Y-%m-%d"),
                    "departure_time": departure_dt.strftime("%H:%M"),
                    "arrival_date": arrival_dt.strftime("%Y-%m-%d"),
                    "arrival_time": arrival_display,
                    "departure_airport": segment['departure']['iataCode'],
                    "arrival_airport": segment['arrival']['iataCode'],
                    "departure_terminal": segment['departure'].get('terminal', ''),
                    "arrival_terminal": segment['arrival'].get('terminal', ''),
                    "duration": itinerary['duration'],
                    "price": f"{offer['price']['total']} {offer['price']['currency']}",
                    "price_numeric": float(offer['price']['total']),
                    "currency": offer['price']['currency'],
                    "stops": len(itinerary['segments']) - 1,
                    "booking_class": cabin_class,
                    "route": f"{origin}->{destination}",
                    "is_direct": len(itinerary['segments']) == 1,
                    "aircraft": segment.get('aircraft', {}).get('code', ''),
                    "operating_carrier": segment.get('operating', {}).get('carrierCode', segment['carrierCode']),
                    "number_of_stops": segment.get('numberOfStops', 0)
                }
                
                # Add connecting flights information if not direct
                if len(itinerary['segments']) > 1:
                    flight['connecting_flights'] = []
                    for i, seg in enumerate(itinerary['segments']):
                        flight['connecting_flights'].append({
                            "segment": i + 1,
                            "flight_number": f"{seg['carrierCode']}{seg['number']}",
                            "departure": f"{seg['departure']['iataCode']} {seg['departure']['at']}",
                            "arrival": f"{seg['arrival']['iataCode']} {seg['arrival']['at']}",
                            "duration": seg['duration']
                        })
                
                flights.append(flight)
                
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                print(f"Error parsing flight data: {e}")
                print(f"Problematic offer: {offer}")
                continue
        
        return flights

    async def get_airport_info_tool(self, arguments: dict) -> Dict:
        """MCP Tool: Get airport information"""
        try:
            airport_code = arguments.get("airport_code")
            if not airport_code:
                raise ValueError("airport_code is required")
                
            response = self.amadeus.reference_data.locations.airports.get(
                keyword=airport_code
            )
            
            if response.status_code == 200 and response.data:
                airport = response.data[0]
                return {
                    "code": airport.iataCode,
                    "name": airport.name,
                    "city": airport.address.cityName,
                    "country": airport.address.countryName,
                    "timezone": getattr(airport, 'timeZoneOffset', 'Unknown')
                }
            else:
                return {"error": "Airport not found"}
                
        except Exception as e:
            return {"error": f"Airport lookup failed: {str(e)}"}

    async def get_airline_info_tool(self, arguments: dict) -> Dict:
        """MCP Tool: Get airline information"""
        try:
            airline_code = arguments.get("airline_code")
            if not airline_code:
                raise ValueError("airline_code is required")
                
            response = self.amadeus.reference_data.airlines.get(
                airlineCodes=airline_code
            )
            
            if response.status_code == 200 and response.data:
                airline = response.data[0]
                return {
                    "code": airline.iataCode,
                    "name": airline.businessName,
                    "country": getattr(airline, 'countryCode', 'Unknown')
                }
            else:
                return {"error": "Airline not found"}
                
        except Exception as e:
            return {"error": f"Airline lookup failed: {str(e)}"}


# Query Processing and Tool Selection
class QueryProcessor:
    def __init__(self, flight_service: FlightSearchMCPServer):
        self.flight_service = flight_service
        self.tools = {
            "search_flights": flight_service.search_flights_tool,
            "get_airport_info": flight_service.get_airport_info_tool,
            "get_airline_info": flight_service.get_airline_info_tool
        }
    
    def extract_airport_codes(self, query: str) -> tuple[str, str]:
        """Extract origin and destination from natural language"""
        # Common airport code patterns
        airport_patterns = [
            r'\b([A-Z]{3})\b',  # 3-letter codes like JFK, LAX
            r'\b(AMD|COK|BLR|DEL|BOM|CCU|MAA|HYD)\b'  # Indian airports
        ]
        
        # City to airport code mapping
        city_to_airport = {
            'ahmedabad': 'AMD', 'kochi': 'COK', 'cochin': 'COK',
            'bangalore': 'BLR', 'bengaluru': 'BLR', 'delhi': 'DEL',
            'mumbai': 'BOM', 'kolkata': 'CCU', 'chennai': 'MAA',
            'hyderabad': 'HYD', 'pune': 'PNQ', 'goa': 'GOI',
            'new york': 'JFK', 'london': 'LHR', 'paris': 'CDG',
            'tokyo': 'NRT', 'dubai': 'DXB', 'singapore': 'SIN'
        }
        
        query_lower = query.lower()
        found_airports = []
        
        # First try to find airport codes
        for pattern in airport_patterns:
            matches = re.findall(pattern, query.upper())
            found_airports.extend(matches)
        
        # Then try city names
        for city, code in city_to_airport.items():
            if city in query_lower:
                found_airports.append(code)
        
        # Extract from common patterns like "from X to Y"
        from_to_pattern = r'from\s+(\w+)\s+to\s+(\w+)'
        match = re.search(from_to_pattern, query_lower)
        if match:
            origin_city = match.group(1)
            dest_city = match.group(2)
            origin = city_to_airport.get(origin_city, origin_city.upper())
            dest = city_to_airport.get(dest_city, dest_city.upper())
            return origin, dest
        
        # Return first two found airports
        if len(found_airports) >= 2:
            return found_airports[0], found_airports[1]
        elif len(found_airports) == 1:
            return found_airports[0], None
        
        return None, None
    
    def extract_date_from_query(self, query: str) -> str:
        """Extract departure date from natural language"""
        # Look for date patterns in the query
        date_patterns = [
            r'tomorrow',
            r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(0)
        
        return None
    
    def determine_tool_and_params(self, query: str) -> tuple[str, dict]:
        """Analyze query and determine which tool to use with parameters"""
        query_lower = query.lower()
        
        # Flight search patterns
        if any(keyword in query_lower for keyword in ['flight', 'fly', 'book', 'search', 'from', 'to']):
            origin, destination = self.extract_airport_codes(query)
            departure_date = self.extract_date_from_query(query)
            
            params = {}
            if origin:
                params['origin'] = origin
            if destination:
                params['destination'] = destination
            if departure_date:
                params['departure_date'] = departure_date
            
            # Extract passengers
            passenger_match = re.search(r'(\d+)\s+(passenger|person|people)', query_lower)
            if passenger_match:
                params['passengers'] = int(passenger_match.group(1))
            
            # Extract class
            if 'business' in query_lower:
                params['travel_class'] = 'BUSINESS'
            elif 'first' in query_lower:
                params['travel_class'] = 'FIRST'
            
            return 'search_flights', params
        
        # Airport info patterns
        elif any(keyword in query_lower for keyword in ['airport', 'what is', 'info about']):
            # Try to extract airport code
            codes = re.findall(r'\b([A-Z]{3})\b', query.upper())
            if codes:
                return 'get_airport_info', {'airport_code': codes[0]}
        
        # Airline info patterns
        elif any(keyword in query_lower for keyword in ['airline', 'carrier']):
            codes = re.findall(r'\b([A-Z]{2})\b', query.upper())
            if codes:
                return 'get_airline_info', {'airline_code': codes[0]}
        
        # Default to flight search if we can extract airports
        origin, destination = self.extract_airport_codes(query)
        if origin and destination:
            return 'search_flights', {'origin': origin, 'destination': destination}
        
        return None, {}


# FastAPI Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict

class QueryResponse(BaseModel):
    status: str
    tool_used: str
    data: Any
    message: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(title="Flight Search MCP API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
flight_service = FlightSearchMCPServer()
query_processor = QueryProcessor(flight_service)


# FastAPI Routes
@app.get("/")
async def root():
    return {"message": "Flight Search MCP API", "version": "1.0.0"}

@app.get("/tools")
async def list_available_tools():
    """List all available MCP tools"""
    return {
        "tools": [
            {
                "name": "search_flights",
                "description": "Search for flights between airports with flexible date support",
                "parameters": ["origin", "destination", "departure_date", "return_date", "passengers", "travel_class"]
            },
            {
                "name": "get_airport_info", 
                "description": "Get information about an airport",
                "parameters": ["airport_code"]
            },
            {
                "name": "get_airline_info",
                "description": "Get information about an airline", 
                "parameters": ["airline_code"]
            }
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def process_natural_query(request: QueryRequest):
    """Process natural language query and execute appropriate tool"""
    try:
        # Determine which tool to use
        tool_name, params = query_processor.determine_tool_and_params(request.query)
        
        if not tool_name:
            raise HTTPException(
                status_code=400, 
                detail="Could not understand the query. Please specify origin and destination airports."
            )
        
        # Execute the tool
        if tool_name in query_processor.tools:
            result = await query_processor.tools[tool_name](params)
            
            return QueryResponse(
                status="success",
                tool_used=tool_name,
                data=result,
                message=f"Processed query using {tool_name}"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Tool {tool_name} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tool", response_model=QueryResponse)
async def execute_tool_directly(request: ToolRequest):
    """Execute a specific tool with given parameters"""
    try:
        if request.tool_name not in query_processor.tools:
            raise HTTPException(status_code=400, detail=f"Tool {request.tool_name} not found")
        
        result = await query_processor.tools[request.tool_name](request.arguments)
        
        return QueryResponse(
            status="success",
            tool_used=request.tool_name,
            data=result,
            message=f"Executed {request.tool_name} successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/flights/search")
async def search_flights_endpoint(
    origin: str,
    destination: str,
    departure_date: Optional[str] = None,
    return_date: Optional[str] = None,
    passengers: int = 1,
    travel_class: str = "ECONOMY"
):
    """Direct flight search endpoint"""
    try:
        params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "passengers": passengers,
            "travel_class": travel_class
        }
        
        result = await flight_service.search_flights_tool(params)
        return {"status": "success", "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/airport/{airport_code}")
async def get_airport_info_endpoint(airport_code: str):
    """Get airport information"""
    try:
        result = await flight_service.get_airport_info_tool({"airport_code": airport_code})
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/airline/{airline_code}")
async def get_airline_info_endpoint(airline_code: str):
    """Get airline information"""
    try:
        result = await flight_service.get_airline_info_tool({"airline_code": airline_code})
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)