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
        """
        MCP Tool: Search for flights with smart date handling
        """
        try:
            # Extract parameters from arguments
            origin = arguments.get("origin")
            destination = arguments.get("destination") 
            departure_date_input = arguments.get("departure_date")
            passengers = arguments.get("passengers", 1)
            travel_class = arguments.get("travel_class", "ECONOMY")
            return_date_input = arguments.get("return_date")
            
            # Validate required parameters
            if not all([origin, destination]):
                raise ValueError("Missing required parameters: origin, destination")
            
            # Parse dates with smart handling
            departure_date = self.parse_flexible_date(departure_date_input)
            return_date = self.parse_flexible_date(return_date_input) if return_date_input else None
            
            print(f"Searching flights from {origin} to {destination} on {departure_date}")
            if return_date:
                print(f"Return date: {return_date}")
            
            # Prepare search parameters
            search_params = {
                "originLocationCode": origin.upper(),
                "destinationLocationCode": destination.upper(), 
                "departureDate": departure_date,
                "adults": passengers,
                "max": 10
            }
            
            if return_date:
                search_params["returnDate"] = return_date
            if travel_class and travel_class.upper() != "ECONOMY":
                search_params["travelClass"] = travel_class.upper()
            
            # Execute search
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
                    "total_found": len(flights)
                }
            else:
                raise Exception(f"Flight search failed with status {response.status_code}: {response.body if hasattr(response, 'body') else 'Unknown error'}")
                
        except Exception as e:
            # Enhanced error handling for common issues
            error_msg = str(e)
            if "Invalid airport code" in error_msg or "400" in error_msg:
                return {
                    "error": f"Invalid airport codes: {origin} or {destination}. Please use valid IATA codes (e.g., 'AMD' for Ahmedabad, 'COK' for Kochi)",
                    "search_info": {
                        "origin": origin,
                        "destination": destination,
                        "search_date": departure_date if 'departure_date' in locals() else None
                    },
                    "flights": [],
                    "total_found": 0
                }
            else:
                raise Exception(f"Flight search error: {error_msg}")
    
    def parse_flight_results(self, flight_data, origin, destination) -> List[Dict]:
        """Parse Amadeus API response with enhanced information"""
        flights = []
        
        for offer in flight_data[:10]:  # Up to 10 results
            try:
                itinerary = offer.itineraries[0]
                segment = itinerary.segments[0]
                
                # Parse departure and arrival times
                departure_dt = datetime.fromisoformat(segment.departure.at.replace('Z', '+00:00'))
                arrival_dt = datetime.fromisoformat(segment.arrival.at.replace('Z', '+00:00'))
                
                # Calculate if arrival is next day
                arrival_day_diff = (arrival_dt.date() - departure_dt.date()).days
                arrival_display = arrival_dt.strftime("%H:%M")
                if arrival_day_diff > 0:
                    arrival_display += f"+{arrival_day_diff}"
                
                flight = {
                    "airline": segment.carrierCode,
                    "flight_number": f"{segment.carrierCode}{segment.number}",
                    "departure_date": departure_dt.strftime("%Y-%m-%d"),
                    "departure_time": departure_dt.strftime("%H:%M"),
                    "arrival_date": arrival_dt.strftime("%Y-%m-%d"),
                    "arrival_time": arrival_display,
                    "departure_airport": segment.departure.iataCode,
                    "arrival_airport": segment.arrival.iataCode,
                    "duration": itinerary.duration,
                    "price": f"{offer.price.total} {offer.price.currency}",
                    "price_numeric": float(offer.price.total),
                    "currency": offer.price.currency,
                    "stops": len(itinerary.segments) - 1,
                    "booking_class": getattr(segment, 'cabin', 'ECONOMY'),
                    "route": f"{origin}->{destination}",
                    "is_direct": len(itinerary.segments) == 1
                }
                flights.append(flight)
                
            except (AttributeError, KeyError) as e:
                print(f"Error parsing flight data: {e}")
                continue
        
        return flights

    async def get_airport_info_tool(self, arguments: dict) -> Dict:
        """
        MCP Tool: Get airport information
        """
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
        """
        MCP Tool: Get airline information
        """
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

# Create MCP Server
server = Server("flight-search-server")
flight_service = FlightSearchMCPServer()

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_flights",
            description="Search for flights between two airports. Supports flexible date inputs like 'tomorrow', 'next Friday', 'August', 'December 15', etc. If no date is provided, searches for flights one week from today.",
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin airport code (e.g., 'AMD', 'JFK', 'LHR')"
                    },
                    "destination": {
                        "type": "string", 
                        "description": "Destination airport code (e.g., 'COK', 'LAX', 'CDG')"
                    },
                    "departure_date": {
                        "type": "string",
                        "description": "Departure date - supports formats like 'tomorrow', 'next Friday', 'August 15', '2024-08-15', etc. (optional - defaults to one week from today)"
                    },
                    "return_date": {
                        "type": "string",
                        "description": "Return date in flexible format (optional)"
                    },
                    "passengers": {
                        "type": "integer",
                        "description": "Number of passengers (default: 1)"
                    },
                    "travel_class": {
                        "type": "string",
                        "description": "Travel class: ECONOMY, BUSINESS, FIRST (default: ECONOMY)"
                    }
                },
                "required": ["origin", "destination"]
            }
        ),
        Tool(
            name="get_airport_info",
            description="Get information about an airport",
            inputSchema={
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "Airport IATA code (e.g., 'JFK', 'LHR')"
                    }
                },
                "required": ["airport_code"]
            }
        ),
        Tool(
            name="get_airline_info", 
            description="Get information about an airline",
            inputSchema={
                "type": "object",
                "properties": {
                    "airline_code": {
                        "type": "string",
                        "description": "Airline IATA code (e.g., 'BA', 'AA')"
                    }
                },
                "required": ["airline_code"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    try:
        if name == "search_flights":
            results = await flight_service.search_flights_tool(arguments)
            
            # Handle error cases
            if "error" in results:
                return [types.TextContent(type="text", text=results["error"])]
            
            # Format the response nicely
            search_info = results["search_info"]
            flights = results["flights"]
            
            if not flights:
                response = f"No flights found from {search_info['origin']} to {search_info['destination']} on {search_info['search_date']}"
            else:
                # Sort flights by price
                flights.sort(key=lambda x: x['price_numeric'])
                
                # Build response
                response = f"Found {len(flights)} flights from {search_info['origin']} to {search_info['destination']} on {search_info['search_date']}:\n\n"
                
                # Show flights
                for i, flight in enumerate(flights, 1):
                    duration_clean = flight['duration'].replace('PT', '').replace('H', 'h ').replace('M', 'm')
                    stops_text = "direct" if flight['is_direct'] else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"
                    
                    response += f"{i}. {flight['price']} - {flight['flight_number']}\n"
                    response += f"   {flight['departure_time']} → {flight['arrival_time']} ({duration_clean}, {stops_text})\n"
                    response += f"   {flight['departure_airport']} → {flight['arrival_airport']}\n\n"
                
                # Add summary
                cheapest = flights[0]  # Already sorted by price
                direct_flights = [f for f in flights if f['is_direct']]
                
                response += "Summary:\n"
                response += f"• Cheapest: {cheapest['price']} - {cheapest['flight_number']}\n"
                if direct_flights:
                    fastest_direct = min(direct_flights, key=lambda x: x['duration'])
                    response += f"• Best direct: {fastest_direct['price']} - {fastest_direct['flight_number']}\n"
                response += f"• All flights on {search_info['search_date']}"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "get_airport_info":
            result = await flight_service.get_airport_info_tool(arguments)
            return [types.TextContent(
                type="text", 
                text=json.dumps(result, indent=2)
            )]
            
        elif name == "get_airline_info":
            result = await flight_service.get_airline_info_tool(arguments)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    arguments: dict

@app.post("/mcp_method/")
def execute_mcp_method(item: Item):
    try:
        # Call the MCP tool
        result = handle_call_tool(item.arguments)

        # If tool execution was successfully, return result
        return {"status": "success", "data": result}
        
    except Exception as error:
        return {"status": "error", "message": str(error)}
# async def main():
#     """Run the MCP server"""
#     # MCP servers communicate over stdio
#     from mcp.server.stdio import stdio_server
    
#     async with stdio_server() as (read_stream, write_stream):
#         await server.run(
#             read_stream,
#             write_stream,
#             InitializationOptions(
#                 server_name="flight-search",
#                 server_version="1.0.0",
#                 capabilities=server.get_capabilities(
#                     notification_options=NotificationOptions(),
#                     experimental_capabilities={},
#                 ),
#             ),
#         )

if __name__ == "__main__":
    asyncio.run(main())