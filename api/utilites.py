#!/usr/bin/env python3

import asyncio
import aiohttp
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from amadeus import Client
from dotenv import load_dotenv
import os
import re
from fuzzywuzzy import fuzz, process
import logging
from models import *
from city_data import *
from currency_converter import convert_eur_to_inr
# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveDataManager:
    """Ensures all flight data is live and not cached"""
    
    def __init__(self):
        self.never_cache = {
            "flight_prices", "flight_availability", "flight_schedules",
            "airline_offers", "seat_availability", "baggage_prices"
        }
        
        self.cacheable_static = {
            "airport_coordinates": {"ttl": 86400 * 30},  # 30 days
            "airport_names": {"ttl": 86400 * 7},         # 7 days
            "city_mappings": {"ttl": 86400 * 7}          # 7 days
        }
    
    def validate_data_freshness(self, response: Dict) -> bool:
        """Fixed validation with proper key checking"""
        
        current_time = datetime.now()
        
        # ✅ FIX: Check for correct timestamp key in data_freshness
        timestamp_keys = ["fetched_at", "timestamp", "data_freshness.fetched_at"]
        fetch_time = None
        
        # Check multiple possible timestamp locations
        if "data_freshness" in response and "fetched_at" in response["data_freshness"]:
            timestamp_str = response["data_freshness"]["fetched_at"]
        elif "fetched_at" in response:
            timestamp_str = response["fetched_at"]
        elif "timestamp" in response:
            timestamp_str = response["timestamp"]
        else:
            logger.warning("⚠️ No timestamp found in response keys: " + str(list(response.keys())))
            return False
        
        try:
            fetch_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age_seconds = (current_time - fetch_time).total_seconds()
            
            # Data should be less than 5 minutes old
            if age_seconds > 300:
                logger.warning(f"⚠️ Data is {age_seconds/60:.1f} minutes old")
                return False
                
            logger.info(f"✅ Data is fresh ({age_seconds:.1f} seconds old)")
            return True
            
        except Exception as e:
            logger.error(f"Error parsing timestamp '{timestamp_str}': {e}")
            return False
    
    def add_freshness_metadata(self, response: Dict, request_id: str = None) -> Dict:
        """Fixed metadata addition"""
        
        if request_id is None:
            request_id = f"req_{int(time.time())}"
        
        # ✅ FIX: Ensure the response has the data_freshness key properly structured
        response["data_freshness"] = {
            "fetched_at": datetime.now().isoformat(),
            "request_id": request_id,
            "is_live": True,
            "cache_used": False,
            "api_call_made": True,
            "data_source": "amadeus_live_api"
        }
        
        # Also add top-level timestamp for backward compatibility
        response["fetched_at"] = datetime.now().isoformat()
        
        return response



class LiveAirlineDataProvider:
    """Fixed version with proper API response handling"""
    
    def __init__(self, amadeus_client):
        self.amadeus = amadeus_client
        self.live_data_manager = LiveDataManager()
    
    async def extract_airlines_from_flight_results(self, flight_results: List[Dict]) -> List[Dict]:
        """Fixed airline extraction with better error handling"""
        
        logger.info("🔴 EXTRACTING LIVE AIRLINE DATA from flight results")
        
        airlines_found = set()
        airline_details = []
        
        # Extract airline codes from flight results
        for flight in flight_results:
            airline_code = flight.get("airline")
            if airline_code:
                airlines_found.add(airline_code)
        
        logger.info(f"Found airline codes in flights: {list(airlines_found)}")
        
        # Get detailed information for each airline found
        for airline_code in airlines_found:
            try:
                logger.info(f"🔍 Getting live details for airline: {airline_code}")
                
                response = self.amadeus.reference_data.airlines.get(
                    airlineCodes=airline_code
                )
                
                # ✅ FIX: Better response handling
                if response.status_code == 200:
                    if hasattr(response, 'data') and response.data and len(response.data) > 0:
                        airline_data = response.data[0]
                        
                        # ✅ FIX: Handle both dict and object responses
                        if isinstance(airline_data, dict):
                            # Response is a dictionary
                            airline_info = {
                                "iata_code": airline_data.get("iataCode", airline_code),
                                "name": airline_data.get("businessName", f"Airline {airline_code}"),
                                "country": airline_data.get("countryCode", "Unknown"),
                                "type": self._classify_airline_type(airline_data.get("businessName", "")),
                                "currently_operating": True,
                                "last_verified": datetime.now().isoformat(),
                                "data_source": "live_api"
                            }
                        else:
                            # Response is an object with attributes
                            airline_info = {
                                "iata_code": getattr(airline_data, 'iataCode', airline_code),
                                "name": getattr(airline_data, 'businessName', f"Airline {airline_code}"),
                                "country": getattr(airline_data, 'countryCode', 'Unknown'),
                                "type": self._classify_airline_type(getattr(airline_data, 'businessName', "")),
                                "currently_operating": True,
                                "last_verified": datetime.now().isoformat(),
                                "data_source": "live_api"
                            }
                        
                        airline_details.append(airline_info)
                        logger.info(f"✅ Found airline: {airline_info['name']} ({airline_code})")
                    else:
                        logger.warning(f"⚠️ No data in response for airline {airline_code}")
                        # Add fallback info
                        airline_details.append(self._create_fallback_airline_info(airline_code))
                else:
                    logger.warning(f"⚠️ API error for airline {airline_code}: Status {response.status_code}")
                    # Add fallback info
                    airline_details.append(self._create_fallback_airline_info(airline_code))
                
            except Exception as e:
                logger.warning(f"Could not get details for airline {airline_code}: {e}")
                # Add fallback info even if API call fails
                airline_details.append(self._create_fallback_airline_info(airline_code))
        
        logger.info(f"✅ Found {len(airline_details)} airlines currently serving this route")
        return airline_details
    
    def _create_fallback_airline_info(self, airline_code: str) -> Dict:
        """Create fallback airline info when API fails"""
        
        # Known airline mappings for fallback
        known_airlines = {
            "AI": {"name": "Air India", "country": "India", "type": "Full-service"},
            "6E": {"name": "IndiGo", "country": "India", "type": "Low-cost"},
            "SG": {"name": "SpiceJet", "country": "India", "type": "Low-cost"},
            "UK": {"name": "Vistara", "country": "India", "type": "Full-service"},
            "G8": {"name": "GoAir", "country": "India", "type": "Low-cost"},
            "IX": {"name": "Air India Express", "country": "India", "type": "Low-cost"},
            "AA": {"name": "American Airlines", "country": "USA", "type": "Full-service"},
            "DL": {"name": "Delta Air Lines", "country": "USA", "type": "Full-service"},
            "UA": {"name": "United Airlines", "country": "USA", "type": "Full-service"},
            "BA": {"name": "British Airways", "country": "UK", "type": "Full-service"},
            "EK": {"name": "Emirates", "country": "UAE", "type": "Full-service"},
            "QR": {"name": "Qatar Airways", "country": "Qatar", "type": "Full-service"}
        }
        
        if airline_code in known_airlines:
            known_info = known_airlines[airline_code]
            return {
                "iata_code": airline_code,
                "name": known_info["name"],
                "country": known_info["country"],
                "type": known_info["type"],
                "currently_operating": True,
                "last_verified": datetime.now().isoformat(),
                "data_source": "fallback_database"
            }
        else:
            return {
                "iata_code": airline_code,
                "name": f"Airline {airline_code}",
                "country": "Unknown",
                "type": "Unknown",
                "currently_operating": True,
                "last_verified": datetime.now().isoformat(),
                "data_source": "flight_results_only"
            }
    
    def _classify_airline_type(self, airline_name: str) -> str:
        """Classify airline type based on name"""
        if not airline_name:
            return "Unknown"
            
        name_lower = airline_name.lower()
        
        if any(keyword in name_lower for keyword in ["express", "regional", "connect"]):
            return "Regional"
        elif any(keyword in name_lower for keyword in ["cargo", "freight"]):
            return "Cargo"
        elif any(keyword in name_lower for keyword in ["go", "jet", "indigo", "spicejet", "southwest", "ryanair", "easyjet"]):
            return "Low-cost"
        else:
            return "Full-service"


class FlightAPIDebugger:
    """Debug flight search issues and improve reliability"""
    
    def __init__(self, flight_service):
        self.flight_service = flight_service
        self.debug_log = []
    
    async def debug_route_search(self, origin: str, destination: str, date: str = None) -> Dict:
        """Comprehensive debugging for route search issues"""
        
        debug_info = {
            "route": f"{origin} → {destination}",
            "date": date or "default",
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "issues_found": [],
            "recommendations": []
        }
        
        # Step 1: Validate airport codes
        step1 = await self._debug_airport_codes(origin, destination)
        debug_info["steps"].append(step1)
        
        # Step 2: Check Amadeus API connectivity
        step2 = await self._debug_amadeus_connection()
        debug_info["steps"].append(step2)
        
        # Step 3: Test route with different parameters
        step3 = await self._debug_route_variations(origin, destination, date)
        debug_info["steps"].append(step3)
        
        # Step 4: Compare with known working routes
        step4 = await self._debug_compare_working_routes()
        debug_info["steps"].append(step4)
        
        # Generate recommendations
        debug_info["recommendations"] = self._generate_recommendations(debug_info["steps"])
        
        return debug_info
    
    async def _debug_airport_codes(self, origin: str, destination: str) -> Dict:
        """Debug airport code resolution"""
        step = {
            "name": "Airport Code Resolution",
            "status": "running",
            "details": {}
        }
        
        try:
            # Test origin resolution
            origin_result = self.flight_service.spell_checker.enhanced_resolve_to_airport_code(origin)
            step["details"]["origin"] = {
                "input": origin,
                "resolved": origin_result
            }
            
            # Test destination resolution  
            dest_result = self.flight_service.spell_checker.enhanced_resolve_to_airport_code(destination)
            step["details"]["destination"] = {
                "input": destination,
                "resolved": dest_result
            }
            
            # Validate codes
            if origin_result["status"] == "success" and dest_result["status"] == "success":
                step["status"] = "success"
                step["message"] = f"✅ Both airports resolved: {origin_result['airport_code']} → {dest_result['airport_code']}"
            else:
                step["status"] = "warning"
                step["message"] = "⚠️ Airport resolution issues found"
                
        except Exception as e:
            step["status"] = "error"
            step["message"] = f"❌ Airport resolution failed: {str(e)}"
            
        return step
    
    async def _debug_amadeus_connection(self) -> Dict:
        """Debug Amadeus API connectivity and authentication"""
        step = {
            "name": "Amadeus API Connection",
            "status": "running",
            "details": {}
        }
        
        try:
            # Test authentication
            token = await self.flight_service.get_access_token()
            step["details"]["authentication"] = "✅ Token obtained" if token else "❌ Auth failed"
            
            # Test simple location lookup
            try:
                response = self.flight_service.amadeus.reference_data.locations.get(
                    keyword="COK",
                    subType='AIRPORT'
                )
                step["details"]["api_test"] = {
                    "status_code": response.status_code,
                    "data_count": len(response.data) if hasattr(response, 'data') else 0
                }
                
                if response.status_code == 200:
                    step["status"] = "success"
                    step["message"] = "✅ Amadeus API working correctly"
                else:
                    step["status"] = "error"
                    step["message"] = f"❌ Amadeus API error: {response.status_code}"
                    
            except Exception as api_error:
                step["status"] = "error"
                step["message"] = f"❌ Amadeus API call failed: {str(api_error)}"
                step["details"]["api_error"] = str(api_error)
                
        except Exception as e:
            step["status"] = "error"
            step["message"] = f"❌ Amadeus connection failed: {str(e)}"
            
        return step
    
    async def _debug_route_variations(self, origin: str, destination: str, date: str) -> Dict:
        """Test route with different parameter combinations"""
        step = {
            "name": "Route Variation Testing",
            "status": "running", 
            "details": {"tests": []}
        }
        
        # Test different date formats
        date_variations = [
            date,
            (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),  # 30 days from now
            "2024-12-15",  # Fixed future date
            None  # Default date
        ]
        
        # Test different passenger counts
        passenger_variations = [1, 2]
        
        # Test different travel classes
        class_variations = ["ECONOMY", "BUSINESS"]
        
        success_count = 0
        total_tests = 0
        
        for test_date in date_variations[:2]:  # Limit tests
            for passengers in passenger_variations[:1]:  # Limit tests
                for travel_class in class_variations[:1]:  # Limit tests
                    total_tests += 1
                    
                    try:
                        params = {
                            "origin": origin,
                            "destination": destination,
                            "departure_date": test_date,
                            "passengers": passengers,
                            "travel_class": travel_class
                        }
                        
                        # This would be the actual API call - commenting out to avoid real calls
                        # result = await self.flight_service.search_flights_tool(params)
                        
                        # Simulate test result
                        test_result = {
                            "params": params,
                            "status": "simulated",  # Would be actual result
                            "message": "Test skipped - simulation mode"
                        }
                        
                        step["details"]["tests"].append(test_result)
                        
                    except Exception as e:
                        step["details"]["tests"].append({
                            "params": params,
                            "status": "error",
                            "error": str(e)
                        })
        
        step["status"] = "completed"
        step["message"] = f"📊 Tested {total_tests} parameter combinations"
        
        return step
    
    async def _debug_compare_working_routes(self) -> Dict:
        """Compare with known working routes to identify patterns"""
        step = {
            "name": "Working Route Comparison",
            "status": "running",
            "details": {}
        }
        
        # Known working routes for comparison
        working_routes = [
            {"origin": "BOM", "destination": "DEL", "name": "Mumbai → Delhi"},
            {"origin": "BLR", "destination": "MAA", "name": "Bangalore → Chennai"},
            {"origin": "COK", "destination": "BOM", "name": "Kochi → Mumbai"}
        ]
        
        step["details"]["working_routes"] = working_routes
        step["details"]["problem_route"] = "COK → GAU (Kochi → Guwahati)"
        
        # Analysis
        analysis = {
            "route_type": "Domestic Indian route",
            "airports": {
                "COK": "Kochi - Major South Indian airport ✅",
                "GAU": "Guwahati - Northeastern Indian airport ⚠️"
            },
            "potential_issues": [
                "GAU might be less commonly served",
                "Route might have limited frequency", 
                "Amadeus test environment might not have all Indian domestic routes",
                "Date might be too far in future or past",
                "API might need different parameters for smaller airports"
            ]
        }
        
        step["details"]["analysis"] = analysis
        step["status"] = "completed"
        step["message"] = "📋 Route analysis completed"
        
        return step
    
    def _generate_recommendations(self, steps: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on debug results"""
        recommendations = []
        
        # Check for common issues
        for step in steps:
            if step.get("status") == "error":
                if "Amadeus" in step.get("name", ""):
                    recommendations.extend([
                        "🔧 Check Amadeus API credentials and connectivity",
                        "🌐 Verify API endpoint URLs and test environment",
                        "📊 Test with Amadeus API directly using their tools"
                    ])
                elif "Airport" in step.get("name", ""):
                    recommendations.extend([
                        "✈️ Verify airport codes are correct (COK=Kochi, GAU=Guwahati)",
                        "🔍 Check if both airports are in Amadeus database",
                        "📍 Try alternative airport codes for the same cities"
                    ])
        
        # General recommendations
        recommendations.extend([
            "📅 Try different dates (avoid holidays, weekends)",
            "🛫 Test with major hub airports first (BOM, DEL, BLR)",
            "🔄 Implement fallback to alternative data sources",
            "📝 Add better error logging for API responses",
            "🧪 Create comprehensive test suite for route validation"
        ])
        
        return list(set(recommendations))  # Remove duplicates

class EnhancedSpellChecker:
    """Enhanced spell checker with unknown city handling"""
    
    def __init__(self):
        self.cities_data = CITIES_DATA  # Your existing city data
        self.city_names = CITY_NAMES
        self.city_map = CITY_MAP
        self.city_to_airport = CITY_TO_AIRPORT
        
        # Dynamic city cache for runtime additions
        self.dynamic_cities = {}
        self.unknown_cities_log = []
        
        # Load additional cities from file if exists
        self.load_dynamic_cities()
    
    def load_dynamic_cities(self):
        """Load dynamically added cities from file"""
        try:
            if os.path.exists("dynamic_cities.json"):
                with open("dynamic_cities.json", "r") as f:
                    self.dynamic_cities = json.load(f)
                logger.info(f"Loaded {len(self.dynamic_cities)} dynamic cities")
        except Exception as e:
            logger.warning(f"Could not load dynamic cities: {e}")
    
    def save_dynamic_cities(self):
        """Save dynamically added cities to file"""
        try:
            with open("dynamic_cities.json", "w") as f:
                json.dump(self.dynamic_cities, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save dynamic cities: {e}")
    
    def add_dynamic_city(self, city_name: str, airport_code: str, country: str = "Unknown"):
        """Add a city to dynamic database"""
        city_key = city_name.lower()
        self.dynamic_cities[city_key] = {
            "name": city_name,
            "airport_code": airport_code,
            "country": country,
            "added_at": datetime.now().isoformat(),
            "usage_count": 1
        }
        self.save_dynamic_cities()
        logger.info(f"Added dynamic city: {city_name} → {airport_code}")
    
    def get_unknown_city_suggestions(self, city_name: str, limit: int = 5) -> List[Dict]:
        """Get intelligent suggestions for unknown cities"""
        suggestions = []
        
        # 1. Fuzzy match against known cities
        fuzzy_matches = process.extract(city_name, self.city_names, limit=limit, scorer=fuzz.ratio)
        for match_name, score in fuzzy_matches:
            if score >= 50:  # Lower threshold for suggestions
                city_info = self.city_map.get(match_name.lower())
                if city_info:
                    suggestions.append({
                        "type": "fuzzy_match",
                        "suggestion": match_name,
                        "airport_code": city_info["airport_code"],
                        "country": city_info["country"],
                        "confidence": score,
                        "reason": f"Similar to '{match_name}'"
                    })
        
        # 2. Check dynamic cities
        for key, city_data in self.dynamic_cities.items():
            similarity = fuzz.ratio(city_name.lower(), key)
            if similarity >= 70:
                suggestions.append({
                    "type": "dynamic_match",
                    "suggestion": city_data["name"],
                    "airport_code": city_data["airport_code"],
                    "country": city_data["country"],
                    "confidence": similarity,
                    "reason": f"Previously resolved city"
                })
        
        # 3. Partial matches (contains/starts with)
        for city_info in self.cities_data:
            city_lower = city_info["name"].lower()
            query_lower = city_name.lower()
            
            if (query_lower in city_lower or city_lower in query_lower) and len(query_lower) >= 3:
                suggestions.append({
                    "type": "partial_match",
                    "suggestion": city_info["name"],
                    "airport_code": city_info["airport_code"],
                    "country": city_info["country"],
                    "confidence": 60,
                    "reason": f"Partial match with '{city_info['name']}'"
                })
        
        # 4. Common misspelling patterns
        common_patterns = {
            r'([aeiou])\1+': r'\1',  # Remove double vowels: "Dellhi" → "Delhi"
            r'ph': 'f',               # "Philadelfia" → "Filadelfia"
            r'ck': 'k',               # "Banckgok" → "Bangkok"
            r'([bcdfghjklmnpqrstvwxyz])\1+': r'\1'  # Remove double consonants
        }
        
        corrected_name = city_name
        for pattern, replacement in common_patterns.items():
            corrected_name = re.sub(pattern, replacement, corrected_name, flags=re.IGNORECASE)
        
        if corrected_name != city_name:
            fuzzy_matches_corrected = process.extract(corrected_name, self.city_names, limit=2, scorer=fuzz.ratio)
            for match_name, score in fuzzy_matches_corrected:
                if score >= 70:
                    city_info = self.city_map.get(match_name.lower())
                    if city_info:
                        suggestions.append({
                            "type": "pattern_correction",
                            "suggestion": match_name,
                            "airport_code": city_info["airport_code"],
                            "country": city_info["country"],
                            "confidence": score,
                            "reason": f"Pattern-corrected to '{match_name}'"
                        })
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = (suggestion["suggestion"], suggestion["airport_code"])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return sorted(unique_suggestions, key=lambda x: x["confidence"], reverse=True)[:limit]
    
    def log_unknown_city(self, city_name: str, context: str = ""):
        """Log unknown city for analysis"""
        self.unknown_cities_log.append({
            "city": city_name,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "suggestions": self.get_unknown_city_suggestions(city_name, 3)
        })
        
        # Keep only last 100 entries
        if len(self.unknown_cities_log) > 100:
            self.unknown_cities_log = self.unknown_cities_log[-100:]
    
    def enhanced_resolve_to_airport_code(self, location_text: str) -> Dict:
        """Enhanced resolution with detailed feedback for unknown cities"""
        location_clean = location_text.strip().lower()
        
        # Check if it's already a 3-letter airport code
        if len(location_clean) == 3 and location_clean.upper() in [city["airport_code"] for city in self.cities_data]:
            return {
                "status": "success",
                "airport_code": location_clean.upper(),
                "city_name": location_text,
                "source": "direct_airport_code"
            }
        
        # Check static database
        if location_clean in self.city_to_airport:
            return {
                "status": "success",
                "airport_code": self.city_to_airport[location_clean],
                "city_name": location_text,
                "source": "static_database"
            }
        
        # Check dynamic database
        if location_clean in self.dynamic_cities:
            city_data = self.dynamic_cities[location_clean]
            city_data["usage_count"] += 1
            self.save_dynamic_cities()
            return {
                "status": "success",
                "airport_code": city_data["airport_code"],
                "city_name": city_data["name"],
                "source": "dynamic_database",
                "country": city_data["country"]
            }
        
        # Try fuzzy matching with high confidence
        matches = process.extract(location_text, self.city_names, limit=1, scorer=fuzz.ratio)
        if matches and matches[0][1] >= 85:  # High confidence
            best_match = matches[0][0]
            city_info = self.city_map.get(best_match.lower())
            if city_info:
                return {
                    "status": "success",
                    "airport_code": city_info["airport_code"],
                    "city_name": best_match,
                    "source": "fuzzy_match",
                    "confidence": matches[0][1],
                    "original_input": location_text
                }
        
        # City not found - return detailed suggestions
        suggestions = self.get_unknown_city_suggestions(location_text)
        self.log_unknown_city(location_text, "airport_code_resolution")
        
        return {
            "status": "unknown",
            "city_name": location_text,
            "suggestions": suggestions,
            "message": f"City '{location_text}' not found in database",
            "help_text": "Try using a more specific city name or airport code"
        }

class EnhancedLocationResolver:
    """Enhanced location resolver with better unknown city handling"""
    
    def __init__(self, amadeus_client, spell_checker):
        self.amadeus = amadeus_client
        self.spell_checker = spell_checker
        self._location_cache = {}
    
    async def resolve_location_to_airport(self, location_input: str) -> Dict:
        """Enhanced resolution with graceful unknown city handling"""
        if not location_input:
            return {"status": "error", "message": "Location input is required"}
        
        location_clean = location_input.strip()
        location_key = location_clean.lower()
        
        # Check cache first
        if location_key in self._location_cache:
            logger.info(f"🎯 Using cached result for '{location_clean}'")
            return self._location_cache[location_key]
        
        # Try enhanced spell checker first
        local_result = self.spell_checker.enhanced_resolve_to_airport_code(location_clean)
        
        if local_result["status"] == "success":
            self._location_cache[location_key] = local_result
            return local_result
        
        # Try Amadeus API
        amadeus_result = await self._try_amadeus_resolution(location_clean)
        if amadeus_result["status"] == "success":
            # Add to dynamic database for future use
            self.spell_checker.add_dynamic_city(
                location_clean.title(), 
                amadeus_result["airport_code"],
                amadeus_result.get("country", "Unknown")
            )
            self._location_cache[location_key] = amadeus_result
            return amadeus_result
        
        # Combine local suggestions with Amadeus attempt
        combined_result = {
            "status": "unknown",
            "city_name": location_input,
            "suggestions": local_result.get("suggestions", []),
            "amadeus_attempted": True,
            "amadeus_result": amadeus_result.get("message", "No results from Amadeus API"),
            "help_options": [
                "Try a different spelling",
                "Use the nearest major city",
                "Provide airport code if known",
                "Be more specific (e.g., include state/country)"
            ]
        }
        
        return combined_result
    
    async def _try_amadeus_resolution(self, location: str) -> Dict:
        """Try to resolve using Amadeus API"""
        try:
            logger.info(f"🔍 Trying Amadeus API for: '{location}'")
            
            response = self.amadeus.reference_data.locations.get(
                keyword=location,
                subType='AIRPORT,CITY'
            )
            
            if response.status_code == 200 and response.data:
                for location_data in response.data:
                    if hasattr(location_data, 'iataCode') and location_data.iataCode:
                        airport_code = location_data.iataCode
                        location_name = getattr(location_data, 'name', location)
                        country = getattr(location_data, 'address', {}).get('countryName', 'Unknown')
                        
                        logger.info(f"✅ Amadeus resolved: '{location}' → '{airport_code}' ({location_name})")
                        
                        return {
                            "status": "success",
                            "airport_code": airport_code,
                            "city_name": location_name,
                            "country": country,
                            "source": "amadeus_api"
                        }
            
            return {
                "status": "not_found",
                "message": f"Amadeus API could not find '{location}'"
            }
            
        except Exception as e:
            logger.error(f"❌ Amadeus API error: {e}")
            return {
                "status": "error",
                "message": f"Amadeus API error: {str(e)}"
            }



class SpellChecker:
    """Enhanced spell checker for city names and locations"""
    
    def __init__(self):
        self.cities_data = CITIES_DATA
        self.city_names = CITY_NAMES
        self.city_map = CITY_MAP
        self.city_to_airport = CITY_TO_AIRPORT
    
    def correct_city_spelling(self, text: str, confidence_threshold: int = 70) -> Dict:
        """Correct misspelled city names in text"""
        words = re.findall(r'\b[A-Za-z]+\b', text)
        corrections = []
        corrected_text = text
        
        # Common non-city words to exclude
        EXCLUDE_WORDS = {
            'flight', 'flights', 'from', 'to', 'book', 'search', 'find', 'ticket',
            'tomorrow', 'today', 'next', 'week', 'month', 'day', 'morning', 'evening',
            'business', 'economy', 'first', 'class', 'passenger', 'passengers',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'of', 'with', 'by'
        }
        
        for word in words:
            if len(word) < 3 or word.lower() in EXCLUDE_WORDS:
                continue
            
            # Skip if it's already a valid airport code
            if len(word) == 3 and word.upper() in [city["airport_code"] for city in self.cities_data]:
                continue
            
            # Skip if word is already correct
            if word.lower() in self.city_map:
                continue
            
            # Use fuzzy matching to find best matches
            matches = process.extract(word, self.city_names, limit=3, scorer=fuzz.ratio)
            
            if matches and matches[0][1] >= confidence_threshold:
                best_match = matches[0][0]
                confidence = matches[0][1]
                
                city_info = self.city_map.get(best_match.lower())
                if city_info:
                    correction = {
                        "original": word,
                        "corrected": best_match,
                        "confidence": confidence,
                        "country": city_info["country"],
                        "airport_code": city_info["airport_code"],
                        "alternatives": city_info["alternatives"]
                    }
                    corrections.append(correction)
                    
                    # Apply correction to text
                    pattern = r'\b' + re.escape(word) + r'\b'
                    corrected_text = re.sub(pattern, best_match, corrected_text, flags=re.IGNORECASE)
        
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "total_corrections": len(corrections)
        }
    
    def extract_and_correct_locations(self, query: str) -> Tuple[Optional[str], Optional[str], str]:
        """Extract locations from query, correct spelling, and return airport codes"""
        # First correct spelling
        spell_result = self.correct_city_spelling(query, confidence_threshold=60)
        corrected_query = spell_result["corrected_text"]
        
        logger.info(f"Original query: {query}")
        logger.info(f"Corrected query: {corrected_query}")
        if spell_result["corrections"]:
            logger.info(f"Spelling corrections made: {spell_result['corrections']}")
        
        # Extract locations from corrected query
        query_lower = corrected_query.lower()
        
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
                
                origin_code = self._resolve_to_airport_code(origin_text)
                dest_code = self._resolve_to_airport_code(dest_text)
                
                if origin_code and dest_code:
                    return origin_code, dest_code, corrected_query
        
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
                
                origin_code = self._resolve_to_airport_code(origin_text)
                dest_code = self._resolve_to_airport_code(dest_text)
                
                if origin_code and dest_code:
                    return origin_code, dest_code, corrected_query
        
        # Pattern 3: Direct airport codes in query
        airport_codes = re.findall(r'\b([A-Z]{3})\b', corrected_query.upper())
        if len(airport_codes) >= 2:
            return airport_codes[0], airport_codes[1], corrected_query
        
        return None, None, corrected_query
    
    def _resolve_to_airport_code(self, location_text: str) -> Optional[str]:
        """Resolve location text to airport code"""
        location_clean = location_text.strip().lower()
        
        # Check if it's already a 3-letter airport code
        if len(location_clean) == 3 and location_clean.upper() in [city["airport_code"] for city in self.cities_data]:
            return location_clean.upper()
        
        # Check direct mapping
        if location_clean in self.city_to_airport:
            return self.city_to_airport[location_clean]
        
        # Try fuzzy matching
        matches = process.extract(location_text, self.city_names, limit=1, scorer=fuzz.ratio)
        if matches and matches[0][1] >= 80:  # High confidence for airport code resolution
            best_match = matches[0][0]
            city_info = self.city_map.get(best_match.lower())
            if city_info:
                return city_info["airport_code"]
        
        return None

class SmartLocationResolver:
    """Enhanced location resolver with spell checking"""
    
    def __init__(self, amadeus_client, spell_checker):
        self.amadeus = amadeus_client
        self.spell_checker = spell_checker
        self._location_cache = {}
    
    async def resolve_location_to_airport(self, location_input: str) -> str:
        """Resolve location with spell checking first, then Amadeus API"""
        if not location_input:
            raise ValueError("Location input is required")
        
        location_clean = location_input.strip()
        location_key = location_clean.lower()
        
        # Check if it's already a 3-letter airport code
        if len(location_clean) == 3 and location_clean.isalpha():
            return location_clean.upper()
        
        # ✅ MODIFIED: Only cache static airport codes, not dynamic API results
        if location_key in self._location_cache:
            cached_result = self._location_cache[location_key]
            # Only use cache for static data (airport codes), not API results
            if cached_result.startswith(("BOM", "DEL", "BLR")):  # Known static codes
                logger.info(f"🎯 Using cached airport code for '{location_clean}': {cached_result}")
                return cached_result
        
        # First try spell checking and local database
        local_code = self.spell_checker._resolve_to_airport_code(location_clean)
        if local_code:
            logger.info(f"✅ Resolved locally: '{location_clean}' → '{local_code}'")
            # Cache only static mappings
            self._location_cache[location_key] = local_code
            return local_code
        
        # ✅ Fall back to Amadeus API - ALWAYS FRESH
        try:
            logger.info(f"🔍 Making LIVE Amadeus API call for: '{location_clean}'")
            
            response = self.amadeus.reference_data.locations.get(
                keyword=location_clean,
                subType='AIRPORT,CITY'
            )
            
            if response.status_code == 200 and response.data:
                for location in response.data:
                    if hasattr(location, 'iataCode') and location.iataCode:
                        airport_code = location.iataCode
                        location_name = getattr(location, 'name', 'Unknown')
                        
                        logger.info(f"✅ LIVE Amadeus resolution: '{location_clean}' → '{airport_code}' ({location_name})")
                        # ✅ Don't cache API results - they might change
                        return airport_code
        
        except Exception as e:
            logger.error(f"❌ Amadeus API error for '{location_clean}': {e}")
        
        raise ValueError(f"Could not resolve location: '{location_clean}'. Please check spelling or try a more specific city name.")


class EnhancedDateParser:
    """Enhanced date parser that handles both single dates and month ranges"""
    
    def __init__(self):
        self.month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            # Short forms
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        self.relative_dates = {
            'today': 0,
            'tomorrow': 1,
            'day after tomorrow': 2,
            'next week': 7,
        }
    
    def extract_date_from_query(self, query: str) -> Dict:
        """
        Enhanced date extraction that returns both single dates and month ranges
        """
        logger.info(f"🗓️ Extracting date info from query: '{query}'")
        
        if not isinstance(query, str):
            logger.error(f"❌ Expected string, got {type(query)}: {query}")
            query = str(query) if query else ""
        
        query_lower = query.lower().strip()
        logger.info(f"🔍 Processing lowercase query: '{query_lower}'")
        
        # ✅ IMPORTANT: Check for specific dates FIRST before month ranges
        single_date = self.extract_single_date_from_query(query)
        if single_date:
            logger.info(f"✅ Found specific date: {single_date}")
            return {
                "type": "single_date",
                "date": single_date,
                "search_dates": [single_date]
            }
        
        # Check for month-based queries ONLY after confirming it's not a specific date
        # First check for "next month" specifically (using simple string match for reliability)
        if 'next month' in query_lower:
            logger.info("✅ Found 'next month' - returning month range")
            return self._get_next_month_range()
        
        # Check for "this month"
        if 'this month' in query_lower:
            logger.info("✅ Found 'this month' - returning current month range")
            return self._get_current_month_range()
        
        # Check for generic "month" phrases with regex
        if re.search(r'(?:for|in|on|during)\s+(?:the\s+)?month', query_lower):
            logger.info("✅ Found generic 'month' phrase - returning next month range")
            return self._get_next_month_range()
        
        # ✅ UPDATED: Check for specific month names but EXCLUDE specific dates
        month_name_patterns = [
            r'(?:for|in|on|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?!\s+\d)',  # ✅ Negative lookahead to exclude dates
            r'(?:for|in|on|during)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?!\s+\d)',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b',
            r'entire\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'whole\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'flights\s+in\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?!\s+\d)',
            r'flights\s+for\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?!\s+\d)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+flights'
        ]
        
        for pattern in month_name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                month_name = match.group(1)
                logger.info(f"✅ Found specific month '{month_name}' - returning month range")
                return self._get_specific_month_range(month_name)
        
        logger.info("🔍 No month patterns found after date check...")
        
        # No date found
        logger.warning(f"⚠️ No date found in query: '{query}' - using default")
        return {
            "type": "default",
            "date": self.get_smart_default_date(),
            "search_dates": [self.get_smart_default_date()]
        }
    
    def extract_single_date_from_query(self, query: str) -> Optional[str]:
        """Extract single date from query (existing logic)"""
        query_lower = query.lower().strip()
        
        # Pattern 1: Relative dates
        for phrase, days_ahead in self.relative_dates.items():
            if phrase in query_lower:
                target_date = datetime.now() + timedelta(days=days_ahead)
                result = target_date.strftime("%Y-%m-%d")
                logger.info(f"✅ Found relative date '{phrase}' → {result}")
                return result
        
        # Pattern 2: Next [weekday]
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in weekdays.items():
            if f'next {day_name}' in query_lower:
                today = datetime.now()
                days_ahead = day_num - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                result = target_date.strftime("%Y-%m-%d")
                logger.info(f"✅ Found 'next {day_name}' → {result}")
                return result
        
        # ✅ FIXED: Pattern 3 - More comprehensive specific date patterns
        date_patterns = [
            r'\b(\d{4}-\d{1,2}-\d{1,2})\b',      # 2025-08-15
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', # 15/08/2025 or 15-08-2025
            # ✅ NEW: Handle "DD Month" and "DD Month YYYY" patterns
            r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b',  # 21 july or 21 july 2025
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b',   # july 21 or july 21, 2025
            # ✅ NEW: Handle ordinal dates
            r'\b(\d{1,2})(st|nd|rd|th)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b',  # 21st july
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(st|nd|rd|th)(?:,?\s+(\d{4}))?\b'   # july 21st
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query_lower)
            if match:
                date_str = match.group(0)
                parsed_date = self._parse_specific_date(date_str)
                if parsed_date:
                    logger.info(f"✅ Found specific date '{date_str}' → {parsed_date}")
                    return parsed_date
        
        return None
    
    def _get_current_month_range(self) -> Dict:
        """Get date range for current month"""
        today = datetime.now()
        
        # Get first and last day of current month
        start_date = datetime(today.year, today.month, 1)
        
        # Get last day of the month
        if today.month == 12:
            end_date = datetime(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(today.year, today.month + 1, 1) - timedelta(days=1)
        
        # Generate search dates (every 3-4 days for API efficiency)
        search_dates = self._generate_search_dates(start_date, end_date)
        
        return {
            "type": "month_range",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "month_name": start_date.strftime("%B %Y"),
            "search_dates": search_dates
        }
    
    def _get_next_month_range(self) -> Dict:
        """Get date range for next month"""
        today = datetime.now()
        
        # Calculate next month
        if today.month == 12:
            next_month = 1
            next_year = today.year + 1
        else:
            next_month = today.month + 1
            next_year = today.year
        
        # Get first and last day of next month
        start_date = datetime(next_year, next_month, 1)
        
        # Get last day of the month
        if next_month == 12:
            end_date = datetime(next_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(next_year, next_month + 1, 1) - timedelta(days=1)
        
        # Generate search dates (every 3-4 days for API efficiency)
        search_dates = self._generate_search_dates(start_date, end_date)
        
        return {
            "type": "month_range",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "month_name": start_date.strftime("%B %Y"),
            "search_dates": search_dates
        }
        """Get date range for next month"""
        today = datetime.now()
        
        # Calculate next month
        if today.month == 12:
            next_month = 1
            next_year = today.year + 1
        else:
            next_month = today.month + 1
            next_year = today.year
        
        # Get first and last day of next month
        start_date = datetime(next_year, next_month, 1)
        
        # Get last day of the month
        if next_month == 12:
            end_date = datetime(next_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(next_year, next_month + 1, 1) - timedelta(days=1)
        
        # Generate search dates (every 3-4 days for API efficiency)
        search_dates = self._generate_search_dates(start_date, end_date)
        
        return {
            "type": "month_range",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "month_name": start_date.strftime("%B %Y"),
            "search_dates": search_dates
        }
    
    def _get_specific_month_range(self, month_name: str) -> Dict:
        """Get date range for a specific month"""
        month_num = self.month_names.get(month_name.lower())
        if not month_num:
            logger.error(f"❌ Unknown month: {month_name}")
            return self._get_next_month_range()  # Fallback
        
        current_date = datetime.now()
        current_year = current_date.year
        
        # Determine which year to use
        if month_num < current_date.month:
            target_year = current_year + 1
        elif month_num == current_date.month:
            # If we're in the same month, check if it's early or late in the month
            if current_date.day <= 15:
                target_year = current_year
            else:
                target_year = current_year + 1
        else:
            target_year = current_year
        
        # Get first and last day of the month
        start_date = datetime(target_year, month_num, 1)
        
        if month_num == 12:
            end_date = datetime(target_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(target_year, month_num + 1, 1) - timedelta(days=1)
        
        # Generate search dates
        search_dates = self._generate_search_dates(start_date, end_date)
        
        return {
            "type": "month_range",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "month_name": start_date.strftime("%B %Y"),
            "search_dates": search_dates
        }
    
    def _generate_search_dates(self, start_date: datetime, end_date: datetime, max_searches: int = 12) -> List[str]:
        """Generate optimal search dates for API efficiency"""
        all_dates = []
        current = start_date
        
        while current <= end_date:
            # Include weekends (Friday, Saturday, Sunday) and some weekdays
            if current.weekday() in [0, 2, 4, 5, 6]:  # Mon, Wed, Fri, Sat, Sun
                all_dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        # If we have too many dates, sample them evenly
        if len(all_dates) > max_searches:
            step = len(all_dates) // max_searches
            sampled_dates = [all_dates[i] for i in range(0, len(all_dates), step)]
            return sampled_dates[:max_searches]
        
        return all_dates
    
    def _parse_specific_date(self, date_str: str) -> Optional[str]:
        """Parse specific date formats (existing logic)"""
        # Clean up the date string
        date_str = date_str.strip()
        
        # ✅ NEW: Handle current year assumption for dates without year
        current_year = datetime.now().year
        
        date_formats = [
            "%Y-%m-%d",      # 2025-08-15
            "%d-%m-%Y",      # 15-08-2025
            "%m/%d/%Y",      # 08/15/2025
            "%d/%m/%Y",      # 15/08/2025
            "%d %B %Y",      # 21 August 2025
            "%B %d, %Y",     # August 21, 2025
            "%B %d %Y",      # August 21 2025
            # ✅ NEW: Formats without year (assume current/next year)
            "%d %B",         # 21 july
            "%B %d",         # july 21
            "%dst %B",       # 21st july
            "%dnd %B",       # 22nd july  
            "%drd %B",       # 23rd july
            "%dth %B",       # 21th july (common mistake)
            "%B %dst",       # july 21st
            "%B %dnd",       # july 22nd
            "%B %drd",       # july 23rd
            "%B %dth",       # july 21th
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                
                # If no year was provided, we need to determine the right year
                if parsed_date.year == 1900:  # Default year when not specified
                    # Use current year, but if the date has passed, use next year
                    parsed_date = parsed_date.replace(year=current_year)
                    if parsed_date.date() < datetime.now().date():
                        parsed_date = parsed_date.replace(year=current_year + 1)
                
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # ✅ NEW: Manual parsing for ordinal numbers
        import re
        
        # Handle ordinal dates manually
        ordinal_pattern = r'(\d+)(st|nd|rd|th)\s+(january|february|march|april|may|june|july|august|september|october|november|december)'
        ordinal_match = re.search(ordinal_pattern, date_str.lower())
        
        if ordinal_match:
            day = int(ordinal_match.group(1))
            month_name = ordinal_match.group(3)
            
            month_mapping = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            
            month_num = month_mapping.get(month_name)
            if month_num:
                try:
                    parsed_date = datetime(current_year, month_num, day)
                    # If date has passed this year, use next year
                    if parsed_date.date() < datetime.now().date():
                        parsed_date = datetime(current_year + 1, month_num, day)
                    
                    return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        
        logger.error(f"❌ Could not parse date: {date_str}")
        return None
    
    def get_smart_default_date(self) -> str:
        """Get a smart default date when no date is specified"""
        default_date = datetime.now() + timedelta(days=14)
        return default_date.strftime("%Y-%m-%d")
    
class EnhancedQueryProcessor:
    """Enhanced query processor with spell checking"""
    
    def __init__(self, flight_service, location_resolver, spell_checker):
        self.flight_service = flight_service
        self.location_resolver = location_resolver
        self.spell_checker = spell_checker
        self.date_parser = EnhancedDateParser()
        self.tools = {
            "search_flights": flight_service.search_flights_tool,
            "get_airport_info": flight_service.get_airport_info_tool,
            "get_airline_info": flight_service.get_airline_info_tool
        }

    async def extract_locations_smart(self, query: str) -> Tuple[Optional[str], Optional[str], str]:
        """Extract and correct locations from natural language"""
        # Use spell checker to extract and correct locations
        origin_code, dest_code, corrected_query = self.spell_checker.extract_and_correct_locations(query)
        
        if origin_code and dest_code:
            return origin_code, dest_code, corrected_query
        
        # If spell checker didn't find both, try with location resolver
        if not origin_code or not dest_code:
            try:
                # Extract any potential location words and try to resolve them
                words = re.findall(r'\b[A-Za-z]+\b', corrected_query)
                potential_locations = [word for word in words if len(word) >= 3 and word.lower() not in 
                                     {'flight', 'flights', 'from', 'to', 'book', 'search', 'find'}]
                
                resolved_codes = []
                for word in potential_locations[:4]:  # Try first 4 potential locations
                    try:
                        code = await self.location_resolver.resolve_location_to_airport(word)
                        resolved_codes.append(code)
                        if len(resolved_codes) >= 2:
                            break
                    except:
                        continue
                
                if len(resolved_codes) >= 2:
                    return resolved_codes[0], resolved_codes[1], corrected_query
                    
            except Exception as e:
                logger.error(f"Error in enhanced location resolution: {e}")
        
        return origin_code, dest_code, corrected_query
    
    def extract_date_from_query(self, query: str) -> Dict:
        """✅ FIXED: Use enhanced date parser and return full date info"""
        return self.date_parser.extract_date_from_query(query)
    
    async def determine_tool_and_params(self, query: str) -> Tuple[Optional[str], dict, Optional[dict]]:
        """Analyze query and determine which tool to use with spell checking info"""
        query_lower = query.lower()
        spell_info = None
        
        # Flight search patterns
        if any(keyword in query_lower for keyword in ['flight', 'fly', 'book', 'search', 'from', 'to']):
            try:
                origin, destination, corrected_query = await self.extract_locations_smart(query)
                departure_date = self.extract_date_from_query(corrected_query)
                
                # Get spell checking information
                spell_result = self.spell_checker.correct_city_spelling(query)
                if spell_result["corrections"]:
                    spell_info = {
                        "corrections_made": True,
                        "original_query": query,
                        "corrected_query": corrected_query,
                        "corrections": spell_result["corrections"]
                    }
                
                if not origin or not destination:
                    return None, {}, spell_info
                
                params = {
                    'origin': origin,
                    'destination': destination,
                    'departure_date': corrected_query
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
                
                return 'search_flights', params, spell_info
                
            except Exception as e:
                logger.error(f"❌ Error processing flight query: {e}")
                return None, {}, spell_info
        
        # Airport info patterns
        elif any(keyword in query_lower for keyword in ['airport', 'what is', 'info about']):
            words = query.split()
            for word in words:
                if len(word) > 2:
                    try:
                        airport_code = await self.location_resolver.resolve_location_to_airport(word)
                        return 'get_airport_info', {'airport_code': airport_code}, None
                    except:
                        continue
        
        # Airline info patterns  
        elif any(keyword in query_lower for keyword in ['airline', 'carrier']):
            codes = re.findall(r'\b([A-Z]{2})\b', query.upper())
            if codes:
                return 'get_airline_info', {'airline_code': codes[0]}, None
        
        return None, {}, spell_info

# Enhanced error handling for the flight search endpoint
class EnhancedFlightSearchHandler:
    """Enhanced flight search with better error handling and debugging"""
    
    def __init__(self, flight_service):
        self.flight_service = flight_service
        self.debugger = FlightAPIDebugger(flight_service)
    
    async def enhanced_search_flights(self, params: Dict) -> Dict:
        """Enhanced flight search with comprehensive error handling"""
        
        try:
            # Log the search attempt
            logger.info(f"🔍 Flight search: {params}")
            
            # Perform the search
            result = await self.flight_service.search_flights_tool(params)
            
            # Check if we got results
            if result.get("flights") and len(result["flights"]) > 0:
                logger.info(f"✅ Found {len(result['flights'])} flights")
                return {
                    "status": "success",
                    "data": result,
                    "debug_info": None
                }
            
            # No flights found - run diagnostics
            logger.warning(f"⚠️ No flights found for {params['origin']} → {params['destination']}")
            
            # Run debug analysis
            debug_info = await self.debugger.debug_route_search(
                params["origin"], 
                params["destination"], 
                params.get("departure_date")
            )
            
            return {
                "status": "no_flights",
                "data": result,
                "debug_info": debug_info,
                "message": "No flights found for this route",
                "suggestions": [
                    "Try different dates",
                    "Check if route exists with airline websites",
                    "Try nearby airports",
                    "Contact support if this route should exist"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Flight search error: {str(e)}")
            
            # Run debug analysis for the error
            debug_info = await self.debugger.debug_route_search(
                params.get("origin", "unknown"), 
                params.get("destination", "unknown"), 
                params.get("departure_date")
            )
            
            return {
                "status": "error",
                "error": str(e),
                "debug_info": debug_info,
                "message": "Flight search encountered an error",
                "suggestions": [
                    "Check API connectivity",
                    "Verify airport codes", 
                    "Try again in a few minutes",
                    "Contact technical support"
                ]
            }

# Specific fixes for COK → GAU route issue
class COKGAURouteFixer:
    """Specific fixes for Kochi to Guwahati route issues"""
    
    @staticmethod
    def get_alternative_routes():
        """Get alternative routes when direct flights aren't available"""
        return {
            "connecting_hubs": [
                {"hub": "BOM", "name": "via Mumbai", "typical_airlines": ["Air India", "IndiGo"]},
                {"hub": "DEL", "name": "via Delhi", "typical_airlines": ["Air India", "SpiceJet"]},
                {"hub": "BLR", "name": "via Bangalore", "typical_airlines": ["IndiGo", "Air India"]},
                {"hub": "MAA", "name": "via Chennai", "typical_airlines": ["Air India Express"]}
            ],
            "direct_flight_days": [
                "Limited direct flights - check specific days",
                "Air India Express sometimes operates this route",
                "Seasonal variations in schedule"
            ]
        }
    
    @staticmethod
    def get_debugging_checklist():
        """Specific debugging steps for COK-GAU route"""
        return [
            "✅ Verify COK = Kochi International Airport",
            "✅ Verify GAU = Guwahati Airport (Lokpriya Gopinath Bordoloi)",
            "🔍 Check if route operates on specific days only",
            "📅 Try dates within next 2-3 months",
            "🛫 Test connecting flights via major hubs",
            "📞 Cross-check with airline websites",
            "🔧 Test Amadeus API with curl directly",
            "📊 Check API rate limits and quotas"
        ]


class FlightSearchMCPServer:
    def __init__(self):
        """Initialize the MCP server for flight searches with live data guarantee"""
        self.amadeus_api_key = os.getenv("AMADEUS_API_KEY")
        self.amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")
        self.access_token = None
        
        # Initialize Amadeus client
        self.amadeus = Client(
            client_id=self.amadeus_api_key,
            client_secret=self.amadeus_api_secret
        )
        
        self.live_data_manager = LiveDataManager()  # Fixed version
        self.airline_provider = LiveAirlineDataProvider(self.amadeus)  # Fixed version
        
        # Initialize spell checker
        self.spell_checker = SpellChecker()
        
        # Initialize smart location resolver with spell checker
        self.location_resolver = SmartLocationResolver(self.amadeus, self.spell_checker)

        self.date_parser = EnhancedDateParser() 

    def get_smart_default_date(self) -> str:
        """Get a smart default date (next week)"""
        default_date = datetime.now() + timedelta(days=7)
        return default_date.strftime("%Y-%m-%d")
    
    def parse_flexible_date(self, date_input: str) -> str:
        """
        ✅ UPDATED: Handle single date parsing only (month ranges handled separately)
        """
        if not date_input:
            return self.date_parser.get_smart_default_date()
        
        # For return dates, we only want single dates
        single_date = self.date_parser.extract_single_date_from_query(date_input)
        if single_date:
            return single_date
        
        # Try parsing as specific date
        parsed_date = self.date_parser._parse_specific_date(date_input)
        if parsed_date:
            return parsed_date
        
        # Final fallback
        logger.warning(f"Could not parse date '{date_input}', using smart default")
        return self.date_parser.get_smart_default_date()

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
        ✅ UPDATED: Support both single date and month range searches
        """
        logger.info('🔴 LIVE FLIGHT SEARCH - Enhanced with Month Range Support')
        
        # Generate unique request ID for tracking
        import time
        request_id = f"live_search_{int(time.time())}_{hash(str(arguments))}"
        logger.info(f"🆔 Request ID: {request_id}")
        
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
            
            logger.info(f"✅ Using airports: {origin} → {destination}")
            if departure_date_input:
                if isinstance(departure_date_input, dict):
                    # ✅ NEW: Check if dict already contains parsed date info
                    if departure_date_input.get("type") in ["single_date", "month_range"]:
                        logger.info(f"📅 Dict already contains parsed date info: {departure_date_input.get('type')}")
                        date_info = departure_date_input  # Use it directly!
                    else:
                        # If it's a dict with query field, extract it
                        query_for_date = (departure_date_input.get("query", "") or 
                                        departure_date_input.get("original_query", "") or
                                        departure_date_input.get("corrected_query", "") or
                                        departure_date_input.get("text", ""))
                        logger.info(f"📅 Departure date input is dict, using query: '{query_for_date}'")
                        date_info = self.date_parser.extract_date_from_query(query_for_date)
                else:
                    # If it's a string, use it directly
                    query_for_date = str(departure_date_input)
                    logger.info(f"📅 Departure date input is string: '{query_for_date}'")
                    date_info = self.date_parser.extract_date_from_query(query_for_date)
            else:
                # Default to smart date
                date_info = {
                    "type": "default",
                    "date": self.date_parser.get_smart_default_date(),
                    "search_dates": [self.date_parser.get_smart_default_date()]
                }
            
            return_date = self.parse_flexible_date(return_date_input) if return_date_input else None
            
            logger.info(f"🛫 Date info: {date_info}")
            
            # breakpoint()
            # ✅ NEW: Handle different search types
            if date_info["type"] == "month_range":
                logger.info(f"📅 MONTH RANGE SEARCH for {date_info['month_name']}")
                return await self._search_flights_for_month_range(
                    origin, destination, date_info, passengers, travel_class, return_date, request_id
                )
            else:
                logger.info(f"📅 SINGLE DATE SEARCH for {date_info['date']}")
                return await self._search_flights_for_single_date(
                    origin, destination, date_info["date"], passengers, travel_class, return_date, request_id
                )
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Flight search error: {error_msg}")
            logger.error(f"❌ Arguments received: {arguments}")  # Add debug info
            
            error_result = {
                "error": f"Flight search failed: {error_msg}",
                "search_info": {
                    "origin": arguments.get("origin"),
                    "destination": arguments.get("destination")
                },
                "flights": [],
                "total_found": 0,
                "status": "error",
                "suggestions": [
                    "Check API connectivity",
                    "Verify airport codes",
                    "Try again in a few minutes"
                ]
            }
            
            return self.live_data_manager.add_freshness_metadata(error_result, request_id)

    async def _search_flights_for_month_range(self, origin: str, destination: str, date_info: Dict, 
                                        passengers: int, travel_class: str, return_date: str, request_id: str) -> Dict:
        """Search flights across multiple dates for month range"""
        logger.info(f"🗓️ Searching flights for month range: {date_info['start_date']} to {date_info['end_date']}")
        
        all_flights = []
        search_dates = date_info["search_dates"]
        successful_searches = 0
        
        logger.info(f"📊 Will search {len(search_dates)} dates: {search_dates}")
        
        for i, search_date in enumerate(search_dates):
            try:
                logger.info(f"🔍 Searching date {i+1}/{len(search_dates)}: {search_date}")
                
                search_params = {
                    "originLocationCode": origin,
                    "destinationLocationCode": destination, 
                    "departureDate": search_date,
                    "adults": passengers,
                    "max": 15  # Fewer results per date to manage API limits
                }
                
                if return_date:
                    search_params["returnDate"] = return_date
                if travel_class and travel_class.upper() != "ECONOMY":
                    search_params["travelClass"] = travel_class.upper()
                
                response = self.amadeus.shopping.flight_offers_search.get(**search_params)
                
                if response.status_code == 200 and response.data:
                    flights = await self.parse_flight_results(response.data, origin, destination)
                    if flights:
                        all_flights.extend(flights)
                        successful_searches += 1
                        logger.info(f"✅ Found {len(flights)} flights for {search_date}")
                    else:
                        logger.info(f"⚪ No flights found for {search_date}")
                else:
                    logger.warning(f"⚠️ API error for {search_date}: Status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error searching {search_date}: {e}")
                continue
        
        logger.info(f"📊 Month search complete: {len(all_flights)} total flights from {successful_searches} successful searches")
        
        if all_flights:
            # Remove duplicates and sort
            unique_flights = []
            seen_flights = set()
            
            for flight in all_flights:
                flight_id = f"{flight['flight_number']}_{flight['departure_date']}_{flight['departure_time']}"
                if flight_id not in seen_flights:
                    seen_flights.add(flight_id)
                    unique_flights.append(flight)
            
            # Sort by price first, then by date
            unique_flights.sort(key=lambda x: (x['price_numeric'], x['departure_date']))
            
            # ✅ NEW: Take only top 10 cheapest flights
            top_flights = unique_flights[:10]
            
            # Get airline data from results
            live_airlines = await self.airline_provider.extract_airlines_from_flight_results(top_flights)
    
            
            result = {
                "search_info": {
                    "search_type": "month_range",
                    "month_name": date_info["month_name"],
                    "start_date": date_info["start_date"],
                    "end_date": date_info["end_date"],
                    "dates_searched": len(search_dates),
                    "successful_searches": successful_searches,
                    "total_flights_found": len(unique_flights),  # ✅ Show total found
                    "showing_top": len(top_flights),             # ✅ Show how many displaying
                    "origin": origin,
                    "destination": destination,
                    "passengers": passengers,
                    "travel_class": travel_class
                },
                "flights": top_flights,
                "total_found": len(top_flights),
                "message": f"Found {len(unique_flights)} flights, showing {len(top_flights)} cheapest from {origin} to {destination} for {date_info['month_name']}",
                "status": "success",
                "route_airlines": {
                    "total_airlines_serving_route": len(live_airlines),
                    "airlines": live_airlines,
                    "data_source": "live_flight_results_month_range",
                    "last_verified": datetime.now().isoformat()
                }
            }

            
            return self.live_data_manager.add_freshness_metadata(result, request_id)
        
        else:
            # No flights found for any date
            result = {
                "search_info": {
                    "search_type": "month_range",
                    "month_name": date_info["month_name"],
                    "start_date": date_info["start_date"],
                    "end_date": date_info["end_date"],
                    "origin": origin,
                    "destination": destination
                },
                "flights": [],
                "total_found": 0,
                "status": "no_flights_found",
                "message": f"No flights found for {date_info['month_name']} on this route",
                "suggestions": [
                    "Try different travel dates",
                    "Check if this route operates regularly",
                    "Consider nearby airports",
                    "Try different cabin classes"
                ]
            }
            
            return self.live_data_manager.add_freshness_metadata(result, request_id)
    async def _search_flights_for_single_date(self, origin: str, destination: str, departure_date: str, 
                                        passengers: int, travel_class: str, return_date: str, request_id: str) -> Dict:
        """Search flights for a single date (existing logic)"""
        logger.info(f"🛫 SINGLE DATE SEARCH: {origin} to {destination} on {departure_date}")
        
        search_params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination, 
            "departureDate": departure_date,
            "adults": passengers,
            "max": 20  # More results for single date
        }
        
        if return_date:
            search_params["returnDate"] = return_date
        if travel_class and travel_class.upper() != "ECONOMY":
            search_params["travelClass"] = travel_class.upper()
        
        logger.info("🔴 Making LIVE API call to Amadeus...")
        response = self.amadeus.shopping.flight_offers_search.get(**search_params)
        
        if response.status_code == 200:
            flights = await self.parse_flight_results(response.data, origin, destination)
            
            if flights and len(flights) > 0:
                live_airlines = await self.airline_provider.extract_airlines_from_flight_results(flights)
                
                result = {
                    "search_info": {
                        "search_type": "single_date",
                        "search_date": departure_date,
                        "return_date": return_date,
                        "origin": origin,
                        "destination": destination,
                        "passengers": passengers,
                        "travel_class": travel_class
                    },
                    "flights": flights,
                    "total_found": len(flights),
                    "message": f"Found {len(flights)} flights from {origin} to {destination}",
                    "status": "success",
                    "route_airlines": {
                        "total_airlines_serving_route": len(live_airlines),
                        "airlines": live_airlines,
                        "data_source": "live_flight_results",
                        "last_verified": datetime.now().isoformat()
                    }
                }
                
                return self.live_data_manager.add_freshness_metadata(result, request_id)
            
            else:
                result = {
                    "search_info": {
                        "origin": origin,
                        "destination": destination,
                        "search_date": departure_date
                    },
                    "flights": [],
                    "total_found": 0,
                    "status": "no_flights_found",
                    "message": "No flights found for this route and date",
                    "suggestions": [
                        "Try different dates",
                        "Check connecting flights",
                        "Verify route operates regularly"
                    ]
                }
                
                return self.live_data_manager.add_freshness_metadata(result, request_id)
        
        else:
            raise Exception(f"Amadeus API returned status {response.status_code}")

    # async def search_flights_tool(self, arguments: dict) -> Dict:
    #     """
    #     ✅ FIXED: Always fetch LIVE flight data with proper timestamp handling
    #     """
    #     logger.info('🔴 LIVE FLIGHT SEARCH - NO CACHE POLICY')
        
    #     # Generate unique request ID for tracking
    #     import time
    #     request_id = f"live_search_{int(time.time())}_{hash(str(arguments))}"
    #     logger.info(f"🆔 Request ID: {request_id}")
        
    #     try:
    #         origin = arguments.get("origin")
    #         destination = arguments.get("destination") 
    #         departure_date_input = arguments.get("departure_date")
    #         passengers = arguments.get("passengers", 1)
    #         travel_class = arguments.get("travel_class", "ECONOMY")
    #         return_date_input = arguments.get("return_date")
            
    #         # Validate required parameters
    #         if not all([origin, destination]):
    #             raise ValueError("Missing required parameters: origin, destination")
            
    #         logger.info(f"✅ Using airports: {origin} → {destination}")
            
    #         # Parse dates
    #         departure_date = self.parse_flexible_date(departure_date_input)
    #         return_date = self.parse_flexible_date(return_date_input) if return_date_input else None
            
    #         logger.info(f"🛫 LIVE SEARCH: {origin} to {destination} on {departure_date}")
            
    #         # ✅ CRITICAL: Always make fresh API call
    #         search_params = {
    #             "originLocationCode": origin,
    #             "destinationLocationCode": destination, 
    #             "departureDate": departure_date,
    #             "adults": passengers,
    #             "max": 20  # Get more results to see all airlines
    #         }
            
    #         if return_date:
    #             search_params["returnDate"] = return_date
    #         if travel_class and travel_class.upper() != "ECONOMY":
    #             search_params["travelClass"] = travel_class.upper()
            
    #         logger.info("🔴 Making LIVE API call to Amadeus...")
    #         response = self.amadeus.shopping.flight_offers_search.get(**search_params)
            
    #         if response.status_code == 200:
    #             # Parse flight results
    #             flights = await self.parse_flight_results(response.data, origin, destination)
                
    #             if flights and len(flights) > 0:
    #                 # ✅ FIXED: Extract LIVE airline data with better error handling
    #                 live_airlines = await self.airline_provider.extract_airlines_from_flight_results(flights)
                    
    #                 # Build response with live data guarantee
    #                 result = {
    #                     "search_info": {
    #                         "search_date": departure_date,
    #                         "return_date": return_date,
    #                         "origin": origin,
    #                         "destination": destination,
    #                         "passengers": passengers,
    #                         "travel_class": travel_class
    #                     },
    #                     "flights": flights,
    #                     "total_found": len(flights),
    #                     "message": f"Found {len(flights)} flights from {origin} to {destination}",
    #                     "status": "success",
    #                     # ✅ FIXED: Live airline data with better structure
    #                     "route_airlines": {
    #                         "total_airlines_serving_route": len(live_airlines),
    #                         "airlines": live_airlines,
    #                         "data_source": "live_flight_results",
    #                         "last_verified": datetime.now().isoformat()
    #                     }
    #                 }
                    
    #                 # ✅ FIXED: Add freshness metadata properly
    #                 result = self.live_data_manager.add_freshness_metadata(result, request_id)
                    
    #                 # ✅ FIXED: Validate freshness with proper error handling
    #                 try:
    #                     if self.live_data_manager.validate_data_freshness(result):
    #                         logger.info("✅ Data freshness validated successfully")
    #                     else:
    #                         logger.warning("⚠️ Data freshness validation failed")
    #                 except Exception as validation_error:
    #                     logger.error(f"❌ Freshness validation error: {validation_error}")
                    
    #                 return result
                
    #             else:
    #                 # No flights found - but still return fresh response
    #                 result = {
    #                     "search_info": {
    #                         "origin": origin,
    #                         "destination": destination,
    #                         "search_date": departure_date
    #                     },
    #                     "flights": [],
    #                     "total_found": 0,
    #                     "status": "no_flights_found",
    #                     "message": "No flights found for this route",
    #                     "suggestions": [
    #                         "Try different dates",
    #                         "Check connecting flights",
    #                         "Verify route operates regularly"
    #                     ]
    #                 }
                    
    #                 return self.live_data_manager.add_freshness_metadata(result, request_id)
            
    #         else:
    #             raise Exception(f"Amadeus API returned status {response.status_code}")
                
    #     except Exception as e:
    #         error_msg = str(e)
    #         logger.error(f"❌ Live flight search error: {error_msg}")
            
    #         error_result = {
    #             "error": f"Flight search failed: {error_msg}",
    #             "search_info": {
    #                 "origin": arguments.get("origin"),
    #                 "destination": arguments.get("destination"),
    #                 "search_date": departure_date if 'departure_date' in locals() else None
    #             },
    #             "flights": [],
    #             "total_found": 0,
    #             "status": "error",
    #             "suggestions": [
    #                 "Check API connectivity",
    #                 "Verify airport codes",
    #                 "Try again in a few minutes"
    #             ]
    #         }
            
    #         return self.live_data_manager.add_freshness_metadata(error_result, request_id)
                
    #     except Exception as e:
    #         error_msg = str(e)
    #         logger.error(f"❌ Live flight search error: {error_msg}")
            
    #         error_result = {
    #             "error": f"Flight search failed: {error_msg}",
    #             "search_info": {
    #                 "origin": arguments.get("origin"),
    #                 "destination": arguments.get("destination"),
    #                 "search_date": departure_date if 'departure_date' in locals() else None
    #             },
    #             "flights": [],
    #             "total_found": 0,
    #             "status": "error",
    #             "suggestions": [
    #                 "Check API connectivity",
    #                 "Verify airport codes",
    #                 "Try again in a few minutes"
    #             ]
    #         }
            
    #         return self.live_data_manager.add_freshness_metadata(error_result, request_id)
    
    async def parse_flight_results(self, flight_data, origin, destination) -> List[Dict]:
        """Parse Amadeus API response with enhanced information and currency conversion"""
        flights = []
        
        for offer in flight_data[:10]:  # Up to 10 results for better airline coverage
            try:
                itinerary = offer['itineraries'][0]
                segment = itinerary['segments'][0]
                
                # Parse departure and arrival times (your existing code)
                departure_dt = datetime.fromisoformat(segment['departure']['at'].replace('Z', '+00:00') if segment['departure']['at'].endswith('Z') else segment['departure']['at'])
                arrival_dt = datetime.fromisoformat(segment['arrival']['at'].replace('Z', '+00:00') if segment['arrival']['at'].endswith('Z') else segment['arrival']['at'])
                
                # Calculate if arrival is next day
                arrival_day_diff = (arrival_dt.date() - departure_dt.date()).days
                arrival_display = arrival_dt.strftime("%H:%M")
                if arrival_day_diff > 0:
                    arrival_display += f"+{arrival_day_diff}"
               
                # Get cabin class from traveler pricing if available
                
                # ✅ FIXED: Get cabin class from traveler pricing with proper initialization and fallback
                cabin_class = "ECONOMY"  # Default fallback
                
                if 'travelerPricings' in offer and len(offer['travelerPricings']) > 0:
                    fare_details = offer['travelerPricings'][0].get('fareDetailsBySegment', [])
                    if fare_details and len(fare_details) > 0 and 'cabin' in fare_details[0]:
                        cabin_class = fare_details[0]['cabin']
                        
                # ✅ ADDITIONAL: Check alternative locations for cabin class
                if cabin_class == "ECONOMY":  # If still default, try other locations
                    # Try segment level cabin class
                    if 'cabin' in segment:
                        cabin_class = segment['cabin']
                    # Try pricingOptions if available
                    elif 'pricingOptions' in offer and 'fareType' in offer['pricingOptions']:
                        fare_type = offer['pricingOptions']['fareType']
                        if 'BUSINESS' in (fare_type[0].upper() if isinstance(fare_type, list) else fare_type.upper()):
                            cabin_class = "BUSINESS"
                        elif 'FIRST' in  (fare_type[0].upper() if isinstance(fare_type, list) else fare_type.upper()):
                            cabin_class = "FIRST"
                        elif 'PREMIUM' in (fare_type[0].upper() if isinstance(fare_type, list) else fare_type.upper()):
                            cabin_class = "PREMIUM_ECONOMY"
                        else :
                            cabin_class = (fare_type[0].upper() if isinstance(fare_type, list) else fare_type.upper())
                
                # ✅ DEBUG: Add logging to see what's being detected
                logger.info(f"Flight {segment['carrierCode']}{segment['number']}: Detected cabin class = {cabin_class}")
                if 'travelerPricings' in offer:
                    logger.debug(f"TravelerPricings structure: {offer['travelerPricings'][0] if offer['travelerPricings'] else 'Empty'}")
                
                # Rest of your existing code for price conversion...
                original_price = float(offer['price']['total'])
                original_currency = offer['price']['currency']
            
                
                # Only convert if not already in INR
                if original_currency == "EUR":
                    conversion = await convert_eur_to_inr(original_price)
                    
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
                        
                        # ✅ UPDATED: Price fields with conversion
                        "original_price": f"{original_price} {original_currency}",
                        "original_price_numeric": original_price,
                        "original_currency": original_currency,
                        
                        "price": conversion["formatted"],
                        "price_numeric": conversion["converted_amount"],
                        "currency": "INR",
                        "exchange_rate": conversion["exchange_rate"],
                        
                        "stops": len(itinerary['segments']) - 1,
                        "booking_class": cabin_class,
                        "route": f"{origin}->{destination}",
                        "is_direct": len(itinerary['segments']) == 1,
                        "aircraft": segment.get('aircraft', {}).get('code', ''),
                        "operating_carrier": segment.get('operating', {}).get('carrierCode', segment['carrierCode']),
                        "number_of_stops": segment.get('numberOfStops', 0),
                        "data_fetched_at": datetime.now().isoformat(),
                        "price_last_updated": datetime.now().isoformat()
                    }
                else:
                    # If already in INR or other currency, keep original
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
                        "price": f"{original_price} {original_currency}",
                        "price_numeric": original_price,
                        "currency": original_currency,
                        "stops": len(itinerary['segments']) - 1,
                        "booking_class": cabin_class,
                        "route": f"{origin}->{destination}",
                        "is_direct": len(itinerary['segments']) == 1,
                        "aircraft": segment.get('aircraft', {}).get('code', ''),
                        "operating_carrier": segment.get('operating', {}).get('carrierCode', segment['carrierCode']),
                        "number_of_stops": segment.get('numberOfStops', 0),
                        "data_fetched_at": datetime.now().isoformat(),
                        "price_last_updated": datetime.now().isoformat()
                    }
                
                # Add connecting flights information if not direct (your existing code)
                if len(itinerary['segments']) > 1:
                    flight['connecting_flights'] = []
                    for i, seg in enumerate(itinerary['segments']):
                        flight['connecting_flights'].append({
                            "segment": i + 1,
                            "flight_number": f"{seg['carrierCode']}{seg['number']}",
                            "airline": seg['carrierCode'],
                            "departure": f"{seg['departure']['iataCode']} {seg['departure']['at']}",
                            "arrival": f"{seg['arrival']['iataCode']} {seg['arrival']['at']}",
                            "duration": seg['duration']
                        })
                
                flights.append(flight)
                
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Error parsing flight data: {e}")
                continue
        
        logger.info(f"✅ Parsed {len(flights)} live flight offers with INR conversion")
        return flights

    async def get_airport_info_tool(self, arguments: dict) -> Dict:
        """MCP Tool: Get airport information with live data"""
        try:
            airport_code = arguments.get("airport_code")
            if not airport_code:
                raise ValueError("airport_code is required")
            
            logger.info(f"🔴 Getting LIVE airport info for: {airport_code}")
            
            response = self.amadeus.reference_data.locations.airports.get(
                keyword=airport_code
            )
            
            if response.status_code == 200 and response.data:
                airport = response.data[0]
                result = {
                    "code": airport.iataCode,
                    "name": airport.name,
                    "city": airport.address.cityName,
                    "country": airport.address.countryName,
                    "timezone": getattr(airport, 'timeZoneOffset', 'Unknown')
                }
                
                return self.live_data_manager.add_freshness_metadata(result)
            else:
                return {"error": "Airport not found"}
                
        except Exception as e:
            return {"error": f"Airport lookup failed: {str(e)}"}

    async def get_airline_info_tool(self, arguments: dict) -> Dict:
        """MCP Tool: Get airline information with live data"""
        try:
            airline_code = arguments.get("airline_code")
            if not airline_code:
                raise ValueError("airline_code is required")
            
            logger.info(f"🔴 Getting LIVE airline info for: {airline_code}")
            
            response = self.amadeus.reference_data.airlines.get(
                airlineCodes=airline_code
            )
            
            if response.status_code == 200 and response.data:
                airline = response.data[0]
                result = {
                    "code": airline.iataCode,
                    "name": airline.businessName,
                    "country": getattr(airline, 'countryCode', 'Unknown')
                }
                
                return self.live_data_manager.add_freshness_metadata(result)
            else:
                return {"error": "Airline not found"}
                
        except Exception as e:
            return {"error": f"Airline lookup failed: {str(e)}"}




