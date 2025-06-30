from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import httpx
import asyncio
import re
import os
from fuzzywuzzy import fuzz, process
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utilites import *
from debug import *
# Initialize FastAPI app
app = FastAPI(title="Enhanced Flight Search API with Spell Checking", version="2.0.0")

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
enhanced_query_processor = EnhancedQueryProcessor(
    flight_service, 
    flight_service.location_resolver, 
    flight_service.spell_checker
)


# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Live Flight Search API - Zero Cache Policy", 
        "version": "3.0.0",
        "features": [
            "🔴 100% Live flight data (no cache)",
            "✈️ Live airline data from flight results",
            "🔧 Intelligent spell checking for city names",
            "🎯 Real-time price and availability",
            "📊 Data freshness validation"
        ],
        "data_policy": {
            "flight_prices": "NEVER cached - always live",
            "airline_data": "Extracted from live flight results",
            "airport_coordinates": "Static data (cacheable)",
            "freshness_guarantee": "All flight data < 5 minutes old"
        }
    }


@app.post("/resolve-location", response_model=LocationResolveResponse)
async def resolve_location_endpoint(request: LocationResolveRequest):
    """
    Resolve any location to airport code with intelligent suggestions for unknown cities
    """
    try:
        result = await flight_service.location_resolver.resolve_location_to_airport(request.location)
        
        return LocationResolveResponse(
            status=result["status"],
            airport_code=result.get("airport_code"),
            city_name=result.get("city_name", request.location),
            suggestions=result.get("suggestions", []),
            message=result.get("message"),
            source=result.get("source")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unknown-cities-log")
async def get_unknown_cities_log():
    """Get log of unknown cities for analysis"""
    return {
        "unknown_cities": flight_service.spell_checker.unknown_cities_log,
        "total_logged": len(flight_service.spell_checker.unknown_cities_log),
        "dynamic_cities_count": len(flight_service.spell_checker.dynamic_cities)
    }

@app.post("/add-city-manually")
async def add_city_manually(city_name: str, airport_code: str, country: str = "Unknown"):
    """Manually add a city to the dynamic database"""
    try:
        flight_service.spell_checker.add_dynamic_city(city_name, airport_code, country)
        return {
            "status": "success",
            "message": f"Added {city_name} → {airport_code}",
            "total_dynamic_cities": len(flight_service.spell_checker.dynamic_cities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/tools")
async def list_available_tools():
    """List all available MCP tools"""
    return {
        "tools": [
            {
                "name": "search_flights",
                "description": "Search for flights with spell checking and smart location resolution",
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

@app.post("/spell-check", response_model=SpellCheckResponse)
async def spell_check_text(request: SpellCheckRequest):
    """Spell check city names in text"""
    try:
        result = flight_service.spell_checker.correct_city_spelling(
            request.text, 
            request.confidence_threshold
        )
        
        return SpellCheckResponse(
            original_text=result["original_text"],
            corrected_text=result["corrected_text"],
            corrections=result["corrections"],
            total_corrections=result["total_corrections"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_natural_query(request: QueryRequest):
    """Fixed query processing with better error handling"""
    try:
        logger.info(f"🔴 LIVE QUERY PROCESSING: {request.query}")
        
        # Determine which tool to use with spell checking
        tool_name, params, spell_info = await enhanced_query_processor.determine_tool_and_params(request.query)
        
        if not tool_name:
            # If no tool determined, try to provide helpful suggestions
            spell_result = flight_service.spell_checker.correct_city_spelling(request.query)
            error_detail = "Could not understand the query. Please specify origin and destination."
            
            if spell_result["corrections"]:
                error_detail += f" Did you mean: '{spell_result['corrected_text']}'?"
            
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Execute the tool with live data guarantee
        if tool_name in enhanced_query_processor.tools:
            logger.info(f"🔴 Executing {tool_name} with live data guarantee")
            result = await enhanced_query_processor.tools[tool_name](params)
            
            # ✅ FIX: Debug the response structure
            debug_response_structure(result, "Query Response")
            
            response = QueryResponse(
                status=result.get("status", "success"),
                tool_used=tool_name,
                data=result,
                message=result.get("message", f"Processed query using {tool_name}"),
                spell_check_info=spell_info,
                data_freshness=result.get("data_freshness")
            )
            
            return response
        else:
            raise HTTPException(status_code=400, detail=f"Tool {tool_name} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tool", response_model=QueryResponse)
async def execute_tool_directly(request: ToolRequest):
    """Execute a specific tool with given parameters"""
    try:
        if request.tool_name not in enhanced_query_processor.tools:
            raise HTTPException(status_code=400, detail=f"Tool {request.tool_name} not found")
        
        result = await enhanced_query_processor.tools[request.tool_name](request.arguments)
        
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
    travel_class: str = "ECONOMY",
    auto_correct: bool = True
):
    """
    ✅ LIVE flight search endpoint - guaranteed fresh data
    """
    try:
        logger.info(f"🔴 LIVE FLIGHT SEARCH: {origin} → {destination}")
        
        # Apply spell checking if enabled
        if auto_correct:
            origin_corrected = flight_service.spell_checker._resolve_to_airport_code(origin)
            dest_corrected = flight_service.spell_checker._resolve_to_airport_code(destination)
            
            if origin_corrected:
                origin = origin_corrected
            if dest_corrected:
                destination = dest_corrected
        
        params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "passengers": passengers,
            "travel_class": travel_class
        }
        
        # ✅ ALWAYS get live data
        result = await flight_service.search_flights_tool(params)
        
        # Add extra validation
        if "data_freshness" in result:
            freshness_valid = flight_service.live_data_manager.validate_data_freshness(result)
            result["freshness_validated"] = freshness_valid
        
        return {"status": "success", "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/airport/{airport_code}")
async def get_airport_info_endpoint(airport_code: str):
    """Get airport information with spell checking"""
    try:
        # Try to resolve airport code if it's not a valid 3-letter code
        if len(airport_code) != 3:
            resolved_code = flight_service.spell_checker._resolve_to_airport_code(airport_code)
            if resolved_code:
                airport_code = resolved_code
        
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

@app.get("/cities")
async def get_supported_cities():
    """Get list of supported cities for spell checking"""
    return {
        "cities": CITIES_DATA,
        "total_cities": len(CITIES_DATA),
        "countries": list(set(city["country"] for city in CITIES_DATA))
    }

@app.get("/debug/airline-api-test/{airline_code}")
async def test_airline_api(airline_code: str):
    """Test endpoint to debug airline API responses"""
    try:
        logger.info(f"🧪 Testing airline API for: {airline_code}")
        
        response = flight_service.amadeus.reference_data.airlines.get(
            airlineCodes=airline_code
        )
        
        return {
            "status_code": response.status_code,
            "has_data": hasattr(response, 'data'),
            "data_length": len(response.data) if hasattr(response, 'data') and response.data else 0,
            "data_type": type(response.data[0]).__name__ if hasattr(response, 'data') and response.data else "None",
            "raw_response": str(response.data[0]) if hasattr(response, 'data') and response.data else "No data",
            "airline_code": airline_code
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "airline_code": airline_code
        }

@app.get("/debug/freshness-test")
async def test_freshness_validation():
    """Test endpoint to debug freshness validation"""
    
    # Create test response
    test_response = {
        "test": "data",
        "data_freshness": {
            "fetched_at": datetime.now().isoformat(),
            "is_live": True
        }
    }
    
    manager = LiveDataManager()
    is_fresh = manager.validate_data_freshness(test_response)
    
    return {
        "test_response": test_response,
        "validation_result": is_fresh,
        "current_time": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with data freshness info"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "spell_checker": "active",
            "amadeus_api": "configured" if flight_service.amadeus_api_key else "missing",
            "live_data_manager": "active"
        },
        "data_policy": {
            "cache_policy": "Zero cache for flight data",
            "freshness_guarantee": "< 5 minutes",
            "static_data_only": "Airport coordinates, city names"
        },
        "version": "3.0.0"
    }
@app.post("/flights/search-enhanced")
async def enhanced_search_flights_endpoint(
    origin: str,
    destination: str,
    departure_date: Optional[str] = None,
    return_date: Optional[str] = None,
    passengers: int = 1,
    travel_class: str = "ECONOMY",
    auto_correct: bool = True,
    include_debug: bool = False
):
    """
    Enhanced flight search with comprehensive debugging and error handling
    """
    try:
        # Initialize enhanced handler
        handler = EnhancedFlightSearchHandler(flight_service)
        
        # Prepare parameters
        params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "passengers": passengers,
            "travel_class": travel_class
        }
        
        # Apply spell correction if enabled
        if auto_correct:
            origin_corrected = flight_service.spell_checker._resolve_to_airport_code(origin)
            dest_corrected = flight_service.spell_checker._resolve_to_airport_code(destination)
            
            if origin_corrected:
                params["origin"] = origin_corrected
            if dest_corrected:
                params["destination"] = dest_corrected
        
        # Perform enhanced search
        result = await handler.enhanced_search_flights(params)
        
        # Include debug info only if requested
        if not include_debug and "debug_info" in result:
            del result["debug_info"]
        
        # Add route-specific help for COK-GAU
        if (params["origin"] == "COK" and params["destination"] == "GAU") or \
           (params["origin"] == "GAU" and params["destination"] == "COK"):
            result["route_specific_help"] = COKGAURouteFixer.get_alternative_routes()
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/airlines/live/{origin}/{destination}")
async def get_live_airlines_for_route(origin: str, destination: str):
    """
    Get LIVE airlines serving a specific route (not from static database)
    """
    try:
        logger.info(f"🔴 Getting LIVE airlines for route: {origin} → {destination}")
        
        # Make a sample flight search to get current airlines
        search_params = {
            "origin": origin,
            "destination": destination,
            "departure_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "passengers": 1
        }
        
        # Get flight results
        flight_results = await flight_service.search_flights_tool(search_params)
        
        if flight_results.get("route_airlines"):
            return {
                "status": "success",
                "route": f"{origin} → {destination}",
                "airlines": flight_results["route_airlines"],
                "note": "Airlines extracted from live flight search results",
                "data_source": "live_flight_api",
                "fetched_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "no_data",
                "route": f"{origin} → {destination}",
                "message": "No airlines found for this route",
                "suggestions": ["Route may not exist", "Try connecting flights", "Check airport codes"]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Testing endpoint for debugging
@app.post("/debug/route")
async def debug_route_endpoint(origin: str, destination: str, date: Optional[str] = None):
    """Debug specific route issues"""
    try:
        debugger = FlightAPIDebugger(flight_service)
        debug_result = await debugger.debug_route_search(origin, destination, date)
        return debug_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-freshness/status")
async def get_data_freshness_status():
    """Get current data freshness status"""
    return {
        "policy": {
            "flight_data": "NEVER cached - always live",
            "price_data": "Real-time from API",
            "airline_data": "Extracted from live flight results",
            "airport_coordinates": "Static (cacheable for 30 days)",
            "maximum_age": "5 minutes"
        },
        "validation": {
            "timestamp_required": True,
            "freshness_check": "Automatic",
            "stale_data_threshold": "300 seconds"
        },
        "current_status": "All flight data guaranteed live",
        "last_policy_update": "2024-12-01T00:00:00Z"
    }
@app.get("/debug/cache-status")
async def debug_cache_status():
    """Debug endpoint to show what's cached vs live"""
    return {
        "never_cached": [
            "Flight prices and availability",
            "Airline route data", 
            "Seat availability",
            "Current flight schedules",
            "Real-time offers"
        ],
        "cacheable_static": [
            "Airport coordinates (30 days TTL)",
            "Airport names (7 days TTL)",
            "City name mappings (7 days TTL)"
        ],
        "current_cache_items": len(flight_service.location_resolver._location_cache),
        "cache_policy": "Static geographical data only",
        "last_cache_clear": "Not applicable - flight data never cached"
    }


if __name__ == "__main__":
    import uvicorn
    from debug import test_date_parsing
    test_date_parsing() 
    print("🔴 Starting Live Flight Search API - Zero Cache Policy")
    print("✅ Flight data: Always live")
    print("✅ Airline data: From live flight results") 
    print("✅ Price data: Real-time")
    uvicorn.run(app, host="0.0.0.0", port=8000)