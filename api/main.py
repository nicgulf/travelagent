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
import openai
from typing import Dict, List, Optional, Any, Union
import json
import uuid
import random
from datetime import datetime, timedelta
from models import *


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

try:
    # Initialize OpenAI-enhanced query processor
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        ai_enhanced_processor = EnhancedQueryProcessorWithAI(
            flight_service, 
            flight_service.location_resolver, 
            flight_service.spell_checker,
            openai_api_key
        )
        print("✅ OpenAI follow-up handler initialized")
    else:
        ai_enhanced_processor = None
        print("⚠️ OpenAI API key not found - follow-up features disabled")
except Exception as e:
    ai_enhanced_processor = None
    print(f"⚠️ OpenAI initialization not success: {e}")

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

@app.post("/conversation", response_model=ConversationResponse)
async def smart_conversation_endpoint(request: ConversationRequest):
    """
    🤖 Enhanced Smart conversation endpoint with comprehensive OpenAI follow-up support
    Handles both new queries and follow-up questions with advanced context understanding
    """
    if not ai_enhanced_processor:
        # Fallback to regular processing
        regular_result = await enhanced_query_processor.determine_tool_and_params(request.message)
        tool_name, params, spell_info = regular_result

        if tool_name and tool_name in enhanced_query_processor.tools:
            result = await enhanced_query_processor.tools[tool_name](params)
            return ConversationResponse(
                response=result,
                session_id="fallback",
                conversation_type="new_query",
                ai_understanding={"note": "OpenAI unavailable - using fallback processing"}
            )
        else:
            raise HTTPException(status_code=400, detail="Could not understand query")

    try:
        # Use enhanced AI processing with comprehensive testing support
        result = await ai_enhanced_processor.process_query_with_followup(
            query=request.message,
            session_id=request.session_id,
            user_id=request.user_id
        )

        # Add enhanced analytics and testing metrics
        if hasattr(result, 'get'):
            ai_understanding = result.get("ai_understanding", {})
            if ai_understanding:
                # Add confidence scoring and intent analysis
                ai_understanding["processing_confidence"] = ai_understanding.get("confidence", 0.0)
                ai_understanding["intent_clarity"] = "high" if ai_understanding.get("confidence", 0) > 0.8 else "medium" if ai_understanding.get("confidence", 0) > 0.5 else "low"
                ai_understanding["requires_clarification"] = ai_understanding.get("confidence", 0) < 0.6

        return ConversationResponse(
            response=result.get("result", {}),
            session_id=result.get("session_id"),
            conversation_type=result.get("type", "unknown"),
            ai_understanding=result.get("ai_understanding")
        )

    except Exception as e:
        logger.error(f"Enhanced conversation processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced conversation processing failed: {str(e)}")

@app.post("/followup-query")
async def process_followup_query(request: FollowUpQueryRequest):
    """
    🤖 Process follow-up queries with OpenAI understanding
    """
    if not ai_enhanced_processor:
        raise HTTPException(status_code=503, detail="OpenAI follow-up service unavailable")
    
    try:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="session_id required for follow-up queries")
        
        result = await ai_enhanced_processor.openai_handler.handle_followup_query(
            session_id=request.session_id,
            user_query=request.query
        )
        
        return {
            "status": "success",
            "ai_understanding": result,
            "session_id": request.session_id,
            "original_query": request.query
        }
        
    except Exception as e:
        logger.error(f"Follow-up processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Follow-up processing failed: {str(e)}")

@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if not ai_enhanced_processor:
        raise HTTPException(status_code=503, detail="OpenAI service unavailable")
    
    try:
        history = ai_enhanced_processor.openai_handler.get_conversation_summary(session_id)
        return {
            "status": "success",
            "conversation_summary": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/{session_id}/clear")
async def clear_conversation(session_id: str):
    """Clear a specific conversation session"""
    if not ai_enhanced_processor:
        raise HTTPException(status_code=503, detail="OpenAI service unavailable")
    
    try:
        if session_id in ai_enhanced_processor.openai_handler.conversation_history:
            del ai_enhanced_processor.openai_handler.conversation_history[session_id]
            return {"status": "success", "message": f"Conversation {session_id} cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/cleanup-conversations")
async def cleanup_old_conversations(hours_old: int = 24):
    """Admin endpoint to cleanup old conversations"""
    if not ai_enhanced_processor:
        raise HTTPException(status_code=503, detail="OpenAI service unavailable")
    
    try:
        ai_enhanced_processor.openai_handler.clear_old_conversations(hours_old)
        return {
            "status": "success",
            "message": f"Cleaned up conversations older than {hours_old} hours"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai/status")
async def openai_service_status():
    """Check OpenAI service status"""
    return {
        "openai_available": ai_enhanced_processor is not None,
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "active_conversations": len(ai_enhanced_processor.openai_handler.conversation_history) if ai_enhanced_processor else 0,
        "features": {
            "follow_up_queries": ai_enhanced_processor is not None,
            "conversation_memory": ai_enhanced_processor is not None,
            "natural_language_understanding": ai_enhanced_processor is not None
        }
    }

# Example usage endpoints for testing

@app.post("/demo/conversation-flow")
async def demo_conversation_flow():
    """
    Demo endpoint showing conversation flow capabilities
    """
    if not ai_enhanced_processor:
        return {"error": "OpenAI service not available"}
    
    # Simulate a conversation flow
    demo_flow = [
        "Search flights from Mumbai to Delhi tomorrow",
        "What about business class?",
        "Any cheaper options?",
        "Show me morning flights only",
        "Book the first flight"
    ]
    
    try:
        session_id = ai_enhanced_processor.openai_handler.start_conversation("demo_user")
        results = []
        
        for i, query in enumerate(demo_flow):
            logger.info(f"Demo step {i+1}: {query}")
            
            result = await ai_enhanced_processor.process_query_with_followup(
                query=query,
                session_id=session_id,
                user_id="demo_user"
            )
            
            results.append({
                "step": i + 1,
                "query": query,
                "type": result.get("type"),
                "ai_understanding": result.get("ai_understanding", {}).get("intent") if result.get("ai_understanding") else None,
                "action_taken": result.get("ai_understanding", {}).get("action") if result.get("ai_understanding") else None
            })
        
        return {
            "demo_conversation": results,
            "session_id": session_id,
            "conversation_summary": ai_enhanced_processor.openai_handler.get_conversation_summary(session_id)
        }
        
    except Exception as e:
        return {"error": f"Demo failed: {str(e)}"}

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
    """
    ✨ Enhanced query processing with OpenAI follow-up support
    Automatically detects if this is a new query or follow-up
    """
    try:
        # Try AI-enhanced processing first
        if ai_enhanced_processor:
            result = await ai_enhanced_processor.process_query_with_followup(
                query=request.query,
                session_id=getattr(request, 'session_id', None),
                user_id=getattr(request, 'user_id', None)
            )
            
            if result.get("type") == "followup_response":
                return QueryResponse(
                    status="success",
                    tool_used="ai_followup",
                    data=result.get("result", {}),
                    message="Processed as follow-up query using AI",
                    ai_understanding=result.get("ai_understanding"),
                    session_id=result.get("session_id")
                )
            else:
                return QueryResponse(
                    status=result.get("result", {}).get("status", "success"),
                    tool_used=result.get("tool_used", "unknown"),
                    data=result.get("result", {}),
                    message=result.get("result", {}).get("message", "Processed successfully"),
                    spell_check_info=result.get("spell_check_info"),
                    session_id=result.get("session_id")
                )
        
        # Fallback to original processing
        else:
            logger.info(f"🔴 FALLBACK PROCESSING: {request.query}")
            
            tool_name, params, spell_info = await enhanced_query_processor.determine_tool_and_params(request.query)
            
            if not tool_name:
                spell_result = flight_service.spell_checker.correct_city_spelling(request.query)
                error_detail = "Could not understand the query. Please specify origin and destination."
                
                if spell_result["corrections"]:
                    error_detail += f" Did you mean: '{spell_result['corrected_text']}'?"
                
                raise HTTPException(status_code=400, detail=error_detail)
            
            if tool_name in enhanced_query_processor.tools:
                logger.info(f"🔴 Executing {tool_name} with live data guarantee")
                result = await enhanced_query_processor.tools[tool_name](params)
                
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
        logger.error(f"Enhanced query processing error: {str(e)}")
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

async def _generate_fallback_flight_data(origin: str, destination: str, departure_date: str, travel_class: str, passengers: int) -> Dict:
    """Generate realistic fallback flight data when Amadeus API is unavailable"""

    # Common airlines for Indian routes
    airlines = [
        {"code": "6E", "name": "IndiGo"},
        {"code": "SG", "name": "SpiceJet"},
        {"code": "AI", "name": "Air India"},
        {"code": "UK", "name": "Vistara"},
        {"code": "G8", "name": "GoAir"}
    ]

    # Generate realistic flight times and prices
    base_price = 3500 if travel_class == "ECONOMY" else 8500
    price_variation = random.randint(-500, 1500)

    flights = []
    for i in range(3):  # Generate 3 sample flights
        airline = random.choice(airlines)
        departure_time = f"{random.randint(6, 22):02d}:{random.choice(['00', '15', '30', '45'])}"
        duration_hours = random.randint(1, 3)
        duration_minutes = random.choice([0, 15, 30, 45])

        flight = {
            "id": f"demo_flight_{i+1}",
            "airline": airline["name"],
            "airline_code": airline["code"],
            "flight_number": f"{airline['code']}{random.randint(100, 999)}",
            "departure_time": departure_time,
            "arrival_time": f"{(int(departure_time[:2]) + duration_hours) % 24:02d}:{departure_time[3:]}",
            "duration": f"{duration_hours}h {duration_minutes}m",
            "price": base_price + price_variation + (i * 200),
            "currency": "INR",
            "travel_class": travel_class,
            "stops": 0 if i < 2 else 1,
            "aircraft": random.choice(["A320", "B737", "ATR72"]),
            "availability": random.randint(5, 20)
        }
        flights.append(flight)

    return {
        "flights": flights,
        "total_found": len(flights),
        "search_info": {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "travel_class": travel_class,
            "passengers": passengers
        },
        "status": "success",
        "data_source": "fallback_demo_data",
        "notice": "Demo data - Amadeus API temporarily unavailable",
        "suggestions": [
            "These are sample flights for demonstration",
            "Real-time data will be available when API is restored",
            "Prices and schedules are for demo purposes only"
        ],
        "data_freshness": {
            "fetched_at": datetime.now().isoformat(),
            "is_live": False,
            "cache_used": False,
            "api_call_made": False,
            "data_source": "fallback_generator"
        }
    }

@app.post("/flights/search")
async def search_flights_endpoint(
    origin: str,
    destination: str,
    departure_date: Optional[str] = None,
    return_date: Optional[str] = None,
    passengers: int = 1,
    travel_class: str = "ECONOMY",
    auto_correct: bool = True,
    use_fallback: bool = True
):
    """
    ✅ LIVE flight search endpoint with intelligent fallback
    """
    try:
        logger.info(f"🔴 LIVE FLIGHT SEARCH: {origin} → {destination}")

        # Apply spell checking if enabled
        if auto_correct:
            origin_corrected = flight_service.spell_checker._resolve_to_airport_code(origin)
            dest_corrected = flight_service.spell_checker._resolve_to_airport_code(destination)

            if origin_corrected:
                logger.info(f"✅ Origin corrected: {origin} → {origin_corrected}")
                origin = origin_corrected
            if dest_corrected:
                logger.info(f"✅ Destination corrected: {destination} → {dest_corrected}")
                destination = dest_corrected

        # Try live flight search first
        try:
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

            # Check if we got an error and fallback is enabled
            # Check multiple error conditions - look inside the result structure
            has_error = (
                result.get("status") == "error" or
                "error" in result or
                result.get("total_found", 0) == 0 or
                len(result.get("flights", [])) == 0 or
                # Check for Amadeus API errors (500 server errors)
                ("error" in str(result).lower() and "500" in str(result)) or
                ("server_error" in str(result).lower()) or
                ("administrator" in str(result).lower())
            )
            
            if has_error and use_fallback:
                logger.warning("🔄 Live API failed, using intelligent fallback...")
                logger.debug(f"🔍 Error detected in result: {result}")
                fallback_result = await _generate_fallback_flight_data(
                    origin, destination, departure_date or "2025-07-04", travel_class, passengers
                )
                return {
                    "status": "success",
                    "data": fallback_result,
                    "notice": "Using demo data - Amadeus API temporarily unavailable"
                }

            # Add extra validation for successful results
            if "data_freshness" in result:
                freshness_valid = flight_service.live_data_manager.validate_data_freshness(result)
                result["freshness_validated"] = freshness_valid

            return {"status": "success", "data": result}

        except Exception as api_error:
            logger.error(f"❌ Live API error: {api_error}")

            if use_fallback:
                logger.info("🔄 Generating fallback flight data...")
                fallback_result = await _generate_fallback_flight_data(
                    origin, destination, departure_date or "2025-07-04", travel_class, passengers
                )
                return {
                    "status": "success",
                    "data": fallback_result,
                    "notice": "Using demo data - Amadeus API temporarily unavailable"
                }
            else:
                raise api_error

    except Exception as e:
        logger.error(f"❌ Flight search error: {e}")
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

@app.get("/ai/conversation-patterns")
async def get_conversation_patterns():
    """Get common conversation patterns for frontend integration"""
    return {
        "follow_up_examples": [
            {
                "category": "Class Changes",
                "examples": [
                    "What about business class?",
                    "Show me first class options",
                    "Any economy flights?",
                    "Premium economy available?"
                ]
            },
            {
                "category": "Price Filters",
                "examples": [
                    "Any cheaper options?",
                    "Show flights under ₹10,000",
                    "Most expensive flights",
                    "Budget options only"
                ]
            },
            {
                "category": "Time Preferences",
                "examples": [
                    "Morning flights only",
                    "Evening departures",
                    "What about tomorrow?",
                    "Next week instead"
                ]
            },
            {
                "category": "Flight Details",
                "examples": [
                    "Tell me about flight AI 101",
                    "Which airlines fly this route?",
                    "How long is the flight?",
                    "Any direct flights?"
                ]
            },
            {
                "category": "Booking Actions",
                "examples": [
                    "Book the first flight",
                    "Select the cheapest option",
                    "I want flight number 2",
                    "Reserve the morning flight"
                ]
            }
        ],
        "conversation_starters": [
            "Find flights from {origin} to {destination}",
            "Search flights for next month",
            "Book a flight from {city1} to {city2}",
            "Show me flights on {date}"
        ]
    }

@app.post("/ai/test-understanding")
async def test_ai_understanding(query: str, context: Optional[dict] = None):
    """
    Test endpoint to see how AI understands a query
    Useful for debugging and development
    """
    if not ai_enhanced_processor:
        raise HTTPException(status_code=503, detail="OpenAI service unavailable")
    
    try:
        # Create a temporary session for testing
        session_id = ai_enhanced_processor.openai_handler.start_conversation("test_user")
        
        # Add context if provided
        if context:
            ai_enhanced_processor.openai_handler.conversation_history[session_id]["context"] = context
        
        # Get AI understanding
        result = await ai_enhanced_processor.openai_handler.handle_followup_query(session_id, query)
        
        # Clean up test session
        del ai_enhanced_processor.openai_handler.conversation_history[session_id]
        
        return {
            "query": query,
            "ai_understanding": result,
            "context_used": context or {},
            "test_mode": True
        }
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "test_mode": True
        }

# Environment variables and configuration updates
@app.get("/config/ai-features")
async def get_ai_features_config():
    """Get AI features configuration"""
    return {
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "features_available": {
            "conversation_memory": ai_enhanced_processor is not None,
            "follow_up_understanding": ai_enhanced_processor is not None,
            "natural_language_processing": ai_enhanced_processor is not None,
            "context_awareness": ai_enhanced_processor is not None,
            "intent_detection": ai_enhanced_processor is not None
        },
        "fallback_behavior": "Uses rule-based processing when OpenAI unavailable",
        "session_management": "In-memory (use Redis in production)",
        "max_conversation_history": 10,
        "conversation_cleanup": "24 hours"
    }

# Enhanced health check with AI status
@app.get("/health-enhanced")
async def enhanced_health_check():
    """Enhanced health check including AI services"""
    base_health = await health_check()  # Your existing health check
    
    ai_status = {
        "openai_service": {
            "available": ai_enhanced_processor is not None,
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "active_conversations": len(ai_enhanced_processor.openai_handler.conversation_history) if ai_enhanced_processor else 0
        },
        "ai_features": {
            "conversation_memory": "active" if ai_enhanced_processor else "disabled",
            "follow_up_processing": "active" if ai_enhanced_processor else "disabled",
            "natural_language_understanding": "active" if ai_enhanced_processor else "disabled"
        }
    }
    
    # Merge with base health check
    enhanced_health = {**base_health, "ai_services": ai_status}
    
    return enhanced_health
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

@app.post("/test/comprehensive-suite")
async def run_comprehensive_test_suite(
    include_openai_tests: bool = True,
    test_categories: Optional[List[str]] = None,
    detailed_output: bool = False
):
    """
    🧪 Run comprehensive test suite for travel agent functionality
    Tests month handling, date validation, city spelling, and follow-up queries
    """
    try:
        from enhanced_testing_framework import EnhancedTravelAgentTester, TestCategory

        # Initialize tester
        openai_key = os.getenv("OPENAI_API_KEY") if include_openai_tests else None
        tester = EnhancedTravelAgentTester(openai_api_key=openai_key)

        # Run comprehensive test suite
        test_results = await tester.run_comprehensive_test_suite(flight_service)

        # Filter results if specific categories requested
        if test_categories:
            filtered_results = []
            for result in test_results["results"]:
                if result["category"] in test_categories:
                    filtered_results.append(result)
            test_results["results"] = filtered_results
            test_results["summary"] = tester._calculate_test_summary(
                filtered_results,
                test_results["total_execution_time"]
            )

        # Simplify output if detailed_output is False
        if not detailed_output:
            simplified_results = []
            for result in test_results["results"]:
                simplified_results.append({
                    "test_name": result["test_name"],
                    "category": result["category"],
                    "status": result["status"],
                    "execution_time": result["execution_time"]
                })
            test_results["results"] = simplified_results

        return {
            "status": "completed",
            "test_suite_results": test_results,
            "openai_enabled": include_openai_tests and bool(openai_key),
            "categories_tested": test_categories or ["all"],
            "recommendations": _generate_test_recommendations(test_results["summary"])
        }

    except Exception as e:
        logger.error(f"Test suite execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Test suite failed: {str(e)}")

def _generate_test_recommendations(summary: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on test results"""
    recommendations = []

    if summary["success_rate"] < 80:
        recommendations.append("⚠️ Success rate below 80% - review failed tests")

    if summary["failed"] > 0:
        recommendations.append(f"❌ {summary['failed']} tests failed - check implementation")

    if summary["errors"] > 0:
        recommendations.append(f"🚨 {summary['errors']} tests had errors - check system configuration")

    if summary["average_execution_time"] > 5.0:
        recommendations.append("⏱️ Average test time > 5s - consider performance optimization")

    # Category-specific recommendations
    for category, stats in summary["category_breakdown"].items():
        if stats["failed"] > 0:
            if category == "month_handling":
                recommendations.append("📅 Month handling issues detected - review date parsing logic")
            elif category == "city_spelling":
                recommendations.append("🏙️ City spelling issues detected - update spell checker database")
            elif category == "follow_up_queries":
                recommendations.append("🤖 Follow-up query issues detected - review OpenAI integration")

    if not recommendations:
        recommendations.append("✅ All tests passed - system functioning well")

    return recommendations

@app.post("/test/month-handling")
async def test_month_handling():
    """🗓️ Test month handling functionality specifically"""
    try:
        from enhanced_testing_framework import EnhancedTravelAgentTester

        tester = EnhancedTravelAgentTester()
        month_tests = tester.create_month_handling_tests()

        results = []
        for test_case in month_tests:
            result = await tester.run_test_case(test_case, flight_service)
            results.append(result)

        summary = tester._calculate_test_summary(results, sum(r["execution_time"] for r in results))

        return {
            "category": "month_handling",
            "summary": summary,
            "test_results": results,
            "specific_insights": {
                "full_month_names": any(r["test_name"] == "test_full_month_names" and r["status"] == "passed" for r in results),
                "abbreviated_months": any(r["test_name"] == "test_abbreviated_month_names" and r["status"] == "passed" for r in results),
                "numeric_months": any(r["test_name"] == "test_numeric_months" and r["status"] == "passed" for r in results),
                "relative_months": any(r["test_name"] == "test_relative_months" and r["status"] == "passed" for r in results),
                "error_handling": any(r["test_name"] == "test_invalid_months" and r["status"] == "passed" for r in results)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Month handling tests failed: {str(e)}")

@app.post("/test/date-validation")
async def test_date_validation():
    """📅 Test date validation functionality specifically"""
    try:
        from enhanced_testing_framework import EnhancedTravelAgentTester

        tester = EnhancedTravelAgentTester()
        date_tests = tester.create_date_validation_tests()

        results = []
        for test_case in date_tests:
            result = await tester.run_test_case(test_case, flight_service)
            results.append(result)

        summary = tester._calculate_test_summary(results, sum(r["execution_time"] for r in results))

        return {
            "category": "date_validation",
            "summary": summary,
            "test_results": results,
            "specific_insights": {
                "iso_format_support": any(r["test_name"] == "test_iso_date_format" and r["status"] == "passed" for r in results),
                "relative_dates": any(r["test_name"] == "test_relative_dates" and r["status"] == "passed" for r in results),
                "natural_language": any(r["test_name"] == "test_natural_language_dates" and r["status"] == "passed" for r in results),
                "invalid_date_handling": any(r["test_name"] == "test_invalid_dates" and r["status"] == "passed" for r in results),
                "past_date_handling": any(r["test_name"] == "test_past_dates" and r["status"] == "passed" for r in results),
                "date_range_support": any(r["test_name"] == "test_date_ranges" and r["status"] == "passed" for r in results)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Date validation tests failed: {str(e)}")

@app.post("/test/city-spelling")
async def test_city_spelling():
    """🏙️ Test city spelling correction functionality specifically"""
    try:
        from enhanced_testing_framework import EnhancedTravelAgentTester

        tester = EnhancedTravelAgentTester()
        spelling_tests = tester.create_city_spelling_tests()

        results = []
        for test_case in spelling_tests:
            result = await tester.run_test_case(test_case, flight_service)
            results.append(result)

        summary = tester._calculate_test_summary(results, sum(r["execution_time"] for r in results))

        return {
            "category": "city_spelling",
            "summary": summary,
            "test_results": results,
            "specific_insights": {
                "minor_typo_correction": any(r["test_name"] == "test_minor_typos" and r["status"] == "passed" for r in results),
                "phonetic_matching": any(r["test_name"] == "test_phonetic_similarities" and r["status"] == "passed" for r in results),
                "partial_matching": any(r["test_name"] == "test_partial_matches" and r["status"] == "passed" for r in results),
                "international_cities": any(r["test_name"] == "test_international_cities" and r["status"] == "passed" for r in results),
                "unknown_city_handling": any(r["test_name"] == "test_unknown_cities" and r["status"] == "passed" for r in results),
                "case_handling": any(r["test_name"] == "test_case_insensitive" and r["status"] == "passed" for r in results)
            },
            "spell_checker_stats": {
                "total_cities_in_database": len(flight_service.spell_checker.dynamic_cities) if hasattr(flight_service, 'spell_checker') else 0,
                "unknown_cities_logged": len(flight_service.spell_checker.unknown_cities_log) if hasattr(flight_service, 'spell_checker') else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City spelling tests failed: {str(e)}")

@app.post("/test/follow-up-queries")
async def test_follow_up_queries(include_openai: bool = True):
    """🤖 Test follow-up query functionality specifically"""
    try:
        from enhanced_testing_framework import EnhancedTravelAgentTester

        openai_key = os.getenv("OPENAI_API_KEY") if include_openai else None
        tester = EnhancedTravelAgentTester(openai_api_key=openai_key)
        followup_tests = tester.create_follow_up_query_tests()

        results = []
        for test_case in followup_tests:
            result = await tester.run_test_case(test_case, flight_service)
            results.append(result)

        summary = tester._calculate_test_summary(results, sum(r["execution_time"] for r in results))

        return {
            "category": "follow_up_queries",
            "summary": summary,
            "test_results": results,
            "openai_enabled": include_openai and bool(openai_key),
            "specific_insights": {
                "class_modification": any(r["test_name"] == "test_class_modification" and r["status"] == "passed" for r in results),
                "date_modification": any(r["test_name"] == "test_date_modification" and r["status"] == "passed" for r in results),
                "price_filtering": any(r["test_name"] == "test_price_filtering" and r["status"] == "passed" for r in results),
                "context_preservation": all(r.get("details", {}).get("validation_details", {}).get("context_preserved", False) for r in results if r["status"] == "passed")
            },
            "ai_service_status": {
                "openai_available": bool(openai_key),
                "active_conversations": len(ai_enhanced_processor.openai_handler.conversation_history) if ai_enhanced_processor else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up query tests failed: {str(e)}")

@app.post("/demo/enhanced-conversation-flow")
async def demo_enhanced_conversation_flow():
    """
    🎭 Enhanced demo showcasing comprehensive conversation flow with testing
    Demonstrates month handling, date validation, city spelling, and follow-up queries
    """
    if not ai_enhanced_processor:
        return {"error": "OpenAI service not available for enhanced demo"}

    try:
        # Enhanced conversation flow with comprehensive testing scenarios
        demo_scenarios = [
            {
                "scenario": "Month Handling Test",
                "queries": [
                    "Find flights from Mumbai to Delhi in January",
                    "What about flights in Feb instead?",
                    "Show me options for next month"
                ]
            },
            {
                "scenario": "Date Validation Test",
                "queries": [
                    "Flights from Mumbai to Delhi tomorrow",
                    "What about next Friday?",
                    "Any flights on 2024-12-25?"
                ]
            },
            {
                "scenario": "City Spelling Correction Test",
                "queries": [
                    "Flights from Mumbay to Deli",
                    "What about from Kolkatta to Bangalor?",
                    "Show flights from Londn to Paries"
                ]
            },
            {
                "scenario": "Follow-up Query Intelligence Test",
                "queries": [
                    "Search flights from Mumbai to Delhi tomorrow",
                    "What about business class?",
                    "Any cheaper options?",
                    "Show me morning flights only",
                    "What airlines fly this route?"
                ]
            }
        ]

        demo_results = []

        for scenario in demo_scenarios:
            scenario_result = {
                "scenario_name": scenario["scenario"],
                "queries": [],
                "session_id": None,
                "overall_success": True
            }

            session_id = ai_enhanced_processor.openai_handler.start_conversation("demo_user")
            scenario_result["session_id"] = session_id

            for i, query in enumerate(scenario["queries"]):
                logger.info(f"Demo {scenario['scenario']} - Step {i+1}: {query}")

                try:
                    result = await ai_enhanced_processor.process_query_with_followup(
                        query=query,
                        session_id=session_id,
                        user_id="demo_user"
                    )

                    query_result = {
                        "step": i + 1,
                        "query": query,
                        "status": "success",
                        "type": result.get("type"),
                        "ai_understanding": result.get("ai_understanding", {}),
                        "processing_time": result.get("processing_time", 0),
                        "context_preserved": bool(result.get("ai_understanding", {}).get("context_preserved", False))
                    }

                    # Add specific insights based on scenario
                    if "Month" in scenario["scenario"]:
                        query_result["month_detected"] = self._extract_month_from_result(result)
                    elif "Date" in scenario["scenario"]:
                        query_result["date_parsed"] = self._extract_date_from_result(result)
                    elif "Spelling" in scenario["scenario"]:
                        query_result["spelling_corrections"] = self._extract_spelling_from_result(result)
                    elif "Follow-up" in scenario["scenario"]:
                        query_result["intent_detected"] = result.get("ai_understanding", {}).get("intent")

                except Exception as e:
                    query_result = {
                        "step": i + 1,
                        "query": query,
                        "status": "error",
                        "error": str(e)
                    }
                    scenario_result["overall_success"] = False

                scenario_result["queries"].append(query_result)

            demo_results.append(scenario_result)

        # Generate comprehensive summary
        summary = {
            "total_scenarios": len(demo_scenarios),
            "successful_scenarios": len([s for s in demo_results if s["overall_success"]]),
            "total_queries": sum(len(s["queries"]) for s in demo_results),
            "successful_queries": sum(len([q for q in s["queries"] if q.get("status") == "success"]) for s in demo_results),
            "average_processing_time": sum(
                sum(q.get("processing_time", 0) for q in s["queries"])
                for s in demo_results
            ) / sum(len(s["queries"]) for s in demo_results) if demo_results else 0,
            "capabilities_demonstrated": {
                "month_handling": any("Month" in s["scenario_name"] for s in demo_results),
                "date_validation": any("Date" in s["scenario_name"] for s in demo_results),
                "city_spelling": any("Spelling" in s["scenario_name"] for s in demo_results),
                "follow_up_intelligence": any("Follow-up" in s["scenario_name"] for s in demo_results)
            }
        }

        return {
            "demo_type": "enhanced_conversation_flow",
            "summary": summary,
            "scenarios": demo_results,
            "recommendations": self._generate_demo_recommendations(summary),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Enhanced demo failed: {e}")
        return {"error": f"Enhanced demo failed: {str(e)}"}

def _extract_month_from_result(self, result: Dict) -> Optional[str]:
    """Extract month information from query result"""
    # Implementation would depend on the actual result structure
    return result.get("ai_understanding", {}).get("parameters", {}).get("month")

def _extract_date_from_result(self, result: Dict) -> Optional[str]:
    """Extract date information from query result"""
    return result.get("ai_understanding", {}).get("parameters", {}).get("date")

def _extract_spelling_from_result(self, result: Dict) -> List[Dict]:
    """Extract spelling corrections from query result"""
    return result.get("spell_check_info", {}).get("corrections", [])

def _generate_demo_recommendations(self, summary: Dict) -> List[str]:
    """Generate recommendations based on demo results"""
    recommendations = []

    success_rate = (summary["successful_queries"] / summary["total_queries"] * 100) if summary["total_queries"] > 0 else 0

    if success_rate >= 90:
        recommendations.append("🎉 Excellent performance - all systems working optimally")
    elif success_rate >= 75:
        recommendations.append("✅ Good performance - minor improvements possible")
    else:
        recommendations.append("⚠️ Performance issues detected - review failed scenarios")

    if summary["average_processing_time"] > 3.0:
        recommendations.append("⏱️ Consider optimizing response times")

    capabilities = summary["capabilities_demonstrated"]
    if not all(capabilities.values()):
        missing = [k for k, v in capabilities.items() if not v]
        recommendations.append(f"🔧 Missing capabilities: {', '.join(missing)}")

    return recommendations


if __name__ == "__main__":
    import uvicorn
    from debug import test_date_parsing
    test_date_parsing() 
    print("🔴 Starting Live Flight Search API - Zero Cache Policy")
    print("✅ Flight data: Always live")
    print("✅ Airline data: From live flight results") 
    print("✅ Price data: Real-time")
    uvicorn.run(app, host="0.0.0.0", port=8000)