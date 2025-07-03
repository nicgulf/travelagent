from pydantic import BaseModel, Field, validator
from typing import Optional, Any, List, Dict, Union, Literal
from datetime import datetime
import re

# Enhanced request models
class EnhancedQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language flight search query", min_length=1, max_length=500)
    user_id: Optional[str] = Field(None, description="User identifier for conversation tracking", max_length=100)
    session_id: Optional[str] = Field(None, description="Session ID for follow-up queries", max_length=100)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    force_live_data: bool = Field(default=True, description="Force live data retrieval")
    include_analytics: bool = Field(default=False, description="Include performance analytics in response")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('User ID can only contain letters, numbers, hyphens, and underscores')
        return v

class FollowUpQueryRequest(BaseModel):
    query: str = Field(..., description="Follow-up query text", min_length=1, max_length=300)
    session_id: str = Field(..., description="Active session ID", min_length=1)
    user_id: Optional[str] = Field(None, description="User identifier")
    context_override: Optional[Dict[str, Any]] = Field(None, description="Override conversation context")

class ConversationRequest(BaseModel):
    message: str = Field(..., description="Conversation message", min_length=1, max_length=500)
    session_id: Optional[str] = Field(None, description="Session ID for continuing conversation")
    user_id: str = Field(..., description="User identifier", min_length=1, max_length=100)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)

# Enhanced response models
class PerformanceMetrics(BaseModel):
    api_response_time: float = Field(..., description="API response time in seconds")
    ai_processing_time: Optional[float] = Field(None, description="AI processing time in seconds")
    database_query_time: Optional[float] = Field(None, description="Database query time in seconds")
    cache_hit: Optional[bool] = Field(None, description="Whether cache was hit")
    timeout_used: Optional[bool] = Field(None, description="Whether timeout was triggered")

class AIUnderstanding(BaseModel):
    intent: str = Field(..., description="Detected user intent")
    action: str = Field(..., description="Recommended action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    user_message: str = Field(..., description="Clarified user intent")
    requires_context: Optional[List[str]] = Field(None, description="Required context items")
    fallback_suggestions: Optional[List[str]] = Field(None, description="Alternative interpretations")
    extraction_method: Optional[str] = Field(None, description="Method used for extraction")

class EnhancedQueryResponse(BaseModel):
    status: Literal["success", "error", "partial"] = Field(..., description="Response status")
    type: Literal["new_query", "followup", "error", "clarification_needed"] = Field(..., description="Query type")
    session_id: str = Field(..., description="Session identifier")
    data: Dict[str, Any] = Field(..., description="Response data")
    ai_understanding: Optional[AIUnderstanding] = Field(None, description="AI understanding details")
    processing_time: float = Field(..., description="Total processing time in seconds")
    spell_check_info: Optional[Dict[str, Any]] = Field(None, description="Spell check information")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for user")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")

class ConversationResponse(BaseModel):
    response: Dict[str, Any] = Field(..., description="Conversation response data")
    session_id: str = Field(..., description="Session identifier")
    conversation_type: Literal["new_query", "followup", "error"] = Field(..., description="Conversation type")
    ai_understanding: Optional[AIUnderstanding] = Field(None, description="AI understanding")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance data")

# Location and search models
class LocationResolveRequest(BaseModel):
    location: str = Field(..., description="Location to resolve", min_length=2, max_length=100)
    auto_add_if_found: bool = Field(default=True, description="Auto-add to dynamic database if found")
    prefer_airports: bool = Field(default=True, description="Prefer airports over cities")

class LocationSuggestion(BaseModel):
    type: Literal["fuzzy_match", "dynamic_match", "partial_match", "pattern_correction"] = Field(..., description="Suggestion type")
    suggestion: str = Field(..., description="Suggested location name")
    airport_code: str = Field(..., description="Airport code")
    country: str = Field(..., description="Country")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence score")
    reason: str = Field(..., description="Reason for suggestion")

class LocationResolveResponse(BaseModel):
    status: Literal["success", "unknown", "error"] = Field(..., description="Resolution status")
    airport_code: Optional[str] = Field(None, description="Resolved airport code")
    city_name: str = Field(..., description="City name")
    suggestions: List[LocationSuggestion] = Field(default_factory=list, description="Alternative suggestions")
    message: Optional[str] = Field(None, description="Status message")
    source: Optional[str] = Field(None, description="Resolution source")
    cache_hit: Optional[bool] = Field(None, description="Whether result was cached")
    processing_time: Optional[float] = Field(None, description="Processing time")

# Flight search models
class FlightSearchRequest(BaseModel):
    origin: str = Field(..., description="Origin airport/city", min_length=2, max_length=50)
    destination: str = Field(..., description="Destination airport/city", min_length=2, max_length=50)
    departure_date: Optional[str] = Field(None, description="Departure date (YYYY-MM-DD or natural language)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD or natural language)")
    passengers: int = Field(default=1, ge=1, le=9, description="Number of passengers")
    travel_class: Literal["ECONOMY", "BUSINESS", "FIRST", "PREMIUM_ECONOMY"] = Field(default="ECONOMY", description="Travel class")
    auto_correct: bool = Field(default=True, description="Enable spell correction")
    include_filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class FlightResult(BaseModel):
    airline: str = Field(..., description="Airline code")
    flight_number: str = Field(..., description="Flight number")
    departure_date: str = Field(..., description="Departure date")
    departure_time: str = Field(..., description="Departure time")
    arrival_date: str = Field(..., description="Arrival date")
    arrival_time: str = Field(..., description="Arrival time")
    departure_airport: str = Field(..., description="Departure airport code")
    arrival_airport: str = Field(..., description="Arrival airport code")
    duration: str = Field(..., description="Flight duration")
    price: str = Field(..., description="Formatted price")
    price_numeric: float = Field(..., description="Numeric price")
    currency: str = Field(..., description="Currency code")
    stops: int = Field(..., description="Number of stops")
    booking_class: str = Field(..., description="Booking class")
    is_direct: bool = Field(..., description="Whether flight is direct")

class FlightSearchResponse(BaseModel):
    search_info: Dict[str, Any] = Field(..., description="Search metadata")
    flights: List[FlightResult] = Field(..., description="Flight results")
    total_found: int = Field(..., description="Total flights found")
    message: str = Field(..., description="Search result message")
    status: Literal["success", "no_flights_found", "error"] = Field(..., description="Search status")
    route_airlines: Optional[Dict[str, Any]] = Field(None, description="Airlines serving route")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance data")

# Spell checking models
class SpellCheckRequest(BaseModel):
    text: str = Field(..., description="Text to spell check", min_length=1, max_length=500)
    confidence_threshold: int = Field(default=70, ge=50, le=100, description="Confidence threshold for corrections")
    context: Optional[str] = Field(None, description="Context for better correction")

class SpellCorrection(BaseModel):
    original: str = Field(..., description="Original word")
    corrected: str = Field(..., description="Corrected word")
    confidence: float = Field(..., description="Correction confidence")
    country: str = Field(..., description="Country of corrected location")
    airport_code: str = Field(..., description="Airport code")
    alternatives: List[str] = Field(default_factory=list, description="Alternative names")

class SpellCheckResponse(BaseModel):
    original_text: str = Field(..., description="Original text")
    corrected_text: str = Field(..., description="Corrected text")
    corrections: List[SpellCorrection] = Field(..., description="List of corrections made")
    total_corrections: int = Field(..., description="Number of corrections")
    confidence_scores: Optional[List[float]] = Field(None, description="Confidence scores for corrections")

# Analytics models
class ConversationAnalytics(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    message_count: int = Field(..., description="Number of messages")
    has_search_context: bool = Field(..., description="Whether session has search context")
    session_duration_minutes: Optional[float] = Field(None, description="Session duration in minutes")
    query_frequency: Optional[float] = Field(None, description="Queries per minute")
    followup_ratio: Optional[float] = Field(None, description="Ratio of follow-up queries")

class SearchAnalytics(BaseModel):
    route: str = Field(..., description="Search route")
    flight_count: int = Field(..., description="Number of flights found")
    processing_time: float = Field(..., description="Processing time")
    success: bool = Field(..., description="Whether search was successful")
    timestamp: datetime = Field(..., description="Search timestamp")

class SystemMetrics(BaseModel):
    timestamp: datetime = Field(..., description="Metrics timestamp")
    total_conversations: int = Field(..., description="Total conversations")
    active_conversations: int = Field(..., description="Active conversations")
    total_searches: int = Field(..., description="Total searches")
    average_response_time: float = Field(..., description="Average response time")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate percentage")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")

# Tool and admin models
class ToolRequest(BaseModel):
    tool_name: Literal["search_flights", "get_airport_info", "get_airline_info"] = Field(..., description="Tool to execute")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    timeout: Optional[float] = Field(default=30.0, description="Execution timeout")

class AdminCleanupRequest(BaseModel):
    hours_old: int = Field(default=24, ge=1, le=168, description="Hours old for cleanup")
    dry_run: bool = Field(default=False, description="Perform dry run without actual cleanup")

class HealthCheckResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    uptime: Optional[str] = Field(None, description="System uptime")
    performance: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")

# Configuration models
class ServiceConfiguration(BaseModel):
    features: Dict[str, bool] = Field(..., description="Available features")
    limits: Dict[str, Union[int, float]] = Field(..., description="Service limits")
    performance: Dict[str, bool] = Field(..., description="Performance features")
    storage: Dict[str, str] = Field(..., description="Storage configuration")

# Error models
class ErrorDetail(BaseModel):
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Trace identifier")

class ValidationErrorResponse(BaseModel):
    status: Literal["validation_error"] = Field(default="validation_error")
    errors: List[ErrorDetail] = Field(..., description="Validation errors")
    message: str = Field(..., description="Overall error message")

# Conversation context models
class ConversationContext(BaseModel):
    last_origin: Optional[str] = Field(None, description="Last search origin")
    last_destination: Optional[str] = Field(None, description="Last search destination")
    last_date: Optional[str] = Field(None, description="Last search date")
    last_travel_class: Optional[str] = Field(None, description="Last travel class")
    available_airlines: List[str] = Field(default_factory=list, description="Available airlines")
    price_range: Optional[Dict[str, float]] = Field(None, description="Price range")
    flight_count: int = Field(default=0, description="Number of flights found")
    search_type: Optional[str] = Field(None, description="Type of search performed")

class ConversationSummary(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Total messages")
    has_search_context: bool = Field(..., description="Has search context")
    last_route: str = Field(..., description="Last searched route")
    context: ConversationContext = Field(..., description="Conversation context")
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list, description="Recent messages")
    analytics: Optional[Dict[str, Any]] = Field(None, description="Session analytics")

# AI-specific models
class AIIntent(BaseModel):
    intent_type: Literal["modify_search", "get_details", "book_flight", "compare_options", "new_search", "get_info", "clarification_needed"] = Field(..., description="Intent type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent parameters")
    context_required: List[str] = Field(default_factory=list, description="Required context")

class AIProcessingResult(BaseModel):
    intent: AIIntent = Field(..., description="Detected intent")
    action: str = Field(..., description="Recommended action")
    processing_time: float = Field(..., description="AI processing time")
    model_used: str = Field(..., description="AI model used")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")

# Filter models for advanced search
class FlightFilters(BaseModel):
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price filter")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price filter")
    direct_only: Optional[bool] = Field(None, description="Direct flights only")
    preferred_time: Optional[Literal["morning", "afternoon", "evening"]] = Field(None, description="Preferred departure time")
    airline_preference: Optional[str] = Field(None, description="Preferred airline code")
    max_duration_hours: Optional[float] = Field(None, ge=0, le=48, description="Maximum flight duration in hours")
    sort_by: Optional[Literal["price_low_to_high", "price_high_to_low", "departure_time", "duration"]] = Field(None, description="Sort criteria")
    exclude_airlines: Optional[List[str]] = Field(None, description="Airlines to exclude")
    preferred_airports: Optional[Dict[str, List[str]]] = Field(None, description="Preferred airports for origin/destination")

class FilteredSearchRequest(FlightSearchRequest):
    filters: Optional[FlightFilters] = Field(None, description="Advanced filters")
    sort_preferences: Optional[List[str]] = Field(None, description="Sort preference order")

# Enhanced date models
class DateInfo(BaseModel):
    type: Literal["single_date", "month_range", "default"] = Field(..., description="Date type")
    date: Optional[str] = Field(None, description="Single date (YYYY-MM-DD)")
    start_date: Optional[str] = Field(None, description="Range start date")
    end_date: Optional[str] = Field(None, description="Range end date")
    month_name: Optional[str] = Field(None, description="Month name for range searches")
    search_dates: List[str] = Field(default_factory=list, description="Dates to search")
    confidence: Optional[float] = Field(None, description="Date parsing confidence")

class EnhancedFlightSearchRequest(BaseModel):
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    departure_date: Union[str, DateInfo, Dict[str, Any]] = Field(..., description="Departure date information")
    return_date: Optional[Union[str, DateInfo]] = Field(None, description="Return date information")
    passengers: int = Field(default=1, ge=1, le=9)
    travel_class: str = Field(default="ECONOMY")
    filters: Optional[FlightFilters] = Field(None)
    preferences: Optional[Dict[str, Any]] = Field(None)

# Airline and airport models
class AirlineInfo(BaseModel):
    iata_code: str = Field(..., description="IATA airline code")
    name: str = Field(..., description="Airline name")
    country: str = Field(..., description="Airline country")
    type: Literal["Full-service", "Low-cost", "Regional", "Cargo", "Unknown"] = Field(..., description="Airline type")
    currently_operating: bool = Field(..., description="Currently operating status")
    last_verified: str = Field(..., description="Last verification timestamp")
    data_source: str = Field(..., description="Data source")

class AirportInfo(BaseModel):
    code: str = Field(..., description="Airport IATA code")
    name: str = Field(..., description="Airport name")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    timezone: Optional[str] = Field(None, description="Airport timezone")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Airport coordinates")

class RouteAirlines(BaseModel):
    total_airlines_serving_route: int = Field(..., description="Number of airlines serving route")
    airlines: List[AirlineInfo] = Field(..., description="Airlines serving the route")
    data_source: str = Field(..., description="Data source")
    last_verified: str = Field(..., description="Last verification timestamp")

# Performance and monitoring models
class APIMetrics(BaseModel):
    total_requests: int = Field(..., description="Total API requests")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: float = Field(..., description="Average response time")
    requests_per_minute: float = Field(..., description="Requests per minute")
    error_rate: float = Field(..., description="Error rate percentage")

class DatabaseMetrics(BaseModel):
    total_queries: int = Field(..., description="Total database queries")
    average_query_time: float = Field(..., description="Average query time")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")

class SystemHealth(BaseModel):
    overall_status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system status")
    api_metrics: APIMetrics = Field(..., description="API performance metrics")
    database_metrics: Optional[DatabaseMetrics] = Field(None, description="Database metrics")
    ai_service_status: str = Field(..., description="AI service status")
    redis_status: str = Field(..., description="Redis status")
    amadeus_api_status: str = Field(..., description="Amadeus API status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

# Batch processing models
class BatchSearchRequest(BaseModel):
    searches: List[FlightSearchRequest] = Field(..., max_items=10, description="Batch of search requests")
    parallel_execution: bool = Field(default=True, description="Execute searches in parallel")
    timeout_per_search: float = Field(default=30.0, description="Timeout per individual search")

class BatchSearchResult(BaseModel):
    request_id: str = Field(..., description="Original request identifier")
    status: Literal["success", "error", "timeout"] = Field(..., description="Search status")
    data: Optional[FlightSearchResponse] = Field(None, description="Search results")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Processing time for this search")

class BatchSearchResponse(BaseModel):
    batch_id: str = Field(..., description="Batch identifier")
    total_searches: int = Field(..., description="Total number of searches")
    successful_searches: int = Field(..., description="Number of successful searches")
    failed_searches: int = Field(..., description="Number of failed searches")
    results: List[BatchSearchResult] = Field(..., description="Individual search results")
    total_processing_time: float = Field(..., description="Total batch processing time")

# User preference models
class UserPreferences(BaseModel):
    preferred_airlines: Optional[List[str]] = Field(None, description="Preferred airline codes")
    excluded_airlines: Optional[List[str]] = Field(None, description="Excluded airline codes")
    preferred_travel_class: Optional[str] = Field(None, description="Default travel class")
    price_range: Optional[Dict[str, float]] = Field(None, description="Preferred price range")
    time_preferences: Optional[Dict[str, str]] = Field(None, description="Time preferences")
    notification_settings: Optional[Dict[str, bool]] = Field(None, description="Notification preferences")

class UserProfile(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: UserPreferences = Field(..., description="User preferences")
    search_history: Optional[List[str]] = Field(None, description="Recent search routes")
    favorite_routes: Optional[List[str]] = Field(None, description="Favorite routes")
    created_at: datetime = Field(..., description="Profile creation date")
    last_updated: datetime = Field(..., description="Last update timestamp")

# WebSocket models for real-time updates
class WebSocketMessage(BaseModel):
    type: Literal["search_update", "price_alert", "system_status", "error"] = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")

# Export models for data analysis
class SearchExportData(BaseModel):
    searches: List[SearchAnalytics] = Field(..., description="Search data")
    conversations: List[ConversationAnalytics] = Field(..., description="Conversation data")
    export_timestamp: datetime = Field(..., description="Export timestamp")
    date_range: Dict[str, str] = Field(..., description="Date range for export")
    total_records: int = Field(..., description="Total number of records")

# Rate limiting models
class RateLimitInfo(BaseModel):
    requests_remaining: int = Field(..., description="Requests remaining in current window")
    reset_time: datetime = Field(..., description="When the rate limit resets")
    limit_per_window: int = Field(..., description="Request limit per window")
    window_duration_seconds: int = Field(..., description="Window duration in seconds")

# Legacy compatibility models (for backward compatibility)
class QueryRequest(BaseModel):
    """Legacy query request model for backward compatibility"""
    query: str
    user_id: Optional[str] = None
    spell_check: bool = Field(default=True)
    confidence_threshold: int = Field(default=70, ge=50, le=100)
    force_live_data: bool = Field(default=True)

class QueryResponse(BaseModel):
    """Legacy query response model for backward compatibility"""
    status: str
    tool_used: str
    data: Any
    message: Optional[str] = None
    spell_check_info: Optional[Dict] = None
    data_freshness: Optional[Dict] = None

# Custom validators and utility functions
def validate_airport_code(code: str) -> str:
    """Validate airport code format"""
    if not code or len(code) != 3 or not code.isalpha():
        raise ValueError("Airport code must be exactly 3 letters")
    return code.upper()

def validate_airline_code(code: str) -> str:
    """Validate airline code format"""
    if not code or len(code) != 2 or not code.isalpha():
        raise ValueError("Airline code must be exactly 2 letters")
    return code.upper()

def validate_date_format(date_str: str) -> str:
    """Validate date format"""
    import re
    from datetime import datetime
    
    # Allow various date formats
    patterns = [
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
        r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
    ]
    
    if not any(re.match(pattern, date_str) for pattern in patterns):
        raise ValueError("Invalid date format. Use YYYY-MM-DD, MM/DD/YYYY, or MM-DD-YYYY")
    
    return date_str

# Configuration for model serialization
class Config:
    """Pydantic configuration for all models"""
    json_encoders = {
        datetime: lambda v: v.isoformat(),
    }
    validate_assignment = True
    use_enum_values = True
    arbitrary_types_allowed = True