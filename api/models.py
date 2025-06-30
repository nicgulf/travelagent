
from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    spell_check: bool = Field(default=True, description="Enable spell checking")
    confidence_threshold: int = Field(default=70, ge=50, le=100, description="Spell check confidence threshold")
    force_live_data: bool = Field(default=True, description="Force live data (no cache)")

class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict

class SpellCheckRequest(BaseModel):
    text: str
    confidence_threshold: int = Field(default=70, ge=50, le=100)

class QueryResponse(BaseModel):
    status: str
    tool_used: str
    data: Any
    message: Optional[str] = None
    spell_check_info: Optional[Dict] = None
    data_freshness: Optional[Dict] = None

class SpellCheckResponse(BaseModel):
    original_text: str
    corrected_text: str
    corrections: List[Dict]
    total_corrections: int

class LocationResolveRequest(BaseModel):
    location: str
    auto_add_if_found: bool = True

class LocationResolveResponse(BaseModel):
    status: str
    airport_code: Optional[str] = None
    city_name: str
    suggestions: List[Dict] = []
    message: Optional[str] = None
    source: Optional[str] = None