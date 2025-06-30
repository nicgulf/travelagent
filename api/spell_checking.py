from fastapi import FastAPI, HTTPException, Query
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

app = FastAPI(
    title="City Name Spelling Correction API",
    description="API for correcting misspelled city names using fuzzy matching and AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CorrectionRequest(BaseModel):
    text: str = Field(..., description="Text containing city names to correct")
    method: Literal["local", "openai", "both"] = Field(
        default="both", 
        description="Correction method to use"
    )
    confidence_threshold: int = Field(
        default=60,
        ge=1,
        le=100,
        description="Minimum confidence score for corrections (1-100)"
    )
    include_unknown: bool = Field(
        default=False,
        description="Include analysis of unknown cities not in database"
    )

class AddCityRequest(BaseModel):
    name: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    alternatives: List[str] = Field(default=[], description="Alternative names for the city")

class Correction(BaseModel):
    original: str
    corrected: str
    confidence: int
    country: str
    alternatives: List[str] = []
    method: str

class UnknownCity(BaseModel):
    word: str
    reason: str
    suggestions: List[str] = []
    confidence_scores: List[int] = []

class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    corrections: List[Correction]
    unknown_cities: List[UnknownCity] = []
    total_corrections: int
    total_unknown: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str

# City database
CITIES_DATA = [
    {"name": "Mumbai", "country": "India", "alternatives": ["Bombay"]},
    {"name": "Delhi", "country": "India", "alternatives": ["New Delhi"]},
    {"name": "Bangalore", "country": "India", "alternatives": ["Bengaluru"]},
    {"name": "Chennai", "country": "India", "alternatives": ["Madras"]},
    {"name": "Kolkata", "country": "India", "alternatives": ["Calcutta"]},
    {"name": "Hyderabad", "country": "India", "alternatives": []},
    {"name": "Pune", "country": "India", "alternatives": ["Poona"]},
    {"name": "Ahmedabad", "country": "India", "alternatives": []},
    {"name": "Jaipur", "country": "India", "alternatives": []},
    {"name": "Lucknow", "country": "India", "alternatives": []},
    {"name": "Surat", "country": "India", "alternatives": []},
    {"name": "Kanpur", "country": "India", "alternatives": []},
    {"name": "Nagpur", "country": "India", "alternatives": []},
    {"name": "Indore", "country": "India", "alternatives": []},
    {"name": "Thane", "country": "India", "alternatives": []},
    {"name": "Bhopal", "country": "India", "alternatives": []},
    {"name": "Visakhapatnam", "country": "India", "alternatives": ["Vizag"]},
    {"name": "Pimpri-Chinchwad", "country": "India", "alternatives": []},
    {"name": "Patna", "country": "India", "alternatives": []},
    {"name": "Vadodara", "country": "India", "alternatives": ["Baroda"]},
    {"name": "New York", "country": "USA", "alternatives": ["NYC", "New York City"]},
    {"name": "Los Angeles", "country": "USA", "alternatives": ["LA"]},
    {"name": "Chicago", "country": "USA", "alternatives": []},
    {"name": "Houston", "country": "USA", "alternatives": []},
    {"name": "Phoenix", "country": "USA", "alternatives": []},
    {"name": "Philadelphia", "country": "USA", "alternatives": []},
    {"name": "San Antonio", "country": "USA", "alternatives": []},
    {"name": "San Diego", "country": "USA", "alternatives": []},
    {"name": "Dallas", "country": "USA", "alternatives": []},
    {"name": "San Jose", "country": "USA", "alternatives": []},
    {"name": "London", "country": "UK", "alternatives": []},
    {"name": "Birmingham", "country": "UK", "alternatives": []},
    {"name": "Manchester", "country": "UK", "alternatives": []},
    {"name": "Liverpool", "country": "UK", "alternatives": []},
    {"name": "Leeds", "country": "UK", "alternatives": []},
    {"name": "Paris", "country": "France", "alternatives": []},
    {"name": "Marseille", "country": "France", "alternatives": []},
    {"name": "Lyon", "country": "France", "alternatives": []},
    {"name": "Toulouse", "country": "France", "alternatives": []},
    {"name": "Nice", "country": "France", "alternatives": []},
    {"name": "Tokyo", "country": "Japan", "alternatives": []},
    {"name": "Osaka", "country": "Japan", "alternatives": []},
    {"name": "Kyoto", "country": "Japan", "alternatives": []},
    {"name": "Yokohama", "country": "Japan", "alternatives": []},
    {"name": "Nagoya", "country": "Japan", "alternatives": []},
    {"name": "Berlin", "country": "Germany", "alternatives": []},
    {"name": "Munich", "country": "Germany", "alternatives": ["München"]},
    {"name": "Hamburg", "country": "Germany", "alternatives": []},
    {"name": "Cologne", "country": "Germany", "alternatives": ["Köln"]},
    {"name": "Frankfurt", "country": "Germany", "alternatives": []},
]

# Create a flattened list for fuzzy matching
CITY_NAMES = []
CITY_MAP = {}

for city in CITIES_DATA:
    CITY_NAMES.append(city["name"])
    CITY_MAP[city["name"].lower()] = city
    
    for alt in city["alternatives"]:
        CITY_NAMES.append(alt)
        CITY_MAP[alt.lower()] = city

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def analyze_unknown_cities(text: str, confidence_threshold: int = 60) -> List[UnknownCity]:
    """Analyze words that might be cities not in our database."""
    words = re.findall(r'\b[A-Za-z]+\b', text)
    unknown_cities = []
    
    # Common non-city words to filter out
    EXCLUDE_WORDS = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
        'its', 'our', 'their', 'myself', 'yourself', 'himself', 'herself', 'itself',
        'ourselves', 'yourselves', 'themselves', 'what', 'which', 'who', 'whom', 'whose',
        'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'then', 'once'
    }
    
    for word in words:
        if len(word) < 3 or word.lower() in EXCLUDE_WORDS:
            continue
            
        # Check if word might be a city (starts with capital, looks like a proper noun)
        if not word[0].isupper():
            continue
            
        # Get fuzzy matches
        matches = process.extract(word, CITY_NAMES, limit=3, scorer=fuzz.ratio)
        
        if matches and matches[0][1] < confidence_threshold:
            # Word doesn't match well with known cities
            suggestions = [match[0] for match in matches[:3]]
            confidence_scores = [match[1] for match in matches[:3]]
            
            unknown_cities.append(UnknownCity(
                word=word,
                reason=f"Not found in database. Best match: {matches[0][0]} ({matches[0][1]}% similarity)",
                suggestions=suggestions,
                confidence_scores=confidence_scores
            ))
        elif not matches:
            # No matches at all
            unknown_cities.append(UnknownCity(
                word=word,
                reason="No similar cities found in database",
                suggestions=[],
                confidence_scores=[]
            ))
    
    return unknown_cities

def correct_spelling_local(text: str, confidence_threshold: int = 60) -> List[Correction]:
    """Correct spelling using local fuzzy matching."""
    words = re.findall(r'\b[A-Za-z]+\b', text)
    corrections = []
    
    for word in words:
        if len(word) < 3:
            continue
            
        # Use fuzzy matching to find best matches
        matches = process.extract(word, CITY_NAMES, limit=3, scorer=fuzz.ratio)
        
        if matches and matches[0][1] > confidence_threshold:  # Use configurable threshold
            best_match = matches[0][0]
            confidence = matches[0][1]
            
            # Skip if it's already correct
            if word.lower() == best_match.lower():
                continue
                
            city_info = CITY_MAP.get(best_match.lower())
            if city_info:
                corrections.append(Correction(
                    original=word,
                    corrected=best_match,
                    confidence=confidence,
                    country=city_info["country"],
                    alternatives=city_info["alternatives"][:2],  # Limit alternatives
                    method="fuzzy_match"
                ))
    
    return corrections

async def correct_spelling_openai(text: str) -> List[Correction]:
    """Correct spelling using OpenAI API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    prompt = f"""Please check and correct any misspelled city names in the following text: "{text}"

Consider common misspellings and provide corrections with confidence scores. Return only a JSON object with this format:
{{
  "corrections": [
    {{
      "original": "misspelled_word",
      "corrected": "correct_spelling",
      "confidence": 95,
      "country": "country_name",
      "alternatives": ["alternative1", "alternative2"]
    }}
  ]
}}

Focus only on city names, not other words. If no corrections are needed, return an empty corrections array."""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a city name spelling correction assistant. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=500, detail="OpenAI API request failed")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            ai_result = json.loads(content)
            corrections = []
            
            for correction_data in ai_result.get("corrections", []):
                corrections.append(Correction(
                    original=correction_data["original"],
                    corrected=correction_data["corrected"],
                    confidence=correction_data["confidence"],
                    country=correction_data.get("country", "Unknown"),
                    alternatives=correction_data.get("alternatives", []),
                    method="openai"
                ))
            
            return corrections
            
    except httpx.TimeoutException:
        logger.error("OpenAI API timeout")
        raise HTTPException(status_code=500, detail="OpenAI API timeout")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        logger.error(f"OpenAI correction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI processing failed: {str(e)}")

def merge_corrections(corrections: List[Correction]) -> List[Correction]:
    """Merge and deduplicate corrections, keeping the highest confidence."""
    correction_map = {}
    
    for correction in corrections:
        key = correction.original.lower()
        if key not in correction_map or correction.confidence > correction_map[key].confidence:
            correction_map[key] = correction
    
    return list(correction_map.values())

def apply_corrections(text: str, corrections: List[Correction]) -> str:
    """Apply corrections to the original text."""
    corrected_text = text
    
    # Sort by length (descending) to handle longer words first
    sorted_corrections = sorted(corrections, key=lambda x: len(x.original), reverse=True)
    
    for correction in sorted_corrections:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(correction.original) + r'\b'
        corrected_text = re.sub(pattern, correction.corrected, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

@app.post("/correct-city-names", response_model=CorrectionResponse)
async def correct_city_names(request: CorrectionRequest):
    """
    Correct misspelled city names in the provided text.
    
    - **text**: The text containing city names to correct
    - **method**: Correction method - 'local', 'openai', or 'both'
    - **confidence_threshold**: Minimum confidence score for corrections (1-100)
    - **include_unknown**: Whether to analyze unknown cities not in database
    """
    try:
        corrections = []
        unknown_cities = []
        
        # Local correction
        if request.method in ["local", "both"]:
            local_corrections = correct_spelling_local(request.text, request.confidence_threshold)
            corrections.extend(local_corrections)
        
        # OpenAI correction
        if request.method in ["openai", "both"]:
            try:
                ai_corrections = await correct_spelling_openai(request.text)
                corrections.extend(ai_corrections)
            except Exception as e:
                logger.warning(f"OpenAI correction failed: {str(e)}")
                if request.method == "openai":
                    raise HTTPException(status_code=500, detail=f"OpenAI correction failed: {str(e)}")
        
        # Analyze unknown cities if requested
        if request.include_unknown:
            unknown_cities = analyze_unknown_cities(request.text, request.confidence_threshold)
        
        # Merge and deduplicate corrections
        unique_corrections = merge_corrections(corrections)
        
        # Apply corrections to text
        corrected_text = apply_corrections(request.text, unique_corrections)
        
        return CorrectionResponse(
            original=request.text,
            corrected=corrected_text,
            corrections=unique_corrections,
            unknown_cities=unknown_cities,
            total_corrections=len(unique_corrections),
            total_unknown=len(unknown_cities)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="OK",
        timestamp=datetime.now().isoformat()
    )

@app.post("/add-city")
async def add_city(request: AddCityRequest):
    """
    Add a new city to the database dynamically.
    
    - **name**: The primary name of the city
    - **country**: The country where the city is located
    - **alternatives**: List of alternative names (optional)
    """
    try:
        # Check if city already exists
        if request.name.lower() in CITY_MAP:
            raise HTTPException(status_code=400, detail=f"City '{request.name}' already exists in database")
        
        # Add to database
        new_city = {
            "name": request.name,
            "country": request.country,
            "alternatives": request.alternatives
        }
        
        CITIES_DATA.append(new_city)
        CITY_NAMES.append(request.name)
        CITY_MAP[request.name.lower()] = new_city
        
        # Add alternatives to maps
        for alt in request.alternatives:
            CITY_NAMES.append(alt)
            CITY_MAP[alt.lower()] = new_city
        
        return {
            "message": f"City '{request.name}' added successfully",
            "city": new_city,
            "total_cities": len(CITIES_DATA)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding city: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add city")

@app.get("/search-cities")
async def search_cities(
    query: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of results")
):
    """
    Search for cities in the database.
    
    - **query**: The search term
    - **limit**: Maximum number of results to return (1-50)
    """
    try:
        matches = process.extract(query, CITY_NAMES, limit=limit, scorer=fuzz.ratio)
        
        results = []
        for match_name, score in matches:
            city_info = CITY_MAP.get(match_name.lower())
            if city_info:
                results.append({
                    "name": match_name,
                    "country": city_info["country"],
                    "alternatives": city_info["alternatives"],
                    "similarity_score": score
                })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching cities: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/cities")
async def get_cities(
    country: Optional[str] = Query(None, description="Filter by country name"),
    limit: Optional[int] = Query(None, ge=1, description="Limit number of results")
):
    """
    Get list of supported cities.
    
    - **country**: Filter by country name (optional)
    - **limit**: Limit number of results (optional)
    """
    cities = CITIES_DATA
    
    if country:
        cities = [city for city in cities if city["country"].lower() == country.lower()]
    
    if limit:
        cities = cities[:limit]
    
    return {
        "cities": cities,
        "total_cities": len(cities),
        "filtered_by_country": country
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)