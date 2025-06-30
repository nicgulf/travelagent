CITIES_DATA = [
    # Indian Cities - Only static information
    {"name": "Mumbai", "country": "India", "alternatives": ["Bombay"], "airport_code": "BOM"},
    {"name": "Delhi", "country": "India", "alternatives": ["New Delhi"], "airport_code": "DEL"},
    {"name": "Bangalore", "country": "India", "alternatives": ["Bengaluru"], "airport_code": "BLR"},
    {"name": "Chennai", "country": "India", "alternatives": ["Madras"], "airport_code": "MAA"},
    {"name": "Kolkata", "country": "India", "alternatives": ["Calcutta"], "airport_code": "CCU"},
    {"name": "Hyderabad", "country": "India", "alternatives": [], "airport_code": "HYD"},
    {"name": "Pune", "country": "India", "alternatives": ["Poona"], "airport_code": "PNQ"},
    {"name": "Ahmedabad", "country": "India", "alternatives": [], "airport_code": "AMD"},
    {"name": "Jaipur", "country": "India", "alternatives": [], "airport_code": "JAI"},
    {"name": "Lucknow", "country": "India", "alternatives": [], "airport_code": "LKO"},
    {"name": "Guwahati", "country": "India", "alternatives": [], "airport_code": "GAU"},
    {"name": "Kochi", "country": "India", "alternatives": ["Cochin"], "airport_code": "COK"},
    {"name": "Goa", "country": "India", "alternatives": [], "airport_code": "GOI"},
    
    # US Cities
    {"name": "New York", "country": "USA", "alternatives": ["NYC", "New York City"], "airport_code": "JFK"},
    {"name": "Los Angeles", "country": "USA", "alternatives": ["LA"], "airport_code": "LAX"},
    {"name": "Chicago", "country": "USA", "alternatives": [], "airport_code": "ORD"},
    {"name": "Miami", "country": "USA", "alternatives": [], "airport_code": "MIA"},
    {"name": "Boston", "country": "USA", "alternatives": [], "airport_code": "BOS"},
    {"name": "Seattle", "country": "USA", "alternatives": [], "airport_code": "SEA"},
    {"name": "Las Vegas", "country": "USA", "alternatives": ["Vegas"], "airport_code": "LAS"},
    
    # International Cities
    {"name": "London", "country": "UK", "alternatives": [], "airport_code": "LHR"},
    {"name": "Paris", "country": "France", "alternatives": [], "airport_code": "CDG"},
    {"name": "Tokyo", "country": "Japan", "alternatives": [], "airport_code": "NRT"},
    {"name": "Dubai", "country": "UAE", "alternatives": [], "airport_code": "DXB"},
    {"name": "Singapore", "country": "Singapore", "alternatives": [], "airport_code": "SIN"},
    {"name": "Bangkok", "country": "Thailand", "alternatives": [], "airport_code": "BKK"},
    {"name": "Hong Kong", "country": "Hong Kong", "alternatives": [], "airport_code": "HKG"},
    {"name": "Sydney", "country": "Australia", "alternatives": [], "airport_code": "SYD"},
    {"name": "Melbourne", "country": "Australia", "alternatives": [], "airport_code": "MEL"},
    {"name": "Frankfurt", "country": "Germany", "alternatives": [], "airport_code": "FRA"},
]

# Create spell checking maps
CITY_NAMES = []
CITY_MAP = {}
CITY_TO_AIRPORT = {}

for city in CITIES_DATA:
    CITY_NAMES.append(city["name"])
    CITY_MAP[city["name"].lower()] = city
    CITY_TO_AIRPORT[city["name"].lower()] = city["airport_code"]
    
    for alt in city["alternatives"]:
        CITY_NAMES.append(alt)
        CITY_MAP[alt.lower()] = city
        CITY_TO_AIRPORT[alt.lower()] = city["airport_code"]