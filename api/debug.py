from typing import Dict
import logging
from utilites import EnhancedDateParser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_response_structure(response: Dict, context: str = ""):
    """Debug helper to understand response structure"""
    
    logger.info(f"🔍 DEBUG {context} - Response structure:")
    logger.info(f"  Top-level keys: {list(response.keys())}")
    
    if "data_freshness" in response:
        logger.info(f"  data_freshness keys: {list(response['data_freshness'].keys())}")
    
    if "route_airlines" in response:
        logger.info(f"  route_airlines keys: {list(response['route_airlines'].keys())}")
        if "airlines" in response["route_airlines"]:
            logger.info(f"  Number of airlines: {len(response['route_airlines']['airlines'])}")


# Example usage scenarios:

def example_unknown_city_scenarios():
    """Examples of how the system handles unknown cities"""
    
    scenarios = [
        {
            "input": "Guwahati",
            "expected_behavior": """
            1. ❌ Not in static database
            2. 🔍 Try Amadeus API → finds GAU
            3. ✅ Add to dynamic database
            4. ✅ Return GAU with source='amadeus_api'
            """
        },
        {
            "input": "Shillong", 
            "expected_behavior": """
            1. ❌ Not in static database
            2. ❌ Amadeus API fails
            3. 💡 Provide suggestions:
               - "Similar to 'Shanghai' (PVG)"
               - "Try nearest major city"
               - "Use airport code if known"
            """
        },
        {
            "input": "Mumbay",  # Misspelled
            "expected_behavior": """
            1. 🔧 Spell checker: 95% match with "Mumbai"
            2. ✅ Auto-correct to Mumbai → BOM
            3. ✅ High confidence correction
            """
        },
        {
            "input": "Xyz123",  # Invalid
            "expected_behavior": """
            1. ❌ No matches found
            2. ❌ Amadeus API fails
            3. 💡 Generic help suggestions
            4. 📝 Log for admin review
            """
        }
    ]
    
    return scenarios


def test_date_parsing():
    """Test the enhanced date parsing"""
    
    parser = EnhancedDateParser()
    
    test_queries = [
        "Find business flights from Delhi to Mumbai for August",
        "Book flights from BOM to DEL in December",
        "Flight from Kochi to Guwahati tomorrow",
        "Find flights next week",
        "Book ticket for 15 August 2025",
        "Flight on August 20, 2025",
        "Travel in next month"
    ]
    
    print("🧪 Testing Enhanced Date Parsing:")
    print("=" * 50)
    
    for query in test_queries:
        extracted_date = parser.extract_date_from_query(query)
        print(f"Query: '{query}'")
        print(f"Extracted: {extracted_date}")
        print("-" * 30)

    
example_results = {
    "query": "Find business flights from Delhi to Mumbai for August",
    "before_fix": {
        "search_date": "2025-07-07",  # ❌ Wrong month
        "reason": "Used default date instead of parsing 'August'"
    },
    "after_fix": {
        "search_date": "2025-08-15",  # ✅ Correct month
        "reason": "Correctly parsed 'August' from query"
    }
}


print("\n📋 Expected Fix Results:")
print("=" * 30)
print(f"Query: {example_results['query']}")
print(f"Before: {example_results['before_fix']['search_date']} - {example_results['before_fix']['reason']}")
print(f"After:  {example_results['after_fix']['search_date']} - {example_results['after_fix']['reason']}")
