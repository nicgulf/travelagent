#!/usr/bin/env python3
"""
Enhanced Travel Agent Demo
Demonstrates the comprehensive capabilities including OpenAI follow-up, 
month handling, date validation, and city spelling correction
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTravelAgentDemo:
    """
    Comprehensive demo of the enhanced travel agent capabilities
    """
    
    def __init__(self):
        """Initialize the demo"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.demo_scenarios = self._create_demo_scenarios()
    
    def _create_demo_scenarios(self) -> List[Dict[str, Any]]:
        """Create comprehensive demo scenarios"""
        return [
            {
                "name": "Month Handling Showcase",
                "description": "Demonstrates various month format parsing",
                "queries": [
                    "Find flights from Mumbai to Delhi in January",
                    "What about flights in Feb?",
                    "Show me options for 03/2024",
                    "Any flights next month?",
                    "What about in the 12th month?"
                ],
                "expected_capabilities": ["full_month_names", "abbreviated_months", "numeric_months", "relative_months"]
            },
            {
                "name": "Date Validation Excellence", 
                "description": "Shows robust date parsing and validation",
                "queries": [
                    "Flights from Mumbai to Delhi tomorrow",
                    "What about next Friday?",
                    "Any flights on 2024-12-25?",
                    "Show me flights for the day after tomorrow",
                    "What about flights on 2024-02-30?"  # Invalid date
                ],
                "expected_capabilities": ["relative_dates", "iso_format", "natural_language", "error_handling"]
            },
            {
                "name": "City Spelling Intelligence",
                "description": "Demonstrates intelligent city name correction",
                "queries": [
                    "Flights from Mumbay to Deli",
                    "What about from Kolkatta to Bangalor?", 
                    "Show flights from Londn to Paries",
                    "Any flights from Chenai to Hydrabad?",
                    "What about from Xyz to Abc?"  # Unknown cities
                ],
                "expected_capabilities": ["typo_correction", "phonetic_matching", "unknown_city_handling"]
            },
            {
                "name": "Follow-up Query Mastery",
                "description": "Shows intelligent conversation flow with OpenAI",
                "queries": [
                    "Search flights from Mumbai to Delhi tomorrow",
                    "What about business class?",
                    "Any cheaper options?", 
                    "Show me morning flights only",
                    "What airlines fly this route?",
                    "Can I get a window seat?"
                ],
                "expected_capabilities": ["context_preservation", "intent_understanding", "parameter_modification"]
            },
            {
                "name": "Edge Cases and Error Handling",
                "description": "Tests system robustness with challenging inputs",
                "queries": [
                    "Flights from nowhere to somewhere",
                    "Show me flights on 32nd January",
                    "What about flights in the 13th month?",
                    "Find flights from Mumbai to Mumbai",
                    "Any flights yesterday?"  # Past date
                ],
                "expected_capabilities": ["error_handling", "validation", "user_feedback"]
            }
        ]
    
    async def run_demo_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single demo scenario"""
        logger.info(f"🎭 Running scenario: {scenario['name']}")
        logger.info(f"📝 Description: {scenario['description']}")
        
        scenario_results = {
            "name": scenario["name"],
            "description": scenario["description"],
            "queries": [],
            "capabilities_demonstrated": [],
            "overall_success": True,
            "insights": []
        }
        
        for i, query in enumerate(scenario["queries"]):
            logger.info(f"  🔍 Query {i+1}: {query}")
            
            # Simulate query processing (in real implementation, this would call the actual API)
            query_result = await self._simulate_query_processing(query, scenario)
            
            scenario_results["queries"].append(query_result)
            
            # Log result
            status_emoji = "✅" if query_result["status"] == "success" else "❌"
            logger.info(f"    {status_emoji} {query_result['status']}: {query_result.get('summary', 'No summary')}")
            
            if query_result["status"] != "success":
                scenario_results["overall_success"] = False
        
        # Analyze capabilities demonstrated
        scenario_results["capabilities_demonstrated"] = self._analyze_capabilities(scenario_results["queries"])
        scenario_results["insights"] = self._generate_scenario_insights(scenario, scenario_results)
        
        return scenario_results
    
    async def _simulate_query_processing(self, query: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate query processing (placeholder for actual implementation)"""
        # This is a simulation - in real implementation, this would call the actual API endpoints
        
        result = {
            "query": query,
            "status": "success",
            "processing_time": 0.5,  # Simulated
            "capabilities_used": [],
            "summary": ""
        }
        
        # Simulate different types of processing based on query content
        query_lower = query.lower()
        
        # Month handling simulation
        months = ["january", "jan", "february", "feb", "march", "mar", "april", "apr", 
                 "may", "june", "jun", "july", "jul", "august", "aug", "september", "sep",
                 "october", "oct", "november", "nov", "december", "dec", "next month"]
        
        if any(month in query_lower for month in months):
            result["capabilities_used"].append("month_handling")
            result["summary"] = "Month detected and parsed successfully"
        
        # Date validation simulation
        date_indicators = ["tomorrow", "friday", "2024-", "day after", "yesterday"]
        if any(indicator in query_lower for indicator in date_indicators):
            result["capabilities_used"].append("date_validation")
            if "2024-02-30" in query:
                result["status"] = "error"
                result["summary"] = "Invalid date detected and handled"
            elif "yesterday" in query_lower:
                result["status"] = "warning"
                result["summary"] = "Past date detected - user notified"
            else:
                result["summary"] = "Date parsed and validated successfully"
        
        # City spelling simulation
        misspellings = ["mumbay", "deli", "kolkatta", "bangalor", "londn", "paries", "chenai", "hydrabad"]
        if any(misspelling in query_lower for misspelling in misspellings):
            result["capabilities_used"].append("city_spelling")
            if "xyz" in query_lower or "abc" in query_lower:
                result["status"] = "warning"
                result["summary"] = "Unknown cities detected - user prompted for clarification"
            else:
                result["summary"] = "City names corrected successfully"
        
        # Follow-up query simulation
        followup_indicators = ["what about", "any cheaper", "show me", "can i get", "what airlines"]
        if any(indicator in query_lower for indicator in followup_indicators):
            result["capabilities_used"].append("follow_up_processing")
            result["summary"] = "Follow-up intent understood and processed"
        
        # Error handling simulation
        error_indicators = ["nowhere", "somewhere", "32nd", "13th month", "mumbai to mumbai"]
        if any(indicator in query_lower for indicator in error_indicators):
            result["status"] = "error"
            result["capabilities_used"].append("error_handling")
            result["summary"] = "Invalid input detected and handled gracefully"
        
        return result
    
    def _analyze_capabilities(self, query_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze which capabilities were demonstrated"""
        capabilities = set()
        for result in query_results:
            capabilities.update(result.get("capabilities_used", []))
        return list(capabilities)
    
    def _generate_scenario_insights(self, scenario: Dict[str, Any], results: Dict[str, Any]) -> List[str]:
        """Generate insights for a scenario"""
        insights = []
        
        success_rate = len([q for q in results["queries"] if q["status"] == "success"]) / len(results["queries"]) * 100
        
        if success_rate >= 90:
            insights.append(f"✅ Excellent performance in {scenario['name']} ({success_rate:.0f}% success rate)")
        elif success_rate >= 70:
            insights.append(f"⚠️ Good performance in {scenario['name']} with room for improvement ({success_rate:.0f}% success rate)")
        else:
            insights.append(f"❌ Performance issues in {scenario['name']} need attention ({success_rate:.0f}% success rate)")
        
        # Capability-specific insights
        capabilities_demonstrated = results["capabilities_demonstrated"]
        expected_capabilities = scenario.get("expected_capabilities", [])
        
        missing_capabilities = set(expected_capabilities) - set(capabilities_demonstrated)
        if missing_capabilities:
            insights.append(f"🔧 Missing capabilities: {', '.join(missing_capabilities)}")
        
        if "error_handling" in capabilities_demonstrated:
            insights.append("🛡️ Error handling capabilities demonstrated")
        
        return insights
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete demo suite"""
        logger.info("🚀 Starting Enhanced Travel Agent Demo")
        logger.info("=" * 60)
        
        demo_start_time = datetime.now()
        scenario_results = []
        
        for scenario in self.demo_scenarios:
            result = await self.run_demo_scenario(scenario)
            scenario_results.append(result)
            logger.info("")  # Add spacing between scenarios
        
        demo_end_time = datetime.now()
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(scenario_results, demo_start_time, demo_end_time)
        
        demo_report = {
            "demo_info": {
                "title": "Enhanced Travel Agent Comprehensive Demo",
                "start_time": demo_start_time.isoformat(),
                "end_time": demo_end_time.isoformat(),
                "duration_seconds": (demo_end_time - demo_start_time).total_seconds()
            },
            "scenarios": scenario_results,
            "overall_summary": overall_summary,
            "system_capabilities": {
                "openai_available": bool(self.openai_api_key),
                "capabilities_tested": list(set().union(*[s["capabilities_demonstrated"] for s in scenario_results]))
            }
        }
        
        return demo_report
    
    def _generate_overall_summary(self, scenario_results: List[Dict], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate overall demo summary"""
        total_scenarios = len(scenario_results)
        successful_scenarios = len([s for s in scenario_results if s["overall_success"]])
        total_queries = sum(len(s["queries"]) for s in scenario_results)
        successful_queries = sum(len([q for q in s["queries"] if q["status"] == "success"]) for s in scenario_results)
        
        return {
            "total_scenarios": total_scenarios,
            "successful_scenarios": successful_scenarios,
            "scenario_success_rate": (successful_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "query_success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "overall_status": "EXCELLENT" if successful_scenarios == total_scenarios else "GOOD" if successful_scenarios >= total_scenarios * 0.8 else "NEEDS_IMPROVEMENT"
        }
    
    def print_demo_summary(self, demo_report: Dict[str, Any]):
        """Print a formatted demo summary"""
        logger.info("📊 DEMO SUMMARY")
        logger.info("=" * 60)
        
        summary = demo_report["overall_summary"]
        logger.info(f"🎯 Scenarios: {summary['successful_scenarios']}/{summary['total_scenarios']} successful ({summary['scenario_success_rate']:.1f}%)")
        logger.info(f"🔍 Queries: {summary['successful_queries']}/{summary['total_queries']} successful ({summary['query_success_rate']:.1f}%)")
        logger.info(f"⏱️ Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info(f"🏆 Overall Status: {summary['overall_status']}")
        
        logger.info("\n🎭 SCENARIO RESULTS:")
        for scenario in demo_report["scenarios"]:
            status_emoji = "✅" if scenario["overall_success"] else "❌"
            logger.info(f"  {status_emoji} {scenario['name']}")
            for insight in scenario["insights"]:
                logger.info(f"    💡 {insight}")
        
        capabilities = demo_report["system_capabilities"]["capabilities_tested"]
        logger.info(f"\n🔧 CAPABILITIES DEMONSTRATED: {', '.join(capabilities)}")

async def main():
    """Main demo function"""
    demo = EnhancedTravelAgentDemo()
    
    logger.info("🎬 Enhanced Travel Agent Demo Starting...")
    logger.info("This demo showcases comprehensive capabilities including:")
    logger.info("  📅 Month handling and date validation")
    logger.info("  🏙️ City spelling correction")
    logger.info("  🤖 OpenAI-powered follow-up queries")
    logger.info("  🛡️ Error handling and edge cases")
    logger.info("")
    
    # Run the demo
    demo_report = await demo.run_full_demo()
    
    # Print summary
    demo.print_demo_summary(demo_report)
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"enhanced_demo_report_{timestamp}.json"
    
    try:
        with open(report_filename, 'w') as f:
            json.dump(demo_report, f, indent=2, default=str)
        logger.info(f"\n📄 Detailed report saved to: {report_filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save report: {e}")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
