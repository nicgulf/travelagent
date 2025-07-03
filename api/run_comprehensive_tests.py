#!/usr/bin/env python3
"""
Comprehensive Test Runner for Travel Agent
Standalone script to run all tests and generate detailed reports
"""

import asyncio
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_testing_framework import EnhancedTravelAgentTester, TestCategory
from utilites import FlightSearchMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveTestRunner:
    """
    Comprehensive test runner for the travel agent system
    """
    
    def __init__(self, openai_api_key: str = None, include_performance_tests: bool = False):
        """Initialize the test runner"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.include_performance_tests = include_performance_tests
        self.flight_service = None
        self.test_results = {}
        
    async def initialize_services(self):
        """Initialize required services for testing"""
        try:
            logger.info("🔧 Initializing flight service...")
            self.flight_service = FlightSearchMCPServer()
            logger.info("✅ Flight service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize flight service: {e}")
            self.flight_service = None
    
    async def run_all_tests(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run all test categories"""
        logger.info("🧪 Starting comprehensive test suite...")
        
        # Initialize tester
        tester = EnhancedTravelAgentTester(openai_api_key=self.openai_api_key)
        
        # Define test categories to run
        test_categories = categories or [
            TestCategory.MONTH_HANDLING,
            TestCategory.DATE_VALIDATION, 
            TestCategory.CITY_SPELLING,
            TestCategory.FOLLOW_UP_QUERIES
        ]
        
        all_results = {}
        overall_start_time = datetime.now()
        
        for category in test_categories:
            logger.info(f"📋 Running {category.value} tests...")
            category_results = await self._run_category_tests(tester, category)
            all_results[category.value] = category_results
        
        overall_end_time = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(all_results, overall_start_time, overall_end_time)
        
        return report
    
    async def _run_category_tests(self, tester: EnhancedTravelAgentTester, category: TestCategory) -> Dict[str, Any]:
        """Run tests for a specific category"""
        try:
            # Get test cases for category
            if category == TestCategory.MONTH_HANDLING:
                test_cases = tester.create_month_handling_tests()
            elif category == TestCategory.DATE_VALIDATION:
                test_cases = tester.create_date_validation_tests()
            elif category == TestCategory.CITY_SPELLING:
                test_cases = tester.create_city_spelling_tests()
            elif category == TestCategory.FOLLOW_UP_QUERIES:
                test_cases = tester.create_follow_up_query_tests()
            else:
                return {"error": f"Unknown category: {category}"}
            
            # Run tests
            results = []
            for test_case in test_cases:
                logger.info(f"  🔍 Running: {test_case.name}")
                result = await tester.run_test_case(test_case, self.flight_service)
                results.append(result)
                
                # Log result
                status_emoji = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⏭️"
                logger.info(f"    {status_emoji} {result['status']} ({result['execution_time']:.2f}s)")
            
            # Calculate category summary
            summary = tester._calculate_test_summary(results, sum(r["execution_time"] for r in results))
            
            return {
                "summary": summary,
                "results": results,
                "category": category.value
            }
            
        except Exception as e:
            logger.error(f"❌ Category {category.value} tests failed: {e}")
            return {"error": str(e), "category": category.value}
    
    def _generate_comprehensive_report(self, all_results: Dict, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        total_duration = (end_time - start_time).total_seconds()
        
        # Aggregate statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_skipped = 0
        
        category_summaries = {}
        
        for category, results in all_results.items():
            if "error" in results:
                category_summaries[category] = {"status": "error", "error": results["error"]}
                continue
                
            summary = results["summary"]
            category_summaries[category] = summary
            
            total_tests += summary["total_tests"]
            total_passed += summary["passed"]
            total_failed += summary["failed"]
            total_errors += summary["errors"]
            total_skipped += summary["skipped"]
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Generate insights and recommendations
        insights = self._generate_insights(category_summaries, overall_success_rate)
        recommendations = self._generate_recommendations(category_summaries, overall_success_rate)
        
        return {
            "test_execution_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "errors": total_errors,
                "skipped": total_skipped,
                "overall_success_rate": overall_success_rate,
                "overall_status": "PASSED" if total_failed == 0 and total_errors == 0 else "FAILED"
            },
            "category_results": all_results,
            "category_summaries": category_summaries,
            "insights": insights,
            "recommendations": recommendations,
            "system_info": {
                "openai_available": bool(self.openai_api_key),
                "flight_service_available": self.flight_service is not None,
                "python_version": sys.version,
                "test_timestamp": datetime.now().isoformat()
            }
        }
    
    def _generate_insights(self, category_summaries: Dict, overall_success_rate: float) -> List[str]:
        """Generate insights from test results"""
        insights = []
        
        # Overall performance insights
        if overall_success_rate >= 95:
            insights.append("🎉 Excellent overall performance - system is highly reliable")
        elif overall_success_rate >= 85:
            insights.append("✅ Good overall performance - minor issues detected")
        elif overall_success_rate >= 70:
            insights.append("⚠️ Moderate performance - several issues need attention")
        else:
            insights.append("🚨 Poor performance - significant issues detected")
        
        # Category-specific insights
        for category, summary in category_summaries.items():
            if "error" in summary:
                insights.append(f"❌ {category} tests could not run - check system configuration")
                continue
                
            success_rate = summary.get("success_rate", 0)
            if success_rate < 80:
                if category == "month_handling":
                    insights.append("📅 Month parsing needs improvement - check date parsing logic")
                elif category == "date_validation":
                    insights.append("📆 Date validation issues - review date handling algorithms")
                elif category == "city_spelling":
                    insights.append("🏙️ City spelling correction needs enhancement - update spell checker database")
                elif category == "follow_up_queries":
                    insights.append("🤖 Follow-up query understanding needs improvement - check OpenAI integration")
        
        return insights
    
    def _generate_recommendations(self, category_summaries: Dict, overall_success_rate: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if overall_success_rate < 90:
            recommendations.append("🔧 Review and fix failing tests to improve system reliability")
        
        # Check for specific issues
        for category, summary in category_summaries.items():
            if "error" in summary:
                recommendations.append(f"🛠️ Fix {category} test infrastructure issues")
                continue
                
            if summary.get("failed", 0) > 0:
                recommendations.append(f"🔍 Investigate {summary['failed']} failed tests in {category}")
            
            if summary.get("average_execution_time", 0) > 5.0:
                recommendations.append(f"⚡ Optimize {category} test performance (avg: {summary['average_execution_time']:.2f}s)")
        
        # OpenAI-specific recommendations
        if not self.openai_api_key:
            recommendations.append("🔑 Configure OpenAI API key to enable follow-up query testing")
        
        if not recommendations:
            recommendations.append("🎯 All tests passing - consider adding more edge case tests")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"travel_agent_test_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Test report saved to: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")

async def main():
    """Main function to run comprehensive tests"""
    parser = argparse.ArgumentParser(description="Run comprehensive travel agent tests")
    parser.add_argument("--categories", nargs="+", help="Test categories to run", 
                       choices=["month_handling", "date_validation", "city_spelling", "follow_up_queries"])
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--performance", action="store_true", help="Include performance tests")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner(
        openai_api_key=args.openai_key,
        include_performance_tests=args.performance
    )
    
    # Initialize services
    await runner.initialize_services()
    
    # Convert category strings to enums if specified
    categories = None
    if args.categories:
        category_map = {
            "month_handling": TestCategory.MONTH_HANDLING,
            "date_validation": TestCategory.DATE_VALIDATION,
            "city_spelling": TestCategory.CITY_SPELLING,
            "follow_up_queries": TestCategory.FOLLOW_UP_QUERIES
        }
        categories = [category_map[cat] for cat in args.categories if cat in category_map]
    
    # Run tests
    logger.info("🚀 Starting comprehensive test execution...")
    report = await runner.run_all_tests(categories)
    
    # Display summary
    summary = report["test_execution_summary"]
    logger.info(f"📊 Test Summary:")
    logger.info(f"   Total Tests: {summary['total_tests']}")
    logger.info(f"   Passed: {summary['passed']}")
    logger.info(f"   Failed: {summary['failed']}")
    logger.info(f"   Errors: {summary['errors']}")
    logger.info(f"   Success Rate: {summary['overall_success_rate']:.1f}%")
    logger.info(f"   Duration: {summary['total_duration_seconds']:.2f}s")
    logger.info(f"   Status: {summary['overall_status']}")
    
    # Save report
    runner.save_report(report, args.output)
    
    # Print recommendations
    if report["recommendations"]:
        logger.info("💡 Recommendations:")
        for rec in report["recommendations"]:
            logger.info(f"   {rec}")
    
    return 0 if summary["overall_status"] == "PASSED" else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
