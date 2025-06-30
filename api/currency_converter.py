import aiohttp
from typing import Dict
import logging

logger = logging.getLogger(__name__)

async def convert_eur_to_inr(amount: float) -> Dict:
    """Simple EUR to INR converter using free API"""
    
    try:
        # Use free exchangerate.host API (no key required)
        url = f"https://api.exchangerate.host/convert?from=EUR&to=INR&amount={amount}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("success"):
                        inr_amount = data["result"]
                        exchange_rate = inr_amount / amount
                        
                        logger.info(f"✅ Converted: €{amount} = ₹{inr_amount:,.2f} (Rate: {exchange_rate})")
                        
                        return {
                            "converted_amount": round(inr_amount, 2),
                            "exchange_rate": round(exchange_rate, 2),
                            "formatted": f"₹{inr_amount:,.2f}"
                        }
        
        # Fallback if API fails
        fallback_rate = 89.50  # EUR to INR approximate rate
        inr_amount = amount * fallback_rate
        
        logger.warning(f"⚠️ Using fallback rate: €{amount} = ₹{inr_amount:,.2f}")
        
        return {
            "converted_amount": round(inr_amount, 2),
            "exchange_rate": fallback_rate,
            "formatted": f"₹{inr_amount:,.2f}",
            "fallback": True
        }
        
    except Exception as e:
        logger.error(f"❌ Currency conversion error: {e}")
        # Emergency fallback
        inr_amount = amount * 89.50
        return {
            "converted_amount": round(inr_amount, 2),
            "exchange_rate": 89.50,
            "formatted": f"₹{inr_amount:,.2f}",
            "error": True
        }