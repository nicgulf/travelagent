#!/usr/bin/env python3

import asyncio
import time
import logging
from functools import wraps, lru_cache
from typing import Dict, Any, Optional, Callable
import aiohttp
import redis.asyncio as redis
from contextlib import asynccontextmanager
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Performance optimization utilities for the flight search API
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.connection_pool = None
        self.circuit_breaker_states = {}
        self.request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        
        # Initialize Redis if available
        if redis_url:
            self._init_redis(redis_url)
        
        # Initialize HTTP connection pool
        self._init_http_pool()
    
    def _init_redis(self, redis_url: str):
        """Initialize Redis connection with optimized settings"""
        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info("✅ Redis connection pool initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization failed: {e}")
    
    def _init_http_pool(self):
        """Initialize HTTP connection pool for external APIs"""
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Max connections per host
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=20  # Socket read timeout
        )
        
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'FlightSearchAPI/3.0.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
        
        logger.info("✅ HTTP connection pool initialized")

    @asynccontextmanager
    async def performance_monitor(self, operation_name: str):
        """Context manager for monitoring operation performance"""
        start_time = time.time()
        self.request_metrics["total_requests"] += 1
        
        try:
            yield
            self.request_metrics["successful_requests"] += 1
            
        except Exception as e:
            self.request_metrics["failed_requests"] += 1
            logger.error(f"❌ Operation {operation_name} failed: {e}")
            raise
            
        finally:
            duration = time.time() - start_time
            # Update rolling average
            current_avg = self.request_metrics["average_response_time"]
            total_requests = self.request_metrics["total_requests"]
            new_avg = ((current_avg * (total_requests - 1)) + duration) / total_requests
            self.request_metrics["average_response_time"] = new_avg
            
            logger.info(f"⏱️ {operation_name} completed in {duration:.3f}s")

    def cache_key_builder(self, prefix: str, **kwargs) -> str:
        """Build cache key from parameters"""
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        param_str = "_".join(f"{k}:{v}" for k, v in sorted_params)
        return f"{prefix}:{param_str}"

    async def cached_operation(self, 
                             cache_key: str, 
                             operation: Callable,
                             ttl: int = 3600,
                             *args, **kwargs) -> Any:
        """Execute operation with caching"""
        if not self.redis_client:
            # No caching available, execute directly
            return await operation(*args, **kwargs)
        
        try:
            # Try to get from cache
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                logger.info(f"🎯 Cache hit for key: {cache_key}")
                return json.loads(cached_result)
            
            # Cache miss - execute operation
            logger.info(f"💾 Cache miss for key: {cache_key}")
            result = await operation(*args, **kwargs)
            
            # Store in cache
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result, default=str)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Cache operation failed for {cache_key}: {e}")
            # Fallback to direct execution
            return await operation(*args, **kwargs)

    def circuit_breaker(self, 
                       service_name: str, 
                       failure_threshold: int = 5,
                       timeout: int = 60):
        """Circuit breaker decorator for external service calls"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                state = self.circuit_breaker_states.get(service_name, {
                    "failures": 0,
                    "last_failure": None,
                    "state": "closed"  # closed, open, half-open
                })
                
                # Check if circuit is open
                if state["state"] == "open":
                    if time.time() - state["last_failure"] < timeout:
                        raise Exception(f"Circuit breaker OPEN for {service_name}")
                    else:
                        state["state"] = "half-open"
                
                try:
                    result = await func(*args, **kwargs)
                    # Success - reset circuit breaker
                    state["failures"] = 0
                    state["state"] = "closed"
                    self.circuit_breaker_states[service_name] = state
                    return result
                    
                except Exception as e:
                    # Failure - update circuit breaker
                    state["failures"] += 1
                    state["last_failure"] = time.time()
                    
                    if state["failures"] >= failure_threshold:
                        state["state"] = "open"
                        logger.warning(f"🔴 Circuit breaker OPENED for {service_name}")
                    
                    self.circuit_breaker_states[service_name] = state
                    raise
                    
            return wrapper
        return decorator

    async def batch_operation(self, 
                            operations: list, 
                            batch_size: int = 10,
                            max_concurrency: int = 5) -> list:
        """Execute operations in batches with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_operation(op):
            async with semaphore:
                return await op
        
        results = []
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[bounded_operation(op) for op in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
            
            # Small delay between batches to prevent overwhelming
            if i + batch_size < len(operations):
                await asyncio.sleep(0.1)
        
        return results

    @lru_cache(maxsize=1000)
    def static_data_cache(self, key: str, data: str) -> Any:
        """LRU cache for static data (airport codes, city mappings, etc.)"""
        return json.loads(data)

    async def cleanup(self):
        """Cleanup resources"""
        if self.http_session:
            await self.http_session.close()
        if self.redis_client:
            await self.redis_client.close()

class ConfigurationManager:
    """
    Configuration management for the flight search API
    """
    
    def __init__(self):
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables and defaults"""
        return {
            # API Configuration
            "API": {
                "HOST": os.getenv("API_HOST", "0.0.0.0"),
                "PORT": int(os.getenv("API_PORT", "8000")),
                "WORKERS": int(os.getenv("API_WORKERS", "1")),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "*").split(","),
                "REQUEST_TIMEOUT": float(os.getenv("REQUEST_TIMEOUT", "30.0")),
                "MAX_REQUEST_SIZE": int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
            },
            
            # External APIs
            "AMADEUS": {
                "API_KEY": os.getenv("AMADEUS_API_KEY"),
                "API_SECRET": os.getenv("AMADEUS_API_SECRET"),
                "BASE_URL": os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com"),
                "TIMEOUT": float(os.getenv("AMADEUS_TIMEOUT", "30.0")),
                "RETRY_ATTEMPTS": int(os.getenv("AMADEUS_RETRY_ATTEMPTS", "3")),
                "RATE_LIMIT": int(os.getenv("AMADEUS_RATE_LIMIT", "100"))  # per minute
            },
            
            "OPENAI": {
                "API_KEY": os.getenv("OPENAI_API_KEY"),
                "MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "TIMEOUT": float(os.getenv("OPENAI_TIMEOUT", "15.0")),
                "MAX_TOKENS": int(os.getenv("OPENAI_MAX_TOKENS", "600")),
                "TEMPERATURE": float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
            },
            
            # Redis Configuration
            "REDIS": {
                "URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
                "MAX_CONNECTIONS": int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
                "CONVERSATION_TTL": int(os.getenv("CONVERSATION_TTL", "86400")),  # 24 hours
                "CACHE_TTL": int(os.getenv("CACHE_TTL", "3600")),  # 1 hour
                "ANALYTICS_RETENTION": int(os.getenv("ANALYTICS_RETENTION", "604800"))  # 7 days
            },
            
            # Performance Settings
            "PERFORMANCE": {
                "MAX_CONVERSATION_HISTORY": int(os.getenv("MAX_CONVERSATION_HISTORY", "10")),
                "MAX_SEARCH_RESULTS": int(os.getenv("MAX_SEARCH_RESULTS", "20")),
                "PARALLEL_SEARCHES": int(os.getenv("PARALLEL_SEARCHES", "5")),
                "CIRCUIT_BREAKER_THRESHOLD": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
                "CIRCUIT_BREAKER_TIMEOUT": int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60")),
                "CONNECTION_POOL_SIZE": int(os.getenv("CONNECTION_POOL_SIZE", "100"))
            },
            
            # Feature Flags
            "FEATURES": {
                "AI_FOLLOWUP": os.getenv("ENABLE_AI_FOLLOWUP", "true").lower() == "true",
                "REDIS_CACHING": os.getenv("ENABLE_REDIS_CACHING", "true").lower() == "true",
                "SPELL_CHECKING": os.getenv("ENABLE_SPELL_CHECKING", "true").lower() == "true",
                "ANALYTICS": os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
                "RATE_LIMITING": os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true",
                "BACKGROUND_TASKS": os.getenv("ENABLE_BACKGROUND_TASKS", "true").lower() == "true"
            },
            
            # Logging Configuration
            "LOGGING": {
                "LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "FORMAT": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                "FILE": os.getenv("LOG_FILE"),
                "MAX_SIZE": int(os.getenv("LOG_MAX_SIZE", "10485760")),  # 10MB
                "BACKUP_COUNT": int(os.getenv("LOG_BACKUP_COUNT", "5"))
            }
        }
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.config.get("FEATURES", {}).get(feature, False)

class RateLimiter:
    """
    Redis-based rate limiter for API endpoints
    """
    
    def __init__(self, redis_client, default_limit: int = 100, window: int = 60):
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.window = window
    
    async def is_allowed(self, 
                        key: str, 
                        limit: int = None, 
                        window: int = None) -> tuple[bool, dict]:
        """Check if request is allowed under rate limit"""
        if not self.redis_client:
            return True, {}
        
        limit = limit or self.default_limit
        window = window or self.window
        
        try:
            # Use sliding window with Redis
            now = time.time()
            pipeline = self.redis_client.pipeline()
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, now - window)
            
            # Count current entries
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(now): now})
            
            # Set expiry
            pipeline.expire(key, window)
            
            results = await pipeline.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= limit:
                return False, {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_time": datetime.fromtimestamp(now + window),
                    "retry_after": window
                }
            
            return True, {
                "allowed": True,
                "limit": limit,
                "remaining": limit - current_count - 1,
                "reset_time": datetime.fromtimestamp(now + window)
            }
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Allow request on error
            return True, {}

class HealthMonitor:
    """
    Health monitoring for all system components
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.startup_time = datetime.now()
        self.health_checks = {}
    
    async def check_amadeus_api(self) -> dict:
        """Check Amadeus API health"""
        try:
            # Simple API call to check connectivity
            async with aiohttp.ClientSession() as session:
                auth_url = f"{self.config.get('AMADEUS', 'BASE_URL')}/v1/security/oauth2/token"
                data = {
                    "grant_type": "client_credentials",
                    "client_id": self.config.get("AMADEUS", "API_KEY"),
                    "client_secret": self.config.get("AMADEUS", "API_SECRET")
                }
                
                async with session.post(auth_url, data=data, timeout=10) as response:
                    if response.status == 200:
                        return {"status": "healthy", "response_time": response.headers.get("response-time")}
                    else:
                        return {"status": "unhealthy", "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_openai_api(self) -> dict:
        """Check OpenAI API health"""
        try:
            if not self.config.get("OPENAI", "API_KEY"):
                return {"status": "not_configured"}
            
            import openai
            client = openai.AsyncOpenAI(api_key=self.config.get("OPENAI", "API_KEY"))
            
            # Simple API call
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=5.0
            )
            
            return {"status": "healthy", "model": response.model}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_redis(self, redis_client) -> dict:
        """Check Redis health"""
        try:
            if not redis_client:
                return {"status": "not_configured"}
            
            start_time = time.time()
            await redis_client.ping()
            response_time = time.time() - start_time
            
            info = await redis_client.info()
            return {
                "status": "healthy",
                "response_time": round(response_time * 1000, 2),  # ms
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "version": info.get("redis_version")
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def get_system_metrics(self) -> dict:
        """Get comprehensive system metrics"""
        uptime = datetime.now() - self.startup_time
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "uptime_human": str(uptime),
            "startup_time": self.startup_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "health_checks": self.health_checks
        }

class DatabaseOptimizer:
    """
    Database and query optimization utilities
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.query_cache = {}
        self.connection_pool = None
    
    async def optimize_conversation_storage(self, session_id: str, data: dict):
        """Optimize conversation data storage"""
        try:
            # Compress conversation data
            compressed_data = self._compress_conversation_data(data)
            
            # Store with TTL
            await self.redis_client.setex(
                f"conversation:{session_id}",
                86400,  # 24 hours
                json.dumps(compressed_data)
            )
            
            # Index for fast lookup
            await self.redis_client.sadd("active_conversations", session_id)
            
        except Exception as e:
            logger.error(f"Conversation storage optimization failed: {e}")
    
    def _compress_conversation_data(self, data: dict) -> dict:
        """Compress conversation data by removing redundant information"""
        compressed = data.copy()
        
        # Keep only essential message data
        if "messages" in compressed:
            for message in compressed["messages"]:
                # Remove large content if older than 1 hour
                if "timestamp" in message:
                    try:
                        msg_time = datetime.fromisoformat(message["timestamp"])
                        if (datetime.now() - msg_time).total_seconds() > 3600:
                            if isinstance(message.get("content"), dict):
                                message["content"] = {"summary": "Compressed old message"}
                    except:
                        pass
        
        # Compress search results - keep only summary
        if "last_search" in compressed and compressed["last_search"]:
            search = compressed["last_search"]
            if "flights" in search and len(search["flights"]) > 5:
                # Keep only top 5 flights
                search["flights"] = search["flights"][:5]
                search["compressed"] = True
        
        return compressed
    
    async def cleanup_expired_data(self):
        """Cleanup expired conversation and analytics data"""
        try:
            # Get all conversation keys
            conversation_keys = await self.redis_client.smembers("active_conversations")
            
            expired_sessions = []
            for session_id in conversation_keys:
                exists = await self.redis_client.exists(f"conversation:{session_id}")
                if not exists:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions from index
            if expired_sessions:
                await self.redis_client.srem("active_conversations", *expired_sessions)
                logger.info(f"Cleaned up {len(expired_sessions)} expired conversation sessions")
            
            # Cleanup old analytics data
            cutoff_time = time.time() - 604800  # 7 days
            await self.redis_client.zremrangebyscore("analytics_timeline", 0, cutoff_time)
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

class APIOptimizer:
    """
    API-specific optimizations and middleware
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.request_cache = {}
        self.response_cache = {}
    
    def optimize_request_payload(self, payload: dict) -> dict:
        """Optimize request payload size and structure"""
        optimized = payload.copy()
        
        # Remove empty or None values
        optimized = {k: v for k, v in optimized.items() if v is not None and v != ""}
        
        # Normalize string values
        for key, value in optimized.items():
            if isinstance(value, str):
                optimized[key] = value.strip()
        
        return optimized
    
    def compress_response(self, response_data: dict) -> dict:
        """Compress response data for faster transmission"""
        compressed = response_data.copy()
        
        # Remove debug information in production
        if not self.config.get("API", "DEBUG", False):
            debug_keys = ["debug_info", "trace_id", "raw_response"]
            for key in debug_keys:
                compressed.pop(key, None)
        
        # Optimize flight data
        if "flights" in compressed:
            for flight in compressed["flights"]:
                # Remove redundant fields
                flight.pop("data_fetched_at", None)
                flight.pop("price_last_updated", None)
                
                # Round numeric values
                if "price_numeric" in flight:
                    flight["price_numeric"] = round(flight["price_numeric"], 2)
        
        return compressed
    
    async def cache_response(self, cache_key: str, response_data: dict, ttl: int = 300):
        """Cache API response with TTL"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                await self.redis_client.setex(
                    f"response_cache:{cache_key}",
                    ttl,
                    json.dumps(response_data, default=str)
                )
        except Exception as e:
            logger.error(f"Response caching failed: {e}")
    
    async def get_cached_response(self, cache_key: str) -> Optional[dict]:
        """Get cached API response"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                cached = await self.redis_client.get(f"response_cache:{cache_key}")
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        return None

class AsyncTaskManager:
    """
    Manage background tasks and async operations
    """
    
    def __init__(self):
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()
        self.workers = []
    
    async def start_workers(self, num_workers: int = 3):
        """Start background worker tasks"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {num_workers} background workers")
    
    async def _worker(self, worker_name: str):
        """Background worker to process tasks"""
        while True:
            try:
                task_func, args, kwargs = await self.task_queue.get()
                
                start_time = time.time()
                try:
                    await task_func(*args, **kwargs)
                    logger.info(f"✅ {worker_name} completed task in {time.time() - start_time:.3f}s")
                except Exception as e:
                    logger.error(f"❌ {worker_name} task failed: {e}")
                finally:
                    self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    async def add_task(self, task_func, *args, **kwargs):
        """Add task to background queue"""
        await self.task_queue.put((task_func, args, kwargs))
    
    async def shutdown(self):
        """Shutdown background workers"""
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Background workers shutdown complete")

class SecurityOptimizer:
    """
    Security optimizations and protections
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.blocked_ips = set()
        self.suspicious_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'expression\s*\(',
            r'eval\s*\(',
            r'document\.',
            r'window\.',
        ]
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        import re
        import html
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove suspicious patterns
        for pattern in self.suspicious_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Limit length
        max_length = self.config.get("API", "MAX_INPUT_LENGTH", 1000)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def validate_session_id(self, session_id: str) -> bool:
        """Validate session ID format"""
        import re
        
        if not session_id:
            return False
        
        # Check format (alphanumeric, hyphens, underscores only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            return False
        
        # Check length
        if len(session_id) < 10 or len(session_id) > 100:
            return False
        
        return True
    
    async def check_rate_limit(self, client_ip: str, endpoint: str) -> tuple[bool, dict]:
        """Check rate limit for client IP and endpoint"""
        # This would integrate with the RateLimiter class
        # Implementation depends on your specific rate limiting strategy
        return True, {}

# Global instances
config_manager = ConfigurationManager()
performance_optimizer = PerformanceOptimizer(config_manager.get("REDIS", "URL"))
async_task_manager = AsyncTaskManager()
security_optimizer = SecurityOptimizer(config_manager)

# Utility decorators
def async_timed(operation_name: str = None):
    """Decorator to time async operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            async with performance_optimizer.performance_monitor(name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def cached(ttl: int = 3600, key_builder: Callable = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            return await performance_optimizer.cached_operation(
                cache_key, func, ttl, *args, **kwargs
            )
        return wrapper
    return decorator

def circuit_breaker(service_name: str, failure_threshold: int = 5, timeout: int = 60):
    """Circuit breaker decorator"""
    return performance_optimizer.circuit_breaker(service_name, failure_threshold, timeout)

# Configuration validation
def validate_configuration():
    """Validate all configuration settings"""
    errors = []
    
    # Check required API keys
    required_keys = ["AMADEUS_API_KEY", "AMADEUS_API_SECRET"]
    for key in required_keys:
        if not config_manager.get("AMADEUS", key.split("_")[-1]):
            errors.append(f"Missing required configuration: {key}")
    
    # Validate numeric configurations
    numeric_configs = [
        ("API", "PORT", 1, 65535),
        ("PERFORMANCE", "MAX_SEARCH_RESULTS", 1, 100),
        ("REDIS", "CONVERSATION_TTL", 60, 604800)  # 1 minute to 7 days
    ]
    
    for section, key, min_val, max_val in numeric_configs:
        value = config_manager.get(section, key)
        if value is not None and not (min_val <= value <= max_val):
            errors.append(f"Invalid {section}.{key}: {value} (must be {min_val}-{max_val})")
    
    return errors

# Startup optimization
async def optimize_startup():
    """Optimize application startup"""
    logger.info("🚀 Starting performance optimizations...")
    
    # Validate configuration
    config_errors = validate_configuration()
    if config_errors:
        for error in config_errors:
            logger.error(f"❌ Configuration error: {error}")
        raise ValueError("Invalid configuration")
    
    # Start background workers
    if config_manager.is_feature_enabled("BACKGROUND_TASKS"):
        await async_task_manager.start_workers(3)
    
    # Warm up caches
    await warmup_caches()
    
    logger.info("✅ Performance optimizations complete")

async def warmup_caches():
    """Warm up caches with frequently used data"""
    try:
        # Cache common airport codes
        common_airports = ["BOM", "DEL", "BLR", "MAA", "COK", "HYD", "CCU", "AMD"]
        
        # This would be replaced with actual cache warming logic
        logger.info(f"🔥 Warming up caches for {len(common_airports)} airports")
        
        # Pre-load spell check data
        logger.info("🔥 Pre-loading spell check dictionaries")
        
    except Exception as e:
        logger.error(f"Cache warmup failed: {e}")

# Cleanup function
async def cleanup_optimizations():
    """Cleanup optimization resources"""
    logger.info("🧹 Cleaning up optimization resources...")
    
    try:
        await async_task_manager.shutdown()
        await performance_optimizer.cleanup()
        logger.info("✅ Optimization cleanup complete")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# Export key components
__all__ = [
    'PerformanceOptimizer',
    'ConfigurationManager', 
    'RateLimiter',
    'HealthMonitor',
    'DatabaseOptimizer',
    'APIOptimizer',
    'AsyncTaskManager',
    'SecurityOptimizer',
    'config_manager',
    'performance_optimizer',
    'async_task_manager',
    'security_optimizer',
    'async_timed',
    'cached',
    'circuit_breaker',
    'optimize_startup',
    'cleanup_optimizations'
]