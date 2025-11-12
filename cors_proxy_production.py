#!/usr/bin/env python3
"""
Production CORS Proxy for Nigeria Pathwise
Proxies requests from frontend to the NAT API server
Optimized for Render deployment
"""
import os
import logging
from typing import Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
AGENT_BASE_URL = os.getenv('AGENT_BASE_URL', 'http://localhost:8000')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'https://pathwise-chi.vercel.app').split(',')
PORT = int(os.getenv('PORT', 3001))

# Configure CORS
CORS(
    app,
    origins=ALLOWED_ORIGINS,
    allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
    methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    supports_credentials=True,
    max_age=3600
)

logger.info(f"CORS enabled for origins: {ALLOWED_ORIGINS}")
logger.info(f"Proxying to: {AGENT_BASE_URL}")


@app.before_request
def log_request():
    """Log incoming requests"""
    logger.info(f"{request.method} {request.path} from {request.remote_addr}")


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    try:
        # Try to ping the agent API
        response = requests.get(f"{AGENT_BASE_URL}/health", timeout=5)
        agent_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        agent_status = "unreachable"
    
    return jsonify({
        "status": "healthy",
        "proxy": "running",
        "agent": agent_status,
        "agent_url": AGENT_BASE_URL
    }), 200


@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        "status": "CORS proxy server is running",
        "agent_url": AGENT_BASE_URL,
        "allowed_origins": ALLOWED_ORIGINS,
        "version": "1.0.0"
    }), 200


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """
    Proxy chat requests to the agent API
    Main endpoint for frontend communication
    """
    if request.method == 'OPTIONS':
        return _handle_preflight()
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Log the request (sanitized)
        logger.info(f"Chat request: {len(data.get('message', ''))} chars")
        
        # Forward to agent API
        url = f"{AGENT_BASE_URL}/chat"
        
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=180  # 3 minutes for LLM responses
        )
        
        # Return the response
        return Response(
            response.content,
            status=response.status_code,
            headers={'Content-Type': 'application/json'}
        )
        
    except requests.exceptions.Timeout:
        logger.error("Request to agent API timed out")
        return jsonify({
            "error": "Request timed out. Please try again.",
            "type": "timeout"
        }), 504
        
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to agent API")
        return jsonify({
            "error": "Unable to connect to AI service. Please try again later.",
            "type": "connection_error"
        }), 503
        
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Proxy error: {str(e)}",
            "type": "proxy_error"
        }), 500


@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_api(path):
    """
    Generic proxy for all API endpoints
    Handles any path under /api/*
    """
    if request.method == 'OPTIONS':
        return _handle_preflight()
    
    url = f"{AGENT_BASE_URL}/{path}"
    
    try:
        logger.info(f"Proxying {request.method} to {url}")
        
        # Prepare request parameters
        kwargs = {
            'timeout': 180,
            'headers': {'Content-Type': 'application/json'}
        }
        
        # Add query parameters
        if request.args:
            kwargs['params'] = request.args
        
        # Add body for POST/PUT
        if request.method in ['POST', 'PUT']:
            if request.is_json:
                kwargs['json'] = request.get_json()
            else:
                kwargs['data'] = request.get_data()
        
        # Make the request
        response = requests.request(
            method=request.method,
            url=url,
            **kwargs
        )
        
        # Return the response
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers)
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return jsonify({
            "error": f"Request failed: {str(e)}",
            "type": "request_error"
        }), 500


def _handle_preflight():
    """Handle CORS preflight requests"""
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(e)}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": str(e)
    }), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Nigeria Pathwise CORS Proxy Starting")
    logger.info(f"📡 Proxying: http://0.0.0.0:{PORT}/api/* → {AGENT_BASE_URL}/*")
    logger.info(f"🌍 Allowed origins: {ALLOWED_ORIGINS}")
    logger.info(f"🔧 Health check: http://0.0.0.0:{PORT}/health")
    logger.info(f"🧪 Test endpoint: http://0.0.0.0:{PORT}/test")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )