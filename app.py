import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google.generativeai.errors import APIError

# --- Configuration and Initialization ---

# Azure App Service will securely provide the API key via Application Settings.
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Hardcoded model name as requested
MODEL_TO_USE = 'gemini-2.5-flash' 

if not API_KEY:
    # In a cloud environment, print a fatal message and allow the host 
    # (like Gunicorn/Azure) to handle the startup failure.
    print("FATAL: GOOGLE_API_KEY environment variable not found. The application cannot start.")

try:
    # Only configure if the key is available to avoid runtime errors on startup
    if API_KEY:
        genai.configure(api_key=API_KEY)
        # The GenerativeModel instance explicitly uses gemini-2.5-flash
        model = genai.GenerativeModel(MODEL_TO_USE)
    else:
        # Create a placeholder for the model if the API key is missing
        model = None 
except Exception as e:
    # Handle configuration failure if key is present but invalid
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    model = None

# Initialize Flask app
# The name must be 'app' for Azure/Gunicorn to easily find it.
app = Flask(__name__)

# Configure CORS (use specific origins in production)
CORS(app, resources={r"/api/*": {"origins": "*", "supports_credentials": True}})

# --- Core LLM Logic ---

def execute_watch_tonight(query: str):
    """
    Skill: What to Watch Tonight - Generates streaming recommendations using the Gemini API.
    """
    if not model:
        raise Exception("AI model failed to initialize due to missing or invalid API key.")

    prompt = f"""
You are my entertainment assistant. Your mission: help me find what to watch tonight.

USER CONTEXT: {query if query.strip() else "No extra constraints provided."}

TASK:
1) SEARCH
• Browse current trends (last 7 days) on Netflix, Apple TV+, Paramount+, Disney+, Prime Video, Zee5, Airtel XStream using Google Search.
• Identify the most popular or freshly released movies and series as well as best rated.
• Make sure they’re available in the user’s country (assume user is in India unless specified).
• Only suggest content with good reviews and a significant number of ratings (IMDb, Rotten Tomatoes, press or audience reviews).

2) CLASSIFY THE RECOMMENDATIONS
Create 3 sections, with 2–3 suggestions in each:
A. Easy to follow → light shows/movies, comedies, feel-good content.
B. To savor → dramas, thrillers, cinematic works, well-crafted stories.
C. Adventure → action, sci-fi, fantasy, epics.

FOR EACH RECOMMENDATION:
• Title (year)
• Rating (e.g., IMDb or Rotten Tomatoes)
• Platform / Indicate if it’s a new release or trending
• Runtime (movie) or number of available episodes (series)
• Genre
• 2–3 sentence pitch

End with an engaging phrase like: “Want further suggestions in another genre or mood?”
"""
    # Using Google Search grounding tool for up-to-date streaming data
    response = model.generate_content(prompt, tools=[{"google_search": {}}])
    return response.text

# --- Health Check Route ---

@app.route('/check', methods=['GET'])
def check():
    """
    Simple health check route used to confirm the backend server is running and accessible.
    """
    status_code = 200
    if not model:
        # Return a 503 status if the AI model failed to initialize
        status_code = 503
        message = "backend is running, but AI model failed to initialize."
    else:
        message = "backend is running"

    return jsonify({
        'status': 'ok' if status_code == 200 else 'error', 
        'message': message, 
        'model': MODEL_TO_USE
    }), status_code

# --- Main API Route ---

@app.route('/api/execute', methods=['POST'])
def execute():
    # 1. Basic Input Validation and Parsing
    if not model:
        # Fail fast if the model isn't configured
        return jsonify({'error': 'AI service not initialized. Check API key configuration.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload.'}), 400

    # Query is optional for this agent, so we default to an empty string
    query = data.get('query', '')
    
    # 2. Execution and Specific Error Handling
    try:
        # Call the core logic (new function)
        result = execute_watch_tonight(query)
        
        return jsonify({'success': True, 'result': result})
        
    except APIError as e:
        # Handle specific Gemini API errors (e.g., rate limits, invalid prompts)
        print(f"Gemini API Error: {e}")
        # Return a 503 Service Unavailable for AI service issues
        return jsonify({'error': f'AI Service Unavailable or request error: {e}'}), 503
        
    except Exception as e:
        # Catch all other unexpected errors (e.g., network, internal logic)
        print(f"Internal Server Error: {e}")
        return jsonify({'error': 'An unexpected internal server error occurred.'}), 500

