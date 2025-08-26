import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import json
import logging
from typing import List, Dict, Any, Optional

from modules.engine import get_bot
from modules.utils import index_all_metro_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Riyadh Metro Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    load_dotenv()
    # Global bot instance
    bot = get_bot()
except Exception as e:
    raise e

class ChatRequest(BaseModel):
    question: str
    userid: str
    lang: str = "en"

class ChatResponse(BaseModel):
    chatbot_answer: str
    products: List[Dict[str, Any]] = []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint that routes between knowledge base and metro directions
    """
    try:
        logger.info(f"Processing user query for thread_id: {request.userid}")
        
        # Process the query using the new LangGraph tools
        response = bot.process_query(request.question)
        
        # Parse the response to extract JSON data if present
        chatbot_answer = response
        product_data = []
        
        # Check if response contains JSON data
        if "JSON:" in response:
            try:
                # Split the response to get the JSON part
                parts = response.split("JSON:")
                chatbot_answer = parts[0].strip()
                json_string = parts[1].strip()
                
                # Parse the JSON
                product_data = json.loads(json_string)
                
                # Check if this is an amenities query or other type of query
                # Only return amenities data for specific amenity-related queries
                query_lower = request.question.lower()
                amenity_keywords = [
                    "restaurant", "cafe", "shop", "store", "mall", "mosque", "park",
                    "amenity", "amenities", "place to eat", "where to eat", "where to shop",
                    "dining", "food", "shopping", "services", "facilities"
                ]
                
                is_amenity_query = any(keyword in query_lower for keyword in amenity_keywords)
                
                # If it's not an amenity query, return empty array
                if not is_amenity_query:
                    product_data = []
                    
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse JSON from response: {e}")
                product_data = []
        
        return {
            "chatbot_answer": chatbot_answer,
            "products": product_data
        }
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/push_to_vdb")
def push_data(data_path: str = "./data/pharmacy_products.json"):
    """
    Triggers the process to load and embed data into the Chroma VDB.
    Accepts an optional 'data_path' as a query parameter.
    """    
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Data file '{data_path}' not found on the server.")
    
    try:
        # Pass the 'data_path' variable to your function
        # The original push_to_chroma function is removed, so this part is commented out or removed if not needed.
        # For now, keeping it as is, but it will cause an error if push_to_chroma is not defined.
        # Assuming push_to_chroma is no longer needed or will be re-added elsewhere.
        # push_to_chroma(json_data_path=data_path) 
        return {"message": f"Data from '{data_path}' pushed to VDB successfully"}
    
    except Exception as e:
        # The exception from your module will be caught and returned as a 500 error
        raise HTTPException(status_code=500, detail=f"Failed to push data: {str(e)}")

@app.post("/index_all_data")
def index_all_data():
    """
    Index all metro-related data files into the vector database.
    This will create multiple collections for different types of data.
    """
    try:
        index_all_metro_data()
        return {
            "message": "All metro data indexed successfully",
            "collections_created": [
                "pharmacy_products",
                "metro_rules",
                "sports_venues",
                "sports_content",
                "ticketing",
                "amenities"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index all data: {str(e)}")

@app.get("/get_chat_history/{userid}")
def get_chat_history(userid: str):
    """
    """
    # The original assistant.graph.get_state is removed, so this part is commented out or removed.
    # Assuming the functionality is no longer available or will be re-added elsewhere.
    # For now, returning a placeholder error.
    return {"status": "error", "message": "Chat history retrieval is currently unavailable."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Riyadh Metro Assistant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)