import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import json
import logging
from typing import List, Dict, Any

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
    """Main chat endpoint"""
    try:
        logger.info(f"Processing user query for thread_id: {request.userid}")
        
        # Process the query with user_id for conversation history
        response_data = bot.process_query(request.question, user_id=request.userid)
        
        # Extract the response and map_list
        response = response_data.get("answer", "I'm sorry, I encountered an error while processing your request. Please try again.")
        map_data = response_data.get("map_list", [])
        
        # Parse the response to extract JSON data if present
        chatbot_answer = response
        product_data = []
        
        # Check if response contains JSON data
        if "JSON:" in response:
            try:
                parts = response.split("JSON:")
                chatbot_answer = parts[0].strip()
                json_string = parts[1].strip()
                
                product_data = json.loads(json_string)
                
                # Check if this is an amenities query
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
            "products": product_data,
            "map_list": map_data
        }
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/stats")
async def get_conversation_stats():
    """Get conversation statistics"""
    try:
        stats = bot.get_conversation_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Clear conversation history for a specific user"""
    try:
        bot.clear_user_conversation(user_id)
        return {"message": f"Conversation history cleared for user: {user_id}"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{user_id}/history")
async def get_conversation_history(user_id: str):
    """Get conversation history for a specific user"""
    try:
        # This would require adding a method to get conversation history
        # For now, we'll return a placeholder
        return {"message": "Conversation history endpoint not yet implemented"}
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index_all_data")
async def index_data():
    """Index all data files"""
    try:
        index_all_metro_data()
        return {
            "message": "All metro data indexed successfully",
            "collections_created": ["metro_rules", "amenities"]
        }
    except Exception as e:
        logger.error(f"Error indexing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Riyadh Metro Assistant"
    }

@app.post("/debug_search")
async def debug_search(request: ChatRequest):
    """Debug endpoint to see what's being retrieved"""
    try:
        from modules.tools import MetroSearchEngine
        
        search_engine = MetroSearchEngine(persist_directory="../repository/chromadb")
        logger.info(f"Debug search: Created search engine for query: {request.question}")
        
        results = search_engine.query_search_docs(request.question, num_retrievals=5)
        logger.info(f"Debug search: Got {len(results)} results")
        
        return {
            "query": request.question,
            "num_results": len(results),
            "results": results[:10]  # Return first 10 results to see both collections
        }
    except Exception as e:
        logger.error(f"Error in debug search: {e}", exc_info=True)
        return {
            "error": str(e),
            "query": request.question,
            "results": []
        }

@app.get("/debug_collections")
async def debug_collections():
    """Debug endpoint to check what collections exist and their sizes"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="../repository/chromadb")
        collections = client.list_collections()
        
        collection_info = []
        for collection in collections:
            count = collection.count()
            collection_info.append({
                "name": collection.name,
                "count": count
            })
        
        return {
            "collections": collection_info,
            "total_collections": len(collections)
        }
    except Exception as e:
        logger.error(f"Error in debug collections: {e}")
        return {
            "error": str(e),
            "collections": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)