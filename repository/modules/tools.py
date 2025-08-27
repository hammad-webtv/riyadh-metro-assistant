import logging
import time
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import os
from dotenv import load_dotenv
import googlemaps
from datetime import datetime
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import hashlib

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Google Maps client initialization
api_key = os.getenv("GOOGLE_MAPS_API_KEY")
if api_key:
    try:
        gmaps = googlemaps.Client(key=api_key)
        GOOGLE_MAPS_AVAILABLE = True
        logger.info("Google Maps API initialized successfully")
    except Exception as e:
        logger.warning(f"Google Maps API initialization failed: {e}")
        GOOGLE_MAPS_AVAILABLE = False
        gmaps = None
else:
    logger.warning("Google Maps API key not found. Metro routing will not be available.")
    GOOGLE_MAPS_AVAILABLE = False
    gmaps = None

class MetroRouteInput(BaseModel):
    destination: str = Field(description="The destination location to route to from KAFD Metro Station")

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="The user's question to search in the knowledge base")

class MetroRouteTool(BaseTool):
    name: str = "metro_route_tool"
    description: str = "Get metro directions from KAFD Metro Station to a destination. Use this when users ask 'how to get to X' or 'directions to X' or 'route to X'."
    args_schema: type = MetroRouteInput
    
    def _run(self, destination: str) -> Dict[str, Any]:
        """Get metro route from KAFD Metro Station to destination"""
        start_time = time.time()
        logger.info(f"=== Starting metro route tool for destination: {destination} ===")
        
        if not GOOGLE_MAPS_AVAILABLE:
            logger.info("Google Maps API not available")
            return {
                "error": "Google Maps API is not available. Please check your API key configuration.",
                "suggestion": "You can ask about metro rules, amenities, or other information instead."
            }
        
        origin = "KAFD Metro Station"
        
        # Add "Riyadh" to destination if not already present
        if "Riyadh" not in destination:
            destination_with_city = destination + " Riyadh"
        else:
            destination_with_city = destination
            
        try:
            maps_start = time.time()
            logger.info("Making Google Maps API call...")
            
            directions = gmaps.directions(
                origin=origin,
                destination=destination_with_city,
                mode="transit",
                departure_time=datetime.now(),
                transit_mode='bus|train',
                alternatives=True
            )
            
            maps_time = time.time() - maps_start
            logger.info(f"Google Maps API call completed in {maps_time:.2f}s")

            if not directions:
                return {
                    "error": f"No metro routes found from {origin} to {destination}",
                    "suggestion": "Try asking about metro rules, nearby amenities, or other information."
                }

            # Sort by shortest duration
            fastest_route = min(directions, key=lambda x: x['legs'][0]['duration']['value'])

            route_details = {
                'summary': fastest_route.get('summary', ''),
                'duration': fastest_route['legs'][0]['duration']['text'],
                'distance': fastest_route['legs'][0]['distance']['text'],
                'steps': [],
                'origin': origin,
                'destination': destination
            }

            for step in fastest_route['legs'][0]['steps']:
                step_details = {
                    'instruction': step['html_instructions'],
                    'distance': step['distance']['text'],
                    'duration': step['duration']['text'],
                    'transit_details': {}
                }
                
                if 'transit_details' in step:
                    transit_info = step['transit_details']
                    line_info = transit_info.get('line', {})
                    step_details['transit_details'] = {
                        'departure_stop': transit_info['departure_stop']['name'],
                        'departure_time': transit_info['departure_time']['text'],
                        'arrival_stop': transit_info['arrival_stop']['name'],
                        'arrival_time': transit_info['arrival_time']['text'],
                        'line_name': line_info.get('name', 'N/A'),
                        'line_color': line_info.get('color', 'Not available')
                    }
                
                route_details['steps'].append(step_details)

            total_time = time.time() - start_time
            logger.info(f"=== Metro route tool completed in {total_time:.2f}s ===")
            
            return {
                "success": True,
                "data": route_details,
                "processing_time": f"{total_time:.2f}s"
            }

        except Exception as e:
            logger.error(f"Error in metro route tool: {e}", exc_info=True)
            return {
                "error": f"Exception occurred: {str(e)}",
                "suggestion": "Try asking about metro rules, amenities, or other information instead."
            }

class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base_tool"
    description: str = "Search ALL collections (metro rules and amenities) for comprehensive information about fares, stations, rules, amenities, and general metro information."
    args_schema: type = KnowledgeBaseInput
    
    def __init__(self):
        super().__init__()
        self._search_engine = get_search_engine()
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Search ALL collections for comprehensive information"""
        start_time = time.time()
        logger.info(f"=== Starting comprehensive search for query: {query[:50]}... ===")
        
        try:
            # Search ALL collections
            search_start = time.time()
            results = self._search_engine.query_search_all_collections(query, num_retrievals=8)
            search_time = time.time() - search_start
            
            logger.info(f"Comprehensive search completed in {search_time:.2f}s")
            
            if results:
                logger.info(f"=== Comprehensive search completed successfully in {search_time:.2f}s ===")
                logger.info(f"  - Search operation: {search_time:.2f}s ({(search_time/(time.time()-start_time))*100:.1f}%)")
                logger.info(f"  - Retrieved {len(results)} results from all collections")
                logger.info(f"  - First result preview: {results[0][:100]}...")
                
                return {
                    "success": True,
                    "results": results,
                    "processing_time": f"{search_time:.2f}s"
                }
            else:
                logger.info(f"=== Comprehensive search completed (no results) in {search_time:.2f}s ===")
                return {
                    "success": False,
                    "suggestion": "I couldn't find specific information about that. Try asking about metro fares, stations, amenities, or general rules.",
                    "processing_time": f"{search_time:.2f}s"
                }
                
        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Search error: {str(e)}",
                "suggestion": "Try asking about metro fares, stations, amenities, or general rules."
            }

class MetroSearchEngine:
    """Comprehensive search engine that searches ALL collections"""
    
    def __init__(self, persist_directory="../repository/chromadb"):
        self.persist_directory = persist_directory
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')
        self._search_cache = {}
        self.collections = ["metro_rules", "amenities"]  # Search both collections
    
    def query_search_all_collections(self, query: str, num_retrievals: int = 8) -> List[str]:
        """Search ALL collections and return combined results"""
        logger.info(f"=== Starting comprehensive search across all collections for query: {query[:50]}... ===")
        
        all_results = []
        
        try:
            # Search each collection
            for collection_name in self.collections:
                try:
                    logger.info(f"Searching collection: {collection_name}")
                    vectordb = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory=self.persist_directory
                    )
                    
                    # Get results from this collection
                    collection_results = vectordb.similarity_search(query, k=num_retrievals)
                    collection_contents = [result.page_content for result in collection_results]
                    
                    # Add collection identifier to each result
                    for content in collection_contents:
                        all_results.append(f"[{collection_name.upper()}] {content}")
                    
                    logger.info(f"Found {len(collection_contents)} results from {collection_name}")
                    
                except Exception as e:
                    logger.warning(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort and limit results
            all_results = all_results[:num_retrievals * 2]  # Allow more results since we're searching multiple collections
            
            logger.info(f"Comprehensive search completed successfully, found {len(all_results)} total results")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}")
            return []
    
    def query_search_docs(self, query: str, num_retrievals: int = 8) -> List[str]:
        """Legacy method - now calls comprehensive search"""
        return self.query_search_all_collections(query, num_retrievals)

# Global instances
_global_search_engine = None

def get_search_engine():
    """Get or create global search engine instance"""
    global _global_search_engine
    if _global_search_engine is None:
        _global_search_engine = MetroSearchEngine()
    return _global_search_engine

def get_tools():
    """Get all available tools"""
    return [
        MetroRouteTool(),
        KnowledgeBaseTool()
    ]