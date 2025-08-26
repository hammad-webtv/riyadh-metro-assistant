import logging
import time
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import os
from dotenv import load_dotenv
import googlemaps
from datetime import datetime
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

import hashlib
from functools import lru_cache

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
        """
        Get metro route from KAFD Metro Station to destination
        """
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
            # Time the Google Maps API call
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
            logger.info(f"=== Metro route tool completed successfully in {total_time:.2f}s ===")
            logger.info(f"  - Google Maps API: {maps_time:.2f}s ({(maps_time/total_time)*100:.1f}%)")
            logger.info(f"  - Route processing: {(total_time-maps_time):.2f}s ({((total_time-maps_time)/total_time)*100:.1f}%)")
            return {"success": True, "route": route_details}

        except Exception as e:
            total_time = time.time() - start_time
            logger.exception(f"Error in metro route tool after {total_time:.2f}s")
            return {
                "error": f"Error getting metro route: {str(e)}",
                "suggestion": "Try asking about metro rules, nearby amenities, or other information."
            }

class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base_tool"
    description: str = "Search the knowledge base for information about metro rules, amenities, sports venues, and other general information. Use this for questions about metro rules, nearby places, sports, etc."
    args_schema: type = KnowledgeBaseInput
    
    def __init__(self, search_engine=None):
        super().__init__()
        self._search_engine = search_engine or MetroSearchEngine()
    
    def _run(self, query: str) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information
        """
        start_time = time.time()
        logger.info(f"=== Starting knowledge base tool for query: {query[:50]}... ===")
        
        try:
            search_start = time.time()
            results = self._search_engine.query_search_docs(query, num_retrievals=5)
            search_time = time.time() - search_start
            logger.info(f"Knowledge base search completed in {search_time:.2f}s")
            
            if not results:
                total_time = time.time() - start_time
                logger.info(f"=== Knowledge base tool completed (no results) in {total_time:.2f}s ===")
                return {
                    "error": "No relevant information found in the knowledge base.",
                    "suggestion": "Try rephrasing your question or ask about metro routes instead."
                }
            
            total_time = time.time() - start_time
            logger.info(f"=== Knowledge base tool completed successfully in {total_time:.2f}s ===")
            logger.info(f"  - Search operation: {search_time:.2f}s ({(search_time/total_time)*100:.1f}%)")
            return {
                "success": True,
                "results": results,
                "query": query
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.exception(f"Error in knowledge base tool after {total_time:.2f}s")
            return {
                "error": f"Error searching knowledge base: {str(e)}",
                "suggestion": "Try asking about metro routes or rephrase your question."
            }

class MetroSearchEngine:
    """
    Comprehensive search engine for Riyadh Metro Assistant that searches across all collections
    and returns the most relevant results based on query type and content.
    """
    
    def __init__(self, persist_directory="./chromadb"):
        self.persist_directory = persist_directory
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self._search_cache = {}
        
        # Define collection priorities for different query types
        self.collection_priorities = {
            "metro": ["metro_rules", "amenities"],
            "sports": ["sports_venues", "sports_content"],
            "pharmacy": ["pharmacy_products"],
            "general": ["amenities", "metro_rules"]
        }
        
        # Limit to top 2 collections for maximum speed
        self.max_collections = 2
    
    def query_search_docs(self, query: str, num_retrievals: int = 5) -> List[str]:
        """
        Main search function that intelligently searches across all collections
        and returns the most relevant results using parallel processing.
        """
        start_time = time.time()
        logger.info(f"=== Starting search engine for query: {query[:50]}... ===")
        
        # Check cache first
        cache_start = time.time()
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        cache_key = f"{query_hash}_{num_retrievals}"
        
        if cache_key in self._search_cache:
            cache_time = time.time() - cache_start
            logger.info(f"Cache hit! Returning cached results in {cache_time:.2f}s")
            return self._search_cache[cache_key]
        
        cache_time = time.time() - cache_start
        logger.info(f"Cache miss. Cache check took {cache_time:.2f}s")
        
        logging.info(f"Starting parallel comprehensive search for query: {query}")
        
        try:
            # Determine query type and select appropriate collections
            classify_start = time.time()
            query_type = self._classify_query(query)
            collections_to_search = self.collection_priorities.get(query_type, self.collection_priorities["general"])
            classify_time = time.time() - classify_start
            logger.info(f"Query classification completed in {classify_time:.2f}s. Type: {query_type}")
            
            # Limit to most relevant collections for faster response
            if len(collections_to_search) > self.max_collections:
                collections_to_search = collections_to_search[:self.max_collections]
            
            all_results = []
            
            # Use batch search for maximum speed
            search_start = time.time()
            all_results = self._batch_search_collections(collections_to_search, query, num_retrievals)
            search_time = time.time() - search_start
            logger.info(f"Batch search across {len(collections_to_search)} collections completed in {search_time:.2f}s")
            
            # Sort results by relevance and return top results
            sorted_results = self._rank_results(all_results, query)
            final_results = sorted_results[:num_retrievals]
            
            # Extract page content from results
            page_contents = [result.page_content for result in final_results]
            
            # Cache the results
            cache_store_start = time.time()
            self._search_cache[cache_key] = page_contents
            cache_store_time = time.time() - cache_store_start
            logger.info(f"Results cached in {cache_store_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"=== Search engine completed in {total_time:.2f}s ===")
            logger.info(f"  - Cache check: {cache_time:.2f}s ({(cache_time/total_time)*100:.1f}%)")
            logger.info(f"  - Query classification: {classify_time:.2f}s ({(classify_time/total_time)*100:.1f}%)")
            logger.info(f"  - Batch search: {search_time:.2f}s ({(search_time/total_time)*100:.1f}%)")
            logger.info(f"  - Cache storage: {cache_store_time:.2f}s ({(cache_store_time/total_time)*100:.1f}%)")
            
            logging.info(f"Parallel search completed successfully, found {len(page_contents)} relevant results")
            return page_contents
            
        except Exception as e:
            total_time = time.time() - start_time
            logging.error(f"An error occurred during parallel query search after {total_time:.2f}s: {e}", exc_info=True)
            return []
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the query type to determine which collections to search
        """
        query_lower = query.lower()
        
        # Metro and transportation related queries
        metro_keywords = [
            "metro", "train", "station", "line", "route", "transport", "travel", "commute",
            "kafd", "king abdullah", "financial district", "ticket", "fare", "pass",
            "platform", "schedule", "frequency", "accessibility", "elevator", "ramp",
            "family", "singles", "first class", "vip", "gold", "etiquette", "rules",
            "safety", "security", "emergency", "intercom", "staff"
        ]
        
        # Sports and venue related queries
        sports_keywords = [
            "football", "soccer", "stadium", "team", "match", "game", "al hilal", "al nassr",
            "al shabab", "al riyadh", "kingdom arena", "al-awwal park", "prince turki",
            "prince faisal", "king saud university", "granadia", "sports city"
        ]
        
        # Pharmacy and health related queries
        pharmacy_keywords = [
            "medicine", "drug", "pill", "tablet", "capsule", "syrup", "cream", "ointment",
            "pain", "fever", "headache", "cold", "flu", "allergy", "vitamin", "supplement",
            "prescription", "otc", "over the counter", "pharmacy", "pharmacist"
        ]
        
        # Count keyword matches
        metro_score = sum(1 for keyword in metro_keywords if keyword in query_lower)
        sports_score = sum(1 for keyword in sports_keywords if keyword in query_lower)
        pharmacy_score = sum(1 for keyword in pharmacy_keywords if keyword in query_lower)
        
        # Return the category with the highest score
        scores = {
            "metro": metro_score,
            "sports": sports_score,
            "pharmacy": pharmacy_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        else:
            return "general"
    
    def _search_collection(self, collection_name: str, query: str, num_retrievals: int) -> List[Any]:
        """
        Search a specific collection and return results
        """
        try:
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            # Search with metadata filtering for better relevance
            results = vectordb.similarity_search(
                query, 
                k=num_retrievals
            )
            
            return results
            
        except Exception as e:
            logging.warning(f"Error searching collection {collection_name}: {e}")
            return []
    
    def _batch_search_collections(self, collections: List[str], query: str, num_retrievals: int) -> List[Any]:
        """
        Batch search multiple collections simultaneously for maximum speed
        """
        all_results = []
        
        # Create all vector databases at once
        vectordbs = {}
        for collection_name in collections:
            try:
                vectordb = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory
                )
                vectordbs[collection_name] = vectordb
            except Exception as e:
                logging.warning(f"Error creating vectordb for {collection_name}: {e}")
                continue
        
        # Perform all searches in parallel
        with ThreadPoolExecutor(max_workers=len(vectordbs)) as executor:
            future_to_collection = {
                executor.submit(vectordb.similarity_search, query, k=num_retrievals): collection_name
                for collection_name, vectordb in vectordbs.items()
            }
            
            for future in as_completed(future_to_collection):
                collection_name = future_to_collection[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logging.warning(f"Error in batch search for {collection_name}: {e}")
                    continue
        
        return all_results
    
    def _rank_results(self, results: List[Any], query: str) -> List[Any]:
        """
        Rank results by relevance to the query
        """
        if not results:
            return []
        
        # Simple ranking based on content relevance
        # In a production system, you might want to use more sophisticated ranking
        ranked_results = []
        
        for result in results:
            content = result.page_content.lower()
            query_terms = query.lower().split()
            
            # Calculate relevance score based on term frequency
            relevance_score = sum(1 for term in query_terms if term in content)
            
            # Add metadata-based scoring
            if hasattr(result, 'metadata'):
                metadata = result.metadata
                if metadata.get('type') == 'metro_rules':
                    relevance_score += 2  # Boost metro rules for general queries
                elif metadata.get('type') == 'amenities':
                    relevance_score += 1  # Slight boost for amenities
            
            ranked_results.append((relevance_score, result))
        
        # Sort by relevance score (descending) and return results
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in ranked_results]

def is_route_query(query: str) -> bool:
    """
    Determine if a query is asking for directions/route
    """
    route_keywords = [
        "how to get to", "directions to", "route to", "way to", "how do i get to",
        "how can i get to", "directions", "route", "navigate to", "travel to",
        "go to", "reach", "find my way to", "get directions"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in route_keywords)

def extract_destination(query: str) -> Optional[str]:
    """
    Extract destination from route query
    """
    route_keywords = [
        "how to get to", "directions to", "route to", "way to", "how do i get to",
        "how can i get to", "navigate to", "travel to", "go to", "reach",
        "find my way to", "get directions to"
    ]
    
    query_lower = query.lower()
    
    for keyword in route_keywords:
        if keyword in query_lower:
            # Extract text after the keyword
            start_idx = query_lower.find(keyword) + len(keyword)
            destination = query[start_idx:].strip()
            
            # Clean up the destination
            destination = destination.replace("?", "").replace(".", "").strip()
            
            if destination:
                return destination
    
    return None

# Legacy function for backward compatibility
def query_search_docs(query: str, num_retrievals: int = 5) -> List[str]:
    """
    Legacy search function that delegates to the new MetroSearchEngine
    """
    search_engine = MetroSearchEngine()
    return search_engine.query_search_docs(query, num_retrievals)

# Create a global search engine instance for better performance
_global_search_engine = None

def get_search_engine() -> MetroSearchEngine:
    """
    Get or create a global search engine instance
    """
    global _global_search_engine
    if _global_search_engine is None:
        _global_search_engine = MetroSearchEngine()
    return _global_search_engine

def get_tools() -> List[BaseTool]:
    """
    Get all available tools for LangGraph
    """
    search_engine = get_search_engine()
    return [
        MetroRouteTool(),
        KnowledgeBaseTool(search_engine=search_engine)
    ]