import logging
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .tools import get_tools, get_search_engine
from .prompts import pharmacy_prompt

logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    user_query: str = ""
    context: str = ""
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    current_tool: str = ""
    final_answer: str = ""

# LLM-based routing prompt
ROUTING_PROMPT = PromptTemplate(
    input_variables=["user_query"],
    template="""You are an intelligent router for a Riyadh Metro Assistant. Your job is to determine which tool should handle the user's query.

Available tools:
1. metro_route_tool - Use ONLY for explicit route/direction requests like "how to get to X", "directions to X", "route to X"
2. knowledge_base_tool - Use for ALL other queries including amenities, fares, metro information, general questions

User Query: {user_query}

Analyze the query and respond with ONLY a JSON object in this exact format:
{{
    "tool": "metro_route_tool" or "knowledge_base_tool",
    "reasoning": "Brief explanation of why this tool was chosen"
}}

Examples:
- "How to get to Riyadh Zoo" → {{"tool": "metro_route_tool", "reasoning": "User is asking for explicit directions/route"}}
- "What is the fare for a 7-day pass?" → {{"tool": "knowledge_base_tool", "reasoning": "User is asking about metro information"}}
- "What restaurants are near KAFD?" → {{"tool": "knowledge_base_tool", "reasoning": "User is asking about amenities (comprehensive search will find this)"}}
- "Tell me about the metro system" → {{"tool": "knowledge_base_tool", "reasoning": "User is asking for general metro information"}}
- "Directions to King Fahd Park" → {{"tool": "metro_route_tool", "reasoning": "User is asking for explicit directions"}}

IMPORTANT: Use knowledge_base_tool for MOST queries since it searches ALL collections (metro rules + amenities). Only use metro_route_tool for explicit route requests.

Respond with JSON only:"""
)

class MetroBot():
    """Enhanced MetroBot with LLM-based routing using LangGraph"""
    
    def __init__(self, llm_model_name="gpt-3.5-turbo") -> None:
        logger.info(f"Initializing MetroBot with model: {llm_model_name}")
        self.llm = ChatOpenAI(
            model=llm_model_name,
            temperature=0.7,
            max_tokens=500,
            request_timeout=30
        )
        self.tools = get_tools()
        self.search_engine = get_search_engine()
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph with LLM-based routing logic"""
        logger.info("Initializing LLM graph with intelligent routing")
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query_llm)
        workflow.add_node("use_metro_tool", self._use_metro_tool)
        workflow.add_node("use_knowledge_tool", self._use_knowledge_tool)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add conditional edges based on LLM routing decision
        workflow.add_conditional_edges(
            "route_query",
            self._should_use_metro_tool,
            {
                "metro": "use_metro_tool",
                "knowledge": "use_knowledge_tool"
            }
        )
        
        # Add edges to response generation
        workflow.add_edge("use_metro_tool", "generate_response")
        workflow.add_edge("use_knowledge_tool", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("route_query")
        
        # Compile the graph
        self.graph = workflow.compile()
        logger.info("LLM Graph with intelligent routing initialized successfully")
    
    def _route_query_llm(self, state: AgentState) -> AgentState:
        """Use LLM to intelligently route the query to appropriate tool"""
        user_query = state.user_query
        
        logger.info(f"Step 1: Starting LLM-based query routing for: {user_query[:50]}...")
        
        try:
            # Use LLM to determine which tool to use
            routing_response = self.llm.invoke(ROUTING_PROMPT.format(user_query=user_query))
            
            # Parse the JSON response - handle potential parsing issues
            import json
            try:
                routing_decision = json.loads(routing_response.content)
                tool_choice = routing_decision.get("tool", "knowledge_base_tool")
                reasoning = routing_decision.get("reasoning", "Default to knowledge base")
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, defaulting to knowledge base")
                tool_choice = "knowledge_base_tool"
                reasoning = "JSON parsing failed, defaulting to knowledge base"
            
            logger.info(f"LLM routing decision: {tool_choice} - {reasoning}")
            
            # Set the tool choice in state
            state.current_tool = tool_choice
            
            # If it's a metro route tool, try to extract destination
            if tool_choice == "metro_route_tool":
                destination = self._extract_destination_llm(user_query)
                if destination:
                    state.tool_results["destination"] = destination
                    logger.info(f"Extracted destination: {destination}")
                else:
                    logger.warning("Could not extract destination, falling back to knowledge base")
                    state.current_tool = "knowledge_base_tool"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
            # Fallback to knowledge base tool
            state.current_tool = "knowledge_base_tool"
            return state
    
    def _extract_destination_llm(self, query: str) -> str:
        """Use LLM to extract destination from route query"""
        try:
            extraction_prompt = PromptTemplate(
                input_variables=["query"],
                template="""Extract the destination from this route query. Return ONLY the destination name, nothing else.

Query: {query}

Examples:
- "How to get to Riyadh Zoo" → "Riyadh Zoo"
- "Directions to King Fahd Park" → "King Fahd Park"
- "Route to Al Nakheel Mall" → "Al Nakheel Mall"

Destination:"""
            )
            
            response = self.llm.invoke(extraction_prompt.format(query=query))
            destination = response.content.strip()
            
            # Clean up the destination
            if destination and len(destination) > 0:
                return destination
            
        except Exception as e:
            logger.error(f"Error extracting destination: {e}")
        
        return ""
    
    def _should_use_metro_tool(self, state: AgentState) -> str:
        """Determine which tool to use based on LLM routing decision"""
        if state.current_tool == "metro_route_tool":
            return "metro"
        else:
            return "knowledge"
    
    def _use_metro_tool(self, state: AgentState) -> AgentState:
        """Execute metro route tool"""
        step2_start = time.time()
        user_query = state.user_query
        destination = state.tool_results.get("destination", "")
        
        logger.info(f"Step 2: Starting metro route tool execution for destination: {destination}")
        
        # Find the metro route tool
        metro_tool = None
        for tool in self.tools:
            if tool.name == "metro_route_tool":
                metro_tool = tool
                break
        
        if metro_tool and destination:
            try:
                result = metro_tool._run(destination)
                state.tool_results["metro_result"] = result
                
                if result.get("success"):
                    route_data = result.get("data", {})
                    context = f"Route from KAFD Metro Station to {destination}:\n"
                    context += f"Duration: {route_data.get('duration', 'N/A')}\n"
                    context += f"Distance: {route_data.get('distance', 'N/A')}\n"
                    context += f"Summary: {route_data.get('summary', 'N/A')}\n"
                    
                    # Add step details
                    steps = route_data.get('steps', [])
                    for i, step in enumerate(steps[:3], 1):  # Limit to first 3 steps
                        context += f"Step {i}: {step.get('instruction', 'N/A')}\n"
                    
                    state.context = context
                else:
                    state.context = result.get('suggestion', 'No route information available.')
                
            except Exception as e:
                logger.error(f"Error in metro route tool: {e}")
                state.context = "I encountered an error while getting route information. Please try asking about metro rules or amenities instead."
        else:
            state.context = "I couldn't determine the destination. Please specify where you want to go from KAFD Metro Station."
        
        step2_time = time.time() - step2_start
        logger.info(f"Step 2: Metro route tool completed in {step2_time:.2f}s")
        
        return state
    
    def _use_knowledge_tool(self, state: AgentState) -> AgentState:
        """Execute knowledge base tool"""
        step2_start = time.time()
        user_query = state.user_query
        
        logger.info(f"Step 2: Starting knowledge base tool execution")
        
        # Find the knowledge base tool
        knowledge_tool = None
        for tool in self.tools:
            if tool.name == "knowledge_base_tool":
                knowledge_tool = tool
                break
        
        if knowledge_tool:
            try:
                result = knowledge_tool._run(user_query)
                
                if result.get("success"):
                    results = result["results"]
                    context = "Relevant information:\n"
                    for i, doc in enumerate(results[:8], 1):  # Limit to first 8 results
                        context += f"{i}. {doc}\n\n"
                    state.context = context
                    
                    # Log the retrieved context for debugging
                    logger.info(f"=== RETRIEVED CONTEXT FOR QUERY: {user_query} ===")
                    logger.info(f"Number of results: {len(results)}")
                    for i, doc in enumerate(results[:3], 1):  # Log first 3 results
                        logger.info(f"Result {i}: {doc[:300]}...")  # Log first 300 chars of each result
                    logger.info(f"=== END RETRIEVED CONTEXT ===")
                else:
                    context = result.get('suggestion', 'No relevant information found.')
                    logger.info(f"=== NO RESULTS FOUND FOR QUERY: {user_query} ===")
                    state.context = context
                
            except Exception as e:
                logger.error(f"Error in knowledge base tool: {e}")
                state.context = "I encountered an error while searching for information. Please try rephrasing your question."
        else:
            state.context = "Knowledge base search is not available."
        
        step2_time = time.time() - step2_start
        logger.info(f"Step 2: Knowledge base tool completed in {step2_time:.2f}s")
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using LLM"""
        step3_start = time.time()
        user_query = state.user_query
        context = state.context
        
        logger.info(f"Step 3: Starting LLM response generation")
        
        # Limit context length for performance
        limited_context = context[:3000] if len(context) > 3000 else context
        
        # Create prompt input
        prompt_input = {
            "context": limited_context,
            "history": "",  # Simplified - no conversation history for now
            "user_query": user_query,
            "lang": "en"
        }
        
        try:
            # Generate response
            response = self.llm.invoke(pharmacy_prompt.format(**prompt_input))
            state.final_answer = response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.final_answer = "I'm sorry, I encountered an error while processing your request. Please try again."
        
        step3_time = time.time() - step3_start
        logger.info(f"Step 3: LLM response generation completed in {step3_time:.2f}s")
        
        return state
    
    def process_query(self, user_query: str) -> str:
        """Process a user query and return the response"""
        start_time = time.time()
        logger.info(f"=== Starting LLM-based query processing for: {user_query[:50]}... ===")
        
        try:
            # Create initial state
            state = AgentState(
                user_query=user_query,
                messages=[HumanMessage(content=user_query)]
            )
            
            # Execute the graph
            result = self.graph.invoke(state)
            
            # Get the final answer - handle both dict and object results
            if isinstance(result, dict):
                final_answer = result.get("final_answer", "I'm sorry, I encountered an error while processing your request. Please try again.")
            else:
                final_answer = result.final_answer
            
            total_time = time.time() - start_time
            logger.info(f"=== Total LLM-based query processing time: {total_time:.2f}s ===")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request. Please try again."

# Global bot instance
_global_bot = None

def get_bot():
    """Get or create global bot instance"""
    global _global_bot
    if _global_bot is None:
        _global_bot = MetroBot()
    return _global_bot