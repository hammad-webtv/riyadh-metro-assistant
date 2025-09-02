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
import threading
import asyncio

from .tools import get_tools, get_search_engine
from .prompts import pharmacy_prompt

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history with automatic cleanup after inactivity"""
    
    def __init__(self, cleanup_interval: int = 30):
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = cleanup_interval
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def add_message(self, user_id: str, message: str, is_user: bool = True):
        """Add a message to the conversation history"""
        with self.lock:
            current_time = time.time()
            
            if user_id not in self.conversations:
                self.conversations[user_id] = {
                    'messages': [],
                    'last_activity': current_time
                }
            
            # Add message
            if is_user:
                self.conversations[user_id]['messages'].append(HumanMessage(content=message))
            else:
                self.conversations[user_id]['messages'].append(AIMessage(content=message))
            
            # Update last activity
            self.conversations[user_id]['last_activity'] = current_time
    
    def get_conversation_history(self, user_id: str) -> List[Any]:
        """Get conversation history for a user"""
        with self.lock:
            if user_id in self.conversations:
                # Update last activity when accessing
                self.conversations[user_id]['last_activity'] = time.time()
                return self.conversations[user_id]['messages']
            return []
    
    def get_conversation_history_text(self, user_id: str) -> str:
        """Get conversation history as formatted text for API responses"""
        with self.lock:
            if user_id in self.conversations:
                # Update last activity when accessing
                self.conversations[user_id]['last_activity'] = time.time()
                messages = self.conversations[user_id]['messages']
                
                # Format messages as text
                history_text = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        history_text.append(f"User: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        history_text.append(f"Assistant: {msg.content}")
                
                return "\n".join(history_text)
            return ""
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for a specific user"""
        with self.lock:
            if user_id in self.conversations:
                del self.conversations[user_id]
    
    def _cleanup_loop(self):
        """Background thread to clean up old conversations"""
        while True:
            try:
                current_time = time.time()
                to_remove = []
                
                with self.lock:
                    for user_id, conv_data in self.conversations.items():
                        if current_time - conv_data['last_activity'] > self.cleanup_interval:
                            to_remove.append(user_id)
                    
                    # Remove old conversations
                    for user_id in to_remove:
                        del self.conversations[user_id]
                        logger.info(f"Cleaned up conversation for user: {user_id} after {self.cleanup_interval}s of inactivity")
                
                # Sleep for 10 seconds before next cleanup check
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(10)
    
    def get_active_conversations_count(self) -> int:
        """Get count of active conversations"""
        with self.lock:
            return len(self.conversations)

# Define the state schema
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    user_query: str = ""
    context: str = ""
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    current_tool: str = ""
    final_answer: str = ""
    user_id: str = ""

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
    """Enhanced MetroBot with LLM-based routing using LangGraph and conversation management"""
    
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
        self.conversation_manager = ConversationManager(cleanup_interval=30)
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
        
        self.graph = workflow.compile()
        logger.info("LLM Graph with intelligent routing initialized successfully")
    
    def _route_query_llm(self, state: AgentState) -> AgentState:
        """Route query using LLM-based decision making"""
        step1_start = time.time()
        user_query = state.user_query
        
        logger.info(f"Step 1: Starting LLM-based query routing for: {user_query[:50]}...")
        
        try:
            # Get conversation history for context
            conversation_history = self.conversation_manager.get_conversation_history(state.user_id)
            
            # Create routing prompt with history context
            routing_input = {
                "user_query": user_query,
                "conversation_history": str(conversation_history[-3:]) if conversation_history else "No previous conversation"
            }
            
            # Get routing decision from LLM
            routing_response = self.llm.invoke(ROUTING_PROMPT.format(**routing_input))
            routing_content = routing_response.content
            
            # Parse routing decision
            try:
                import json
                routing_decision = json.loads(routing_content)
                tool_choice = routing_decision.get("tool", "knowledge_base_tool")
                reasoning = routing_decision.get("reasoning", "Default to knowledge base tool")
                
                logger.info(f"LLM routing decision: {tool_choice} - {reasoning}")
                
                if tool_choice == "metro_route_tool":
                    state.current_tool = "metro"
                else:
                    state.current_tool = "knowledge"
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse routing decision: {e}, defaulting to knowledge base tool")
                state.current_tool = "knowledge"
                
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}, defaulting to knowledge base tool")
            state.current_tool = "knowledge"
        
        step1_time = time.time() - step1_start
        logger.info(f"Step 1: LLM-based query routing completed in {step1_time:.2f}s")
        
        return state
    
    def _should_use_metro_tool(self, state: AgentState) -> str:
        """Determine which tool to use based on routing decision"""
        return state.current_tool
    
    def _use_metro_tool(self, state: AgentState) -> AgentState:
        """Use metro route tool for direction queries"""
        step2_start = time.time()
        user_query = state.user_query
        
        logger.info(f"Step 2: Starting metro route tool execution")
        
        try:
            # Extract destination from query
            # This is a simplified implementation - you might want to enhance this
            if "to" in user_query.lower():
                parts = user_query.lower().split("to")
                if len(parts) > 1:
                    destination = parts[1].strip()
                    # Here you would call your metro route tool
                    # For now, we'll set a placeholder context
                    state.context = f"Route information for destination: {destination}. Metro route tool would be called here."
                else:
                    state.context = "Please specify a destination for route planning."
            else:
                state.context = "Please specify where you want to go for route planning."
                
        except Exception as e:
            logger.error(f"Error in metro route tool: {e}")
            state.context = "I encountered an error while planning your route. Please try rephrasing your question."
        
        step2_time = time.time() - step2_start
        logger.info(f"Step 2: Metro route tool completed in {step2_time:.2f}s")
        
        return state
    
    def _use_knowledge_tool(self, state: AgentState) -> AgentState:
        """Use knowledge base tool for comprehensive search"""
        step2_start = time.time()
        user_query = state.user_query
        
        logger.info(f"Step 2: Starting knowledge base tool execution")
        
        try:
            # Perform comprehensive search
            results = self.search_engine.query_search_all_collections(user_query, num_retrievals=8)
            
            if results:
                # Combine results into context
                context_parts = []
                for i, result in enumerate(results, 1):
                    if result.startswith("[AMENITIES]"):
                        context_parts.append(f"[AMENITIES] {result}")
                    elif result.startswith("[METRO_RULES]"):
                        context_parts.append(f"[METRO_RULES] {result}")
                    else:
                        context_parts.append(result)
                
                context = "\n\n".join(context_parts)
                state.context = context
                
                # Log retrieved context
                logger.info(f"=== RETRIEVED CONTEXT FOR QUERY: {user_query[:50]}... ===")
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
        
        step2_time = time.time() - step2_start
        logger.info(f"Step 2: Knowledge base tool completed in {step2_time:.2f}s")
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using LLM with conversation history"""
        step3_start = time.time()
        user_query = state.user_query
        context = state.context
        user_id = state.user_id
        
        logger.info(f"Step 3: Starting LLM response generation")
        
        # Get conversation history
        conversation_history = self.conversation_manager.get_conversation_history(user_id)
        
        # Limit context length for performance
        limited_context = context[:3000] if len(context) > 3000 else context
        
        # Format conversation history for prompt
        if conversation_history:
            # Take last 5 messages to avoid context overflow
            recent_history = conversation_history[-5:]
            history_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in recent_history])
        else:
            history_text = ""
        
        # Create prompt input
        prompt_input = {
            "context": limited_context,
            "history": history_text,
            "user_query": user_query,
            "lang": "en"
        }
        
        try:
            # Generate response
            response = self.llm.invoke(pharmacy_prompt.format(**prompt_input))
            state.final_answer = response.content
            
            # Add messages to conversation history
            self.conversation_manager.add_message(user_id, user_query, is_user=True)
            self.conversation_manager.add_message(user_id, response.content, is_user=False)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.final_answer = "I'm sorry, I encountered an error while processing your request. Please try again."
        
        step3_time = time.time() - step3_start
        logger.info(f"Step 3: LLM response generation completed in {step3_time:.2f}s")
        
        return state
    
    def process_query(self, user_query: str, user_id: str = "default") -> dict:
        """Process a user query and return the response with additional data"""
        start_time = time.time()
        logger.info(f"=== Starting LLM-based query processing for: {user_query[:50]}... ===")
        
        try:
            # Create initial state with user_id
            state = AgentState(
                user_query=user_query,
                messages=[HumanMessage(content=user_query)],
                user_id=user_id
            )
            
            # Execute the graph
            result = self.graph.invoke(state)
            
            # Get the final answer - handle both dict and object results
            if isinstance(result, dict):
                final_answer = result.get("final_answer", "I'm sorry, I encountered an error while processing your request. Please try again.")
                map_list = result.get("tool_results", {}).get("map_list", [])
            else:
                final_answer = result.final_answer
                map_list = getattr(result, 'tool_results', {}).get("map_list", [])
            
            total_time = time.time() - start_time
            logger.info(f"=== Total LLM-based query processing time: {total_time:.2f}s ===")
            
            return {
                "answer": final_answer,
                "map_list": map_list
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}", exc_info=True)
            return {
                "answer": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "map_list": []
            }
    
    def get_conversation_stats(self) -> dict:
        """Get conversation statistics"""
        return {
            "active_conversations": self.conversation_manager.get_active_conversations_count(),
            "cleanup_interval": self.conversation_manager.cleanup_interval
        }
    
    def clear_user_conversation(self, user_id: str):
        """Clear conversation history for a specific user"""
        self.conversation_manager.clear_conversation(user_id)

# Global bot instance
_global_bot = None

def get_bot():
    """Get or create global bot instance"""
    global _global_bot
    if _global_bot is None:
        _global_bot = MetroBot()
    return _global_bot