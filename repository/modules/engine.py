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

from .tools import get_tools, is_route_query, extract_destination, get_search_engine
from .prompts import pharmacy_prompt
import time
logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    user_query: str = ""
    context: str = ""
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    current_tool: str = ""
    final_answer: str = ""

class PharmacyBot():
    """
    Enhanced PharmacyBot with LangGraph tools for routing between knowledge base and metro directions
    """
    def __init__(self,
                 llm_model_name = "gpt-3.5-turbo",) -> None:
        """
        """
        logger.info(f"Initializing PharmacyBot with model: {llm_model_name}")
        self.llm = ChatOpenAI(
            model=llm_model_name,
            temperature=0.7,
            max_tokens=500,  # Limit response length
            request_timeout=30  # Add timeout
        )
        self.tools = get_tools()
        self.search_engine = get_search_engine()
        self._build_graph()
    
    def _build_graph(self):
        """
        Build the LangGraph with tool routing logic
        """
        logger.info("Initializing LLM graph")
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("use_metro_tool", self._use_metro_tool)
        workflow.add_node("use_knowledge_tool", self._use_knowledge_tool)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add conditional edges
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
        logger.info("LLM Graph initialized successfully")
    
    def _route_query(self, state: AgentState) -> AgentState:
        """
        Route the query to appropriate tool based on content
        """
        user_query = state.user_query
        
        # Check if it's a route query
        if is_route_query(user_query):
            destination = extract_destination(user_query)
            if destination:
                state.current_tool = "metro_route_tool"
                state.tool_results["destination"] = destination
                return state
        
        # Default to knowledge base
        state.current_tool = "knowledge_base_tool"
        return state
    
    def _should_use_metro_tool(self, state: AgentState) -> str:
        """
        Determine which tool to use based on the current_tool field
        """
        if state.current_tool == "metro_route_tool":
            return "metro"
        else:
            return "knowledge"
    
    def _use_metro_tool(self, state: AgentState) -> AgentState:
        """
        Use the metro route tool
        """
        try:
            metro_tool = next(tool for tool in self.tools if tool.name == "metro_route_tool")
            destination = state.tool_results.get("destination", "")
            
            result = metro_tool._run(destination)
            state.tool_results["metro_result"] = result
            
            # Add context for response generation
            if result.get("success"):
                route = result["route"]
                context = f"Metro route from KAFD Metro Station to {destination}:\n"
                context += f"Duration: {route['duration']}\n"
                context += f"Distance: {route['distance']}\n"
                context += f"Summary: {route['summary']}\n\n"
                context += "Steps:\n"
                for i, step in enumerate(route['steps'][:3], 1):  # Limit to first 3 steps
                    context += f"{i}. {step['instruction']} ({step['duration']})\n"
                state.context = context
            else:
                state.context = f"Could not find metro route to {destination}. {result.get('suggestion', '')}"
                
        except Exception as e:
            logger.error(f"Error in metro tool: {e}")
            state.context = "Sorry, I couldn't get metro directions at the moment."
        
        return state
    
    def _use_knowledge_tool(self, state: AgentState) -> AgentState:
        """
        Use the knowledge base tool
        """
        try:
            knowledge_tool = next(tool for tool in self.tools if tool.name == "knowledge_base_tool")
            result = knowledge_tool._run(state.user_query)
            state.tool_results["knowledge_result"] = result
            
            # Add context for response generation
            if result.get("success"):
                results = result["results"]
                context = "Relevant information:\n"
                for i, doc in enumerate(results[:3], 1):  # Limit to first 3 results
                    context += f"{i}. {doc}\n\n"
                state.context = context
            else:
                state.context = result.get('suggestion', 'No relevant information found.')
                
        except Exception as e:
            logger.error(f"Error in knowledge tool: {e}")
            state.context = "Sorry, I couldn't search the knowledge base at the moment."
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """
        Generate the final response using the prompt template
        """
        try:
            # Prepare the prompt with context
            prompt_input = {
                "user_query": state.user_query,
                "context": state.context,
                "messages": state.messages[-3:] if len(state.messages) >= 3 else state.messages
            }
            
            # Generate response
            response = self.llm.invoke(pharmacy_prompt.format(**prompt_input))
            state.final_answer = response.content
            
            logger.info("LLM response generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.final_answer = "I'm sorry, I encountered an error while processing your request. Please try again."
        
        return state
    
    def process_query(self, user_query: str, messages: List[Any] = None) -> str:
        """
        Process a user query and return the response using direct tool routing
        """
        start_time = time.time()
        logger.info(f"=== Starting query processing for: {user_query[:50]}... ===")
        
        try:
            # Step 1: Query classification and routing
            step1_start = time.time()
            logger.info("Step 1: Starting query classification...")
            
            if is_route_query(user_query):
                destination = extract_destination(user_query)
                logger.info(f"Query classified as ROUTE query. Destination: {destination}")
                
                if destination:
                    # Step 2: Metro route tool execution
                    step2_start = time.time()
                    logger.info("Step 2: Starting metro route tool execution...")
                    
                    metro_tool = next(tool for tool in self.tools if tool.name == "metro_route_tool")
                    result = metro_tool._run(destination)
                    
                    step2_time = time.time() - step2_start
                    logger.info(f"Step 2: Metro route tool completed in {step2_time:.2f}s")
                    
                    if result.get("success"):
                        route = result["route"]
                        context = f"Metro route from KAFD Metro Station to {destination}:\n"
                        context += f"Duration: {route['duration']}\n"
                        context += f"Distance: {route['distance']}\n"
                        context += f"Summary: {route['summary']}\n\n"
                        context += "Steps:\n"
                        for i, step in enumerate(route['steps'][:3], 1):
                            context += f"{i}. {step['instruction']} ({step['duration']})\n"
                    else:
                        context = f"Could not find metro route to {destination}. {result.get('suggestion', '')}"
                else:
                    context = "I couldn't determine the destination. Please specify where you want to go."
                
                step1_time = time.time() - step1_start
                logger.info(f"Step 1: Route query processing completed in {step1_time:.2f}s")
            else:
                # Step 2: Knowledge base tool execution
                step2_start = time.time()
                logger.info("Step 2: Starting knowledge base tool execution...")
                
                knowledge_tool = next(tool for tool in self.tools if tool.name == "knowledge_base_tool")
                result = knowledge_tool._run(user_query)
                
                step2_time = time.time() - step2_start
                logger.info(f"Step 2: Knowledge base tool completed in {step2_time:.2f}s")
                
                if result.get("success"):
                    results = result["results"]
                    context = "Relevant information:\n"
                    for i, doc in enumerate(results[:5], 1):  # Show up to 5 results
                        context += f"{i}. {doc}\n\n"
                else:
                    context = result.get('suggestion', 'No relevant information found.')
                
                step1_time = time.time() - step1_start
                logger.info(f"Step 1: Knowledge query processing completed in {step1_time:.2f}s")
            
            # Step 3: LLM response generation
            step3_start = time.time()
            logger.info("Step 3: Starting LLM response generation...")
            
            # Limit context length for faster processing
            limited_context = context[:1000] if len(context) > 1000 else context
            limited_history = messages[-2:] if messages and len(messages) >= 2 else []  # Reduce history
            
            prompt_input = {
                "user_query": user_query,
                "context": limited_context,
                "history": limited_history,
                "lang": "English"
            }
            
            response = self.llm.invoke(pharmacy_prompt.format(**prompt_input))
            
            step3_time = time.time() - step3_start
            logger.info(f"Step 3: LLM response generation completed in {step3_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"=== Total query processing time: {total_time:.2f}s ===")
            logger.info(f"Performance breakdown:")
            logger.info(f"  - Step 1 (Query classification): {step1_time:.2f}s ({(step1_time/total_time)*100:.1f}%)")
            logger.info(f"  - Step 2 (Tool execution): {step2_time:.2f}s ({(step2_time/total_time)*100:.1f}%)")
            logger.info(f"  - Step 3 (LLM generation): {step3_time:.2f}s ({(step3_time/total_time)*100:.1f}%)")
            
            return response.content
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error in process_query after {total_time:.2f}s: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."

# Create a global instance
_global_bot = None

def get_bot() -> PharmacyBot:
    """
    Get or create a global bot instance
    """
    global _global_bot
    if _global_bot is None:
        _global_bot = PharmacyBot()
    return _global_bot