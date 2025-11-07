"""
Multi-Mode Conversational Chatbot with Short-Term Memory
Uses LangChain + LangGraph for session management
"""

from typing import Dict, List, Literal, TypedDict, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rag.query import RAGQueryPipeline


# ============================================================================
# State Definition for LangGraph
# ============================================================================

class ChatbotState(TypedDict):
    """State for chatbot conversation"""
    messages: List  # Conversation history (stored by LangGraph checkpointer)
    user_input: str  # Current user input
    mode: str  # Current mode (learn, hint, quiz, eli5, custom)
    custom_prompt: Optional[str]  # Custom mode instructions
    retrieved_context: str  # RAG context
    response: str  # Generated response


# ============================================================================
# Prompt Templates Manager
# ============================================================================

class PromptManager:
    """Manages prompt templates for different modes"""
    
    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}
    
    def load_prompt(self, mode: str) -> str:
        """Load prompt template for a specific mode"""
        if mode in self._cache:
            return self._cache[mode]
        
        # Map mode to file
        prompt_file = self.prompts_dir / f"{mode}.txt"
        
        if not prompt_file.exists():
            # Fallback to default
            prompt_file = self.prompts_dir / "default.txt"
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        self._cache[mode] = prompt_template
        return prompt_template
    
    def format_prompt(self, mode: str, context: str, question: str, 
                     history: str = "", custom_instructions: str = "") -> str:
        """Format prompt with variables"""
        template = self.load_prompt(mode)
        
        # Replace placeholders
        prompt = template.replace("{context}", context)
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{history}", history)
        prompt = prompt.replace("{custom_instructions}", custom_instructions)
        
        return prompt


# ============================================================================
# Multi-Mode Chatbot with RAG
# ============================================================================

class CognitoDroidChatbot:
    """
    Multi-mode conversational chatbot with RAG and short-term memory
    Uses LangGraph's native memory system via checkpointer
    """
    
    def __init__(self, vector_store_path: str = None):
        """Initialize chatbot with RAG pipeline and memory"""
        print("Initializing Cognito-Droid Chatbot...")
        
        # Use existing RAG pipeline instead of duplicating initialization
        self.rag_pipeline = RAGQueryPipeline(vector_store_path)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Current mode and custom instructions
        self.current_mode = "learn"
        self.custom_instructions = ""
        
        # Build LangGraph workflow with checkpointer for memory
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        
        print("Chatbot initialized successfully!")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph state machine for chatbot"""
        workflow = StateGraph(ChatbotState)
        
        # Define nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("update_memory", self._update_memory_node)
        
        # Define edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)
        
        # Compile workflow with checkpointer for memory persistence
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _retrieve_context_node(self, state: ChatbotState) -> ChatbotState:
        """Node: Retrieve context from RAG"""
        user_input = state["user_input"]
        
        # Use RAG pipeline's retriever
        docs = self.rag_pipeline.retriever.invoke(user_input)
        
        # Format context with source attribution
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{doc.page_content}"
            )
        
        state["retrieved_context"] = "\n\n".join(context_parts)
        return state
    
    def _generate_response_node(self, state: ChatbotState) -> ChatbotState:
        """Node: Generate response using LLM with mode-specific prompt"""
        mode = state["mode"]
        context = state["retrieved_context"]
        user_input = state["user_input"]
        custom_prompt = state.get("custom_prompt", "")
        
        # Get conversation history from state (managed by LangGraph checkpointer)
        messages = state.get("messages", [])
        
        # Format history as string (last 10 messages)
        history_str = ""
        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                history_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"
        
        # Build prompt based on mode
        prompt = self.prompt_manager.format_prompt(
            mode=mode,
            context=context,
            question=user_input,
            history=history_str,
            custom_instructions=custom_prompt
        )
        
        # Use RAG pipeline's LLM
        response = self.rag_pipeline.llm.invoke(prompt)
        
        # Extract text from response
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        state["response"] = response_text
        return state
    
    def _update_memory_node(self, state: ChatbotState) -> ChatbotState:
        """Node: Update conversation memory in state"""
        user_input = state["user_input"]
        response = state["response"]
        
        # Get existing messages from state
        messages = state.get("messages", [])
        
        # Add new messages to state (LangGraph checkpointer will persist these)
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=response))
        
        state["messages"] = messages
        return state
    
    def set_mode(self, mode: Literal["learn", "hint", "quiz", "eli5", "custom"]):
        """Change chatbot mode"""
        self.current_mode = mode
        print(f"Mode changed to: {mode}")
    
    def set_custom_mode(self, instructions: str):
        """Set custom mode with user-defined instructions"""
        self.current_mode = "custom"
        self.custom_instructions = instructions
        print(f"Custom mode activated with instructions: {instructions[:50]}...")
    
    def chat(self, user_input: str, thread_id: str = "default") -> str:
        """
        Chat with the bot
        
        Args:
            user_input: User's message
            thread_id: Session thread ID (for multi-user support)
        
        Returns:
            Bot's response
        """
        # Get current state from checkpointer (if exists)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get existing state or start fresh
        try:
            current_state = self.workflow.get_state(config)
            existing_messages = current_state.values.get("messages", []) if current_state else []
        except:
            existing_messages = []
        
        # Initialize state with existing messages
        initial_state = {
            "messages": existing_messages,
            "user_input": user_input,
            "mode": self.current_mode,
            "custom_prompt": self.custom_instructions,
            "retrieved_context": "",
            "response": ""
        }
        
        # Run workflow (checkpointer automatically persists state)
        result = self.workflow.invoke(initial_state, config)
        
        return result["response"]
    
    def clear_memory(self, thread_id: str = "default"):
        """Clear conversation history for a thread"""
        try:
            self.checkpointer.delete_thread(thread_id)
            print("Conversation memory cleared.")
        except Exception as e:
            print(f"Error clearing memory: {e}")
    
    def get_conversation_history(self, thread_id: str = "default") -> List:
        """Get current conversation history from checkpointer"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.workflow.get_state(config)
            if state:
                return state.values.get("messages", [])
        except:
            pass
        return []
    
    def print_history(self, thread_id: str = "default"):
        """Print conversation history"""
        history = self.get_conversation_history(thread_id)
        print("\n" + "="*60)
        print("CONVERSATION HISTORY")
        print("="*60)
        for msg in history:
            if isinstance(msg, HumanMessage):
                print(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"Assistant: {msg.content}")
        print("="*60 + "\n")


# ============================================================================
# Main Function for Testing
# ============================================================================

def main():
    """Interactive testing of chatbot"""
    try:
        chatbot = CognitoDroidChatbot()
        
        print("\n" + "="*60)
        print("COGNITO-DROID CHATBOT - INTERACTIVE MODE")
        print("="*60)
        print("Commands:")
        print("  /mode <learn|hint|quiz|eli5> - Change mode")
        print("  /custom <instructions> - Set custom mode")
        print("  /history - Show conversation history")
        print("  /clear - Clear conversation history")
        print("  /quit - Exit")
        print("="*60 + "\n")
        
        thread_id = "default"
        
        while True:
            user_input = input(f"[{chatbot.current_mode.upper()}] You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                
                if cmd == "/quit":
                    print("Goodbye!")
                    break
                
                elif cmd == "/mode" and len(cmd_parts) > 1:
                    mode = cmd_parts[1].lower()
                    if mode in ["learn", "hint", "quiz", "eli5"]:
                        chatbot.set_mode(mode)
                    else:
                        print(f"Invalid mode. Choose: learn, hint, quiz, eli5")
                
                elif cmd == "/custom" and len(cmd_parts) > 1:
                    instructions = cmd_parts[1]
                    chatbot.set_custom_mode(instructions)
                
                elif cmd == "/history":
                    chatbot.print_history(thread_id)
                
                elif cmd == "/clear":
                    chatbot.clear_memory(thread_id)
                
                else:
                    print("Unknown command.")
                
                continue
            
            # Chat
            response = chatbot.chat(user_input, thread_id)
            print(f"Assistant: {response}\n")
    
    except FileNotFoundError:
        print("\n❌ Error: Vector store not found!")
        print("Please run document ingestion first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
