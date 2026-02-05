from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define state schema
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    current_input: str
    response: str

# Initialize model
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.7,
        convert_system_message_to_human=True
    )

# Create workflow
def create_chat_workflow():
    model = load_model()
    
    def process_input(state: ChatState) -> ChatState:
        """Add user message to history"""
        messages = state.get('messages', [])
        current_input = state['current_input']
        
        # Add user message
        messages.append({
            'role': 'user',
            'content': current_input
        })
        
        return {'messages': messages}
    
    def generate_response(state: ChatState) -> ChatState:
        """Generate AI response with context"""
        messages = state['messages']
        
        # Convert to LangChain message format
        lc_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                lc_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                lc_messages.append(AIMessage(content=msg['content']))
        
        # Generate response
        response = model.invoke(lc_messages)
        
        # Add AI response to history
        messages.append({
            'role': 'assistant',
            'content': response.content
        })
        
        return {
            'messages': messages,
            'response': response.content
        }
    
    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node('process_input', process_input)
    graph.add_node('generate_response', generate_response)
    
    graph.add_edge(START, 'process_input')
    graph.add_edge('process_input', 'generate_response')
    graph.add_edge('generate_response', END)
    
    return graph.compile()

def main():
    print("=" * 60)
    print("💬 AI CHATBOT WITH MEMORY")
    print("=" * 60)
    print("Model: Google Gemini 3 Flash Preview")
    print("Commands: 'exit' to quit, 'clear' to reset history")
    print("=" * 60)
    print()
    
    # Initialize
    workflow = create_chat_workflow()
    chat_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                chat_history = []
                print("\n✅ History cleared!\n")
                continue
            
            # Process with workflow
            print("AI: ", end="", flush=True)
            
            state = {
                'messages': chat_history.copy(),
                'current_input': user_input
            }
            
            result = workflow.invoke(state)
            chat_history = result['messages']
            
            print(result['response'].text)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

if __name__ == "__main__":
    main()
