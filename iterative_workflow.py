"""
Iterative Product Description Enhancement Workflow
===================================================
Real-world scenario: E-commerce product description improvement
- Writer LLM (ChatGroq) generates/refines descriptions
- Critic LLM (Cohere) evaluates quality and provides feedback
- Editor LLM (OpenAI/ChatGroq) polishes the final version
- Iterates until quality threshold is met or max iterations reached
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 7  # Out of 10

# State Definition
class ProductDescriptionState(TypedDict):
    product_name: str
    product_features: str
    current_description: str
    feedback: str
    quality_score: int
    iteration: int
    should_continue: bool
    history: list[str]

# Initialize LLMs
writer_llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.7
)

critic_llm = ChatCohere(
    model="command-r-plus",
    temperature=0.3
)

editor_llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.5
)

# Node 1: Writer - Generate or Refine Description
def writer_node(state: ProductDescriptionState) -> ProductDescriptionState:
    """Generate initial description or refine based on feedback"""
    print(f"\n{'='*60}")
    print(f"📝 ITERATION {state['iteration']} - WRITER")
    print(f"{'='*60}")
    
    if state['iteration'] == 1:
        # Initial generation
        prompt = f"""Create a compelling e-commerce product description for:

Product Name: {state['product_name']}
Key Features: {state['product_features']}

Write a persuasive, engaging product description that:
- Highlights key benefits (not just features)
- Uses emotional appeal
- Is concise but informative (150-200 words)
- Includes a call-to-action

Product Description:"""
    else:
        # Refinement based on feedback
        prompt = f"""Improve this product description based on the critic's feedback:

Product Name: {state['product_name']}
Current Description: {state['current_description']}

Critic's Feedback: {state['feedback']}

Create an improved version that addresses all feedback points while maintaining persuasiveness and clarity.

Improved Description:"""
    
    response = writer_llm.invoke([HumanMessage(content=prompt)])
    new_description = response.content.strip()
    
    print(f"\n✍️ Generated Description:\n{new_description[:200]}...")
    
    return {
        'current_description': new_description,
        'history': state['history'] + [f"Iteration {state['iteration']} - Writer: {new_description[:100]}..."]
    }

# Node 2: Critic - Evaluate Quality and Provide Feedback
def critic_node(state: ProductDescriptionState) -> ProductDescriptionState:
    """Evaluate description quality and provide constructive feedback"""
    print(f"\n{'='*60}")
    print(f"🔍 ITERATION {state['iteration']} - CRITIC (Cohere)")
    print(f"{'='*60}")
    
    prompt = f"""You are an expert e-commerce copywriter evaluating product descriptions.

Product Name: {state['product_name']}
Description to Evaluate:
{state['current_description']}

Evaluate this description on these criteria:
1. Persuasiveness (Does it convince customers to buy?)
2. Clarity (Is it easy to understand?)
3. Emotional Appeal (Does it connect with customers?)
4. Feature Coverage (Are key features highlighted?)
5. Call-to-Action (Does it encourage action?)

Provide:
1. Overall Quality Score (1-10)
2. Specific, actionable feedback for improvement
3. What's working well

Format your response EXACTLY as:
SCORE: [number]
FEEDBACK: [your detailed feedback]
STRENGTHS: [what's working well]"""
    
    response = critic_llm.invoke([HumanMessage(content=prompt)])
    evaluation = response.content.strip()
    
    # Parse score
    try:
        score_line = [line for line in evaluation.split('\n') if 'SCORE:' in line][0]
        quality_score = int(score_line.split(':')[1].strip().split()[0])
    except:
        quality_score = 5  # Default if parsing fails
    
    print(f"\n⭐ Quality Score: {quality_score}/10")
    print(f"\n📋 Feedback:\n{evaluation[:300]}...")
    
    return {
        'feedback': evaluation,
        'quality_score': quality_score,
        'history': state['history'] + [f"Iteration {state['iteration']} - Critic Score: {quality_score}/10"]
    }

# Node 3: Decision - Continue or End
def decision_node(state: ProductDescriptionState) -> ProductDescriptionState:
    """Decide whether to continue iterating or finalize"""
    print(f"\n{'='*60}")
    print(f"🤔 DECISION CHECK")
    print(f"{'='*60}")
    
    should_continue = (
        state['quality_score'] < QUALITY_THRESHOLD and 
        state['iteration'] < MAX_ITERATIONS
    )
    
    new_iteration = state['iteration'] + 1 if should_continue else state['iteration']
    
    if should_continue:
        print(f"➡️ Continuing to iteration {new_iteration} (Score: {state['quality_score']}/{QUALITY_THRESHOLD})")
    else:
        reason = "Quality threshold met! 🎉" if state['quality_score'] >= QUALITY_THRESHOLD else "Max iterations reached"
        print(f"✅ Stopping: {reason}")
    
    return {
        'should_continue': should_continue,
        'iteration': new_iteration
    }

# Node 4: Editor - Final Polish
def editor_node(state: ProductDescriptionState) -> ProductDescriptionState:
    """Apply final polish to the description"""
    print(f"\n{'='*60}")
    print(f"✨ FINAL EDITOR POLISH")
    print(f"{'='*60}")
    
    prompt = f"""You are a professional editor. Polish this product description to perfection:

Product Name: {state['product_name']}
Description:
{state['current_description']}

Polish it by:
- Fixing any grammar/spelling issues
- Enhancing readability
- Ensuring professional tone
- Optimizing word choice
- Maintaining persuasiveness

Provide ONLY the polished description, nothing else."""
    
    response = editor_llm.invoke([HumanMessage(content=prompt)])
    final_description = response.content.strip()
    
    print(f"\n💎 Final Polished Description:\n{final_description}")
    
    return {
        'current_description': final_description,
        'history': state['history'] + ["Final Editor Polish Applied"]
    }

# Conditional Edge Function
def should_continue_iteration(state: ProductDescriptionState) -> str:
    """Determine next node based on should_continue flag"""
    if state['should_continue']:
        return "writer"
    else:
        return "editor"

# Build the Graph
def create_product_description_workflow():
    """Create and compile the iterative workflow graph"""
    
    workflow = StateGraph(ProductDescriptionState)
    
    # Add nodes
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("editor", editor_node)
    
    # Add edges
    workflow.add_edge(START, "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "decision")
    
    # Conditional edge - loop back to writer or go to editor
    workflow.add_conditional_edges(
        "decision",
        should_continue_iteration,
        {
            "writer": "writer",
            "editor": "editor"
        }
    )
    
    workflow.add_edge("editor", END)
    
    return workflow.compile()

# Main Execution
def main():
    """Run the iterative product description workflow"""
    
    print("🚀 Starting Iterative Product Description Enhancement Workflow")
    print("="*60)
    
    # Initialize state
    initial_state = {
        'product_name': "EcoBreeze Smart Water Bottle",
        'product_features': """
        - Smart temperature display (LED)
        - Keeps drinks cold for 24hrs, hot for 12hrs
        - BPA-free stainless steel
        - 750ml capacity
        - Leak-proof design
        - UV-C self-cleaning technology
        - Companion mobile app for hydration tracking
        """,
        'current_description': "",
        'feedback': "",
        'quality_score': 0,
        'iteration': 1,
        'should_continue': True,
        'history': []
    }
    
    # Create and run workflow
    workflow = create_product_description_workflow()
    
    try:
        result = workflow.invoke(initial_state)
        
        # Display final results
        print("\n" + "="*60)
        print("🎯 FINAL RESULTS")
        print("="*60)
        print(f"\n📊 Final Quality Score: {result['quality_score']}/10")
        print(f"🔄 Total Iterations: {result['iteration']}")
        print(f"\n📝 Final Product Description:\n")
        print(result['current_description'])
        print("\n" + "="*60)
        print("📜 Workflow History:")
        print("="*60)
        for entry in result['history']:
            print(f"  • {entry}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Make sure you have set up your API keys in .env file:")
        print("  - GROQ_API_KEY")
        print("  - COHERE_API_KEY")
        print("  - OPENAI_API_KEY (optional, can use Groq for editor too)")

if __name__ == "__main__":
    main()
