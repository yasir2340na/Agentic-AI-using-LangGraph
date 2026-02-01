import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Review Response Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .sentiment-positive {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    .diagnosis-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
    }
    .response-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4169e1;
    }
    </style>
""", unsafe_allow_html=True)

# Define schemas
class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the review")

class DiagSchema(BaseModel):
    issueType: Literal['UX', 'Performance', 'Bug', 'Other'] = Field(description="The type of issue mentioned in the review")
    tone: Literal['angry', 'sad', 'neutral', 'happy', 'other'] = Field(description="The tone of the review")
    urgency: Literal['low', 'medium', 'high'] = Field(description="The urgency of the issue mentioned in the review")

class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response: str

# Initialize model
@st.cache_resource
def load_model():
    model = ChatGroq(model="openai/gpt-oss-120b")
    return model

@st.cache_resource
def create_workflow():
    model = load_model()
    structured_model = model.with_structured_output(SentimentSchema)
    structured_model2 = model.with_structured_output(DiagSchema)
    
    # Define node functions
    def find_sentiment(state: ReviewState) -> ReviewState:
        prompt = f"Determine if the following review is positive or negative: {state['review']}"
        sentiment = structured_model.invoke(prompt).sentiment
        return {'sentiment': sentiment}
    
    def check_sentiment(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
        if state['sentiment'] == 'positive':
            return 'positive_response'
        else:
            return 'run_diagnosis'
    
    def positive_response(state: ReviewState):
        prompt = f"Generate a positive response with warm thanks to the following review. Review: \n{state['review']}"
        res = model.invoke(prompt)
        return {'response': res.text}
    
    def run_diagnosis(state: ReviewState):
        prompt = f"Analyze the following negative review and provide a diagnosis in terms of issue type, tone, and urgency. Review: \n{state['review']}"
        diag = structured_model2.invoke(prompt)
        return {'diagnosis': diag.model_dump()}
    
    def negative_response(state: ReviewState):
        diagnosis = state['diagnosis']
        prompt = f"""Generate a sympathetic response addressing the issues mentioned in the following review. Review: {state['review']} and '{diagnosis['issueType']}' issue with a '{diagnosis['tone']}' tone and '{diagnosis['urgency']}' urgency."""
        res = model.invoke(prompt)
        return {'response': res.text}
    
    # Build the graph
    graph = StateGraph(ReviewState)
    graph.add_node('find_sentiment', find_sentiment)
    graph.add_node('positive_response', positive_response)
    graph.add_node('run_diagnosis', run_diagnosis)
    graph.add_node('negative_response', negative_response)
    
    graph.add_edge(START, 'find_sentiment')
    graph.add_conditional_edges('find_sentiment', check_sentiment)
    graph.add_edge('positive_response', END)
    graph.add_edge('run_diagnosis', 'negative_response')
    graph.add_edge('negative_response', END)
    
    workflow = graph.compile()
    return workflow

# Main app
st.title("📝 Review Response Assistant")
st.markdown("An AI-powered system that analyzes customer reviews and generates appropriate responses.")

# Sidebar
with st.sidebar:
    st.header("About this App")
    st.info("""
    This app uses a LangGraph workflow to:
    1. Analyze review sentiment
    2. For positive reviews: Generate warm thanks
    3. For negative reviews: Diagnose issues and provide sympathetic responses
    """)
    st.divider()
    st.markdown("**Model:** ChatGroq (openai/gpt-oss-120b)")

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input Review")
    review_text = st.text_area(
        "Enter a customer review:",
        placeholder="Type or paste a customer review here...",
        height=150,
        label_visibility="collapsed"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_button = st.button("🚀 Analyze Review", use_container_width=True, type="primary")
    with col_btn2:
        clear_button = st.button("🔄 Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()

with col2:
    st.subheader("📊 Analysis Results")
    
    if analyze_button and review_text.strip():
        with st.spinner("🔄 Processing review..."):
            try:
                workflow = create_workflow()
                initial_state = {'review': review_text}
                final_state = workflow.invoke(initial_state)
                
                # Display Sentiment
                sentiment = final_state['sentiment']
                sentiment_color = "positive" if sentiment == "positive" else "negative"
                sentiment_emoji = "😊" if sentiment == "positive" else "😞"
                
                st.markdown(f"""
                <div class="sentiment-{sentiment_color}">
                    <h4>{sentiment_emoji} Sentiment: <strong>{sentiment.upper()}</strong></h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Diagnosis if negative
                if sentiment == "negative":
                    diagnosis = final_state['diagnosis']
                    st.markdown("""
                    <div class="diagnosis-box">
                        <h4>🔍 Issue Diagnosis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.metric("Issue Type", diagnosis['issueType'])
                    with col_d2:
                        st.metric("Tone", diagnosis['tone'].title())
                    with col_d3:
                        st.metric("Urgency", diagnosis['urgency'].title())
                
                # Display Generated Response
                st.markdown("""
                <div class="response-box">
                    <h4>💬 Generated Response</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(final_state['response'])
                
                # Copy to clipboard button
                st.divider()
                col_copy1, col_copy2 = st.columns([3, 1])
                with col_copy2:
                    st.button("📋 Copy Response", use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error processing review: {str(e)}")
    elif analyze_button and not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze")
    else:
        st.info("👆 Enter a review and click 'Analyze Review' to get started")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    Built with Streamlit • LangGraph • LangChain
</div>
""", unsafe_allow_html=True)
