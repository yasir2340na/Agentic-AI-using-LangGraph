# Review Response Assistant - Streamlit App

## Overview
This is a Streamlit UI for your conditional review response workflow. It maintains all the same logic and scenario from your LangGraph notebook while providing a professional, user-friendly interface.

## Features
- **Review Input**: Easy text input for customer reviews
- **Sentiment Analysis**: Automatic detection of positive/negative reviews
- **Issue Diagnosis**: For negative reviews, displays issue type, tone, and urgency
- **Response Generation**: AI-generated responses tailored to the sentiment
- **Beautiful UI**: Clean, professional interface with color-coded results

## Installation

### 1. Install Streamlit (if not already installed)
```bash
pip install streamlit
```

### 2. Verify other dependencies are installed
```bash
pip install langchain langchain-google-genai langchain-groq langgraph python-dotenv pydantic
```

## Running the App

### Option 1: From Command Line
```bash
cd "d:\Agentic AI\Agentic-AI-using-LangGraph"
streamlit run app.py
```

### Option 2: From PowerShell (Windows)
```powershell
cd "d:\Agentic AI\Agentic-AI-using-LangGraph"
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## Usage

1. **Enter a Review**: Type or paste a customer review in the input box
2. **Click "Analyze Review"**: The AI will process the review
3. **View Results**:
   - Sentiment (Positive/Negative)
   - For negative reviews: Issue diagnosis with type, tone, and urgency
   - Generated response tailored to the sentiment

## Workflow Logic

The app follows the same conditional workflow as your notebook:

```
Start
  ↓
Find Sentiment (Analyze if positive/negative)
  ↓
Check Sentiment (Conditional branching)
  ├─→ POSITIVE: Generate positive response
  └─→ NEGATIVE: Run diagnosis → Generate sympathetic response
  ↓
End
```

## Important Notes

- ✅ **Same Logic**: Uses the exact same workflow from your notebook
- ✅ **Same Scenario**: Review analysis → Diagnosis (if needed) → Response generation
- ✅ **API Keys**: Ensure your `.env` file is in the same directory with `GROQ_API_KEY` and `GOOGLE_API_KEY`
- ⚠️ **First Run**: May take a moment to load the model for the first time
- 💾 **Caching**: Models are cached using `@st.cache_resource` for faster subsequent runs

## Customization

You can customize the app by modifying:
- Colors in the CSS section at the top
- Model selection in `load_model()`
- Prompts in each node function
- UI layout and components

## Troubleshooting

**Issue**: "ModuleNotFoundError"
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: "API Key not found"
- **Solution**: Check that `.env` file exists in the same directory with valid API keys

**Issue**: App runs slowly
- **Solution**: This is normal for the first run. Subsequent runs are cached and much faster

**Issue**: "Port 8501 is already in use"
- **Solution**: Use `streamlit run app.py --logger.level=debug --server.port 8502`

## Stopping the App

Press `Ctrl+C` in the terminal where the app is running.

---

Enjoy your new UI! 🚀
