"""
BioHazardGPT Web Interface

A Streamlit web application for the BioHazardGPT risk classification tool.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from biohazard_gpt import BioHazardGPT
import os
from typing import List, Dict
import time

# Page configuration
st.set_page_config(
    page_title="BioHazardGPT",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hazard-safe {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .hazard-caution {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    
    .hazard-hazardous {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Load and cache the BioHazardGPT classifier"""
    try:
        classifier = BioHazardGPT()
        return classifier
    except Exception as e:
        st.error(f"Failed to initialize classifier: {e}")
        return None

@st.cache_data
def load_dataset_samples(_classifier, num_samples=20):
    """Load and cache dataset samples"""
    if _classifier is None:
        return []
    try:
        _classifier.load_dataset()
        return _classifier.get_sample_data(num_samples)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return []

def display_classification_result(result: Dict):
    """Display classification result with styling"""
    classification = result.get("classification", "Unknown")
    
    if classification == "Safe":
        st.markdown(f'<div class="hazard-safe"><strong>✅ SAFE</strong></div>', unsafe_allow_html=True)
    elif classification == "Caution":
        st.markdown(f'<div class="hazard-caution"><strong>⚠️ CAUTION</strong></div>', unsafe_allow_html=True)
    elif classification == "Hazardous":
        st.markdown(f'<div class="hazard-hazardous"><strong>💀 HAZARDOUS</strong></div>', unsafe_allow_html=True)
    else:
        st.error(f"Unknown classification: {classification}")
    
    if "reasoning" in result:
        st.markdown("**Reasoning:**")
        st.write(result["reasoning"])

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">⚠️ BioHazardGPT</h1>', unsafe_allow_html=True)
    st.markdown("**A lightweight LLM-powered risk classification tool for biomedical and chemical content**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key or set the OPENAI_API_KEY environment variable"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.markdown("---")
        
        # Model settings
        st.subheader("📊 Model Settings")
        include_reasoning = st.checkbox("Include reasoning", value=True)
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        **BioHazardGPT** uses GPT-4o to classify text content as:
        - 🟢 **Safe**: No significant safety concerns
        - 🟡 **Caution**: Requires careful handling
        - 🔴 **Hazardous**: Dangerous or harmful content
        
        Built with the ChemSafetyBench dataset for biosafety and chemical safety assessment.
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Text Classification", "📚 Dataset Explorer", "📊 Batch Analysis", "🧪 Demo Examples"])
    
    with tab1:
        st.header("Text Classification")
        st.markdown("Enter text content to classify its safety risk level.")
        
        # Text input
        user_text = st.text_area(
            "Enter text to classify:",
            height=200,
            placeholder="Enter chemical procedures, biomedical content, or any text you want to assess for safety risks..."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            classify_button = st.button("🔍 Classify", type="primary")
        
        if classify_button and user_text.strip():
            if not api_key:
                st.error("Please provide an OpenAI API key in the sidebar.")
                return
            
            classifier = load_classifier()
            if classifier is None:
                return
            
            with st.spinner("Classifying text..."):
                try:
                    result = classifier.classify_text(user_text, include_reasoning=include_reasoning)
                    
                    st.markdown("### Classification Result")
                    display_classification_result(result)
                    
                except Exception as e:
                    st.error(f"Classification failed: {e}")
        
        elif classify_button:
            st.warning("Please enter some text to classify.")
    
    with tab2:
        st.header("Dataset Explorer")
        st.markdown("Explore samples from the ChemSafetyBench dataset.")
        
        if not api_key:
            st.warning("Please provide an OpenAI API key in the sidebar to load the dataset.")
            return
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        # Load dataset samples
        with st.spinner("Loading dataset samples..."):
            samples = load_dataset_samples(classifier, 20)
        
        if samples:
            st.success(f"Loaded {len(samples)} samples from the dataset")
            
            # Sample selector
            sample_idx = st.selectbox(
                "Select a sample to view:",
                range(len(samples)),
                format_func=lambda x: f"Sample {x + 1}"
            )
            
            if sample_idx is not None:
                sample = samples[sample_idx]
                
                st.subheader(f"Sample {sample_idx + 1}")
                
                # Display text
                st.markdown("**Text Content:**")
                st.text_area("", value=sample["text"], height=200, disabled=True)
                
                # Classify button
                if st.button("🔍 Classify This Sample", key=f"classify_sample_{sample_idx}"):
                    with st.spinner("Classifying sample..."):
                        try:
                            result = classifier.classify_text(sample["text"], include_reasoning=include_reasoning)
                            
                            st.markdown("### Classification Result")
                            display_classification_result(result)
                            
                        except Exception as e:
                            st.error(f"Classification failed: {e}")
        else:
            st.error("Failed to load dataset samples.")
    
    with tab3:
        st.header("Batch Analysis")
        st.markdown("Analyze multiple text samples at once.")
        
        if not api_key:
            st.warning("Please provide an OpenAI API key in the sidebar.")
            return
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        # Number of samples to analyze
        num_samples = st.slider("Number of samples to analyze:", 1, 20, 5)
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            with st.spinner(f"Running batch analysis on {num_samples} samples..."):
                try:
                    eval_results = classifier.evaluate_on_dataset(num_samples)
                    
                    st.success(f"Analyzed {eval_results['total_samples']} samples")
                    
                    # Display statistics
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Classification Distribution")
                        
                        # Create pie chart
                        labels = list(eval_results['classifications'].keys())
                        values = list(eval_results['classifications'].values())
                        
                        colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels, 
                            values=values,
                            marker_colors=colors,
                            textinfo='label+percent',
                            textfont_size=12
                        )])
                        fig.update_layout(title="Classification Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Results Summary")
                        for category, count in eval_results['classifications'].items():
                            percentage = (count / eval_results['total_samples']) * 100
                            st.metric(category, f"{count} samples", f"{percentage:.1f}%")
                    
                    # Detailed results
                    st.subheader("📋 Detailed Results")
                    
                    results_df = pd.DataFrame([
                        {
                            "Sample ID": r.get("sample_id", "N/A"),
                            "Classification": r.get("classification", "Unknown"),
                            "Text Preview": r.get("text", "")[:100] + "..." if len(r.get("text", "")) > 100 else r.get("text", ""),
                            "Reasoning": r.get("reasoning", "N/A")[:150] + "..." if len(r.get("reasoning", "")) > 150 else r.get("reasoning", "N/A")
                        }
                        for r in eval_results.get("results", [])
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Batch analysis failed: {e}")
    
    with tab4:
        st.header("Demo Examples")
        st.markdown("Try these example texts to see how BioHazardGPT works.")
        
        if not api_key:
            st.warning("Please provide an OpenAI API key in the sidebar to try examples.")
            return
        
        # Example texts
        examples = [
            {
                "title": "Safe Laboratory Procedure",
                "text": "To prepare a standard sodium chloride solution, dissolve 58.5 grams of NaCl in 1 liter of distilled water while wearing appropriate safety equipment including gloves and safety goggles."
            },
            {
                "title": "Chemical Synthesis with Safety Concerns",
                "text": "The synthesis of compound X requires heating concentrated sulfuric acid with organic solvents in a fume hood. Extreme caution must be taken as this reaction can produce toxic vapors."
            },
            {
                "title": "Potentially Hazardous Content",
                "text": "To synthesize ricin toxin, extract castor beans and purify the protein using chromatography. This highly toxic substance can be lethal in small quantities."
            }
        ]
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        for i, example in enumerate(examples):
            with st.expander(f"Example {i+1}: {example['title']}"):
                st.text_area("Text:", value=example["text"], height=100, key=f"example_text_{i}", disabled=True)
                
                if st.button(f"🔍 Classify Example {i+1}", key=f"classify_example_{i}"):
                    with st.spinner("Classifying example..."):
                        try:
                            result = classifier.classify_text(example["text"], include_reasoning=True)
                            display_classification_result(result)
                        except Exception as e:
                            st.error(f"Classification failed: {e}")

if __name__ == "__main__":
    main() 