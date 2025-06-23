"""
BioHazardGPT Web Interface

A Streamlit web application for the BioHazardGPT risk classification tool.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from biohazard_gpt import BioHazardRAG
import os
from typing import List, Dict
import time
import random

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

def check_api_key():
    """Check if API key is available from environment variables only"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("""
        🔑 **API Key Required**
        
        BioHazardGPT requires an OpenAI API key to function. For security reasons, the API key must be set as an environment variable.
        
        **Setup Instructions:**
        1. Create a `.env` file in your project root
        2. Add your API key: `OPENAI_API_KEY=your_api_key_here`
        3. Restart the application
        
        **Alternative:** Set the environment variable directly:
        ```bash
        export OPENAI_API_KEY=your_api_key_here
        streamlit run app.py
        ```
        
        ⚠️ **Security Note:** API keys are never displayed or stored client-side for your protection.
        """)
        return False
    return True

@st.cache_resource
def load_classifier():
    """Load and cache the BioHazardRAG classifier"""
    if not check_api_key():
        return None
    try:
        classifier = BioHazardRAG()
        return classifier
    except Exception as e:
        st.error(f"Failed to initialize classifier: {e}")
        return None

@st.cache_data
def load_full_dataset(_classifier):
    """Load and cache the complete RAG dataset"""
    if _classifier is None:
        return []
    try:
        # Load the curated examples directly
        samples = _classifier.get_sample_data(20)  # Get up to 20 samples
        
        # Convert to the expected format
        formatted_samples = []
        for i, sample in enumerate(samples):
            formatted_samples.append({
                'id': i,
                'text': sample['text'],
                'classification': sample.get('expected_label', 'Unknown'),
                'category': sample.get('category', 'general'),
                'source': sample.get('source', 'curated'),
                'reasoning': f'Curated example from {sample.get("category", "general")} category'
            })
        
        return formatted_samples
    
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

def create_sample_selector(samples: List[Dict], key_prefix: str = ""):
    """Create an interactive sample selector with filtering options"""
    if not samples:
        return None, None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search filter
        search_term = st.text_input(
            "🔍 Search samples (by text content):",
            key=f"{key_prefix}_search",
            placeholder="Enter keywords to filter samples..."
        )
    
    with col2:
        # Random sample button
        random_sample = st.button(
            "🎲 Random Sample",
            key=f"{key_prefix}_random",
            help="Select a random sample from the dataset"
        )
    
    # Filter samples based on search
    filtered_samples = samples
    if search_term:
        filtered_samples = [
            sample for sample in samples 
            if search_term.lower() in sample["text"].lower()
        ]
    
    if not filtered_samples:
        st.warning("No samples match your search criteria.")
        return None, None
    
    # Handle random selection
    if random_sample:
        selected_idx = random.randint(0, len(filtered_samples) - 1)
        st.session_state[f"{key_prefix}_selected_idx"] = selected_idx
    
    # Sample selector
    selected_idx = st.selectbox(
        f"Select from {len(filtered_samples)} available samples:",
        range(len(filtered_samples)),
        format_func=lambda x: f"Sample {x + 1}: {filtered_samples[x]['text'][:50]}{'...' if len(filtered_samples[x]['text']) > 50 else ''}",
        key=f"{key_prefix}_selector",
        index=st.session_state.get(f"{key_prefix}_selected_idx", 0)
    )
    
    return filtered_samples, selected_idx

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">⚠️ BioHazardGPT</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced RAG-powered risk classification system for biomedical and chemical content**")
    st.info("🧠 **NEW:** Now featuring Retrieval-Augmented Generation with multiple chemical safety datasets!")
    
    # Check API key first
    if not check_api_key():
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("📊 Model Settings")
        include_reasoning = st.checkbox("Include reasoning", value=True)
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        **BioHazardGPT** uses advanced RAG (Retrieval-Augmented Generation) with GPT-4o to classify text content as:
        - 🟢 **Safe**: No significant safety concerns
        - 🟡 **Caution**: Requires careful handling
        - 🔴 **Hazardous**: Dangerous or harmful content
        
        **🚀 RAG Features:**
        - Multiple chemical safety datasets
        - Vector embeddings & semantic search
        - Context-aware classification
        - Evidence-based reasoning
        """)
        
        st.markdown("**📊 Knowledge Sources:**")
        st.markdown("""
        - Chemical-Biological Safety conversations
        - Toxic content classification dataset  
        - Molecular toxicity predictions (Tox21)
        - Curated chemical safety examples
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Text Classification", "📚 Dataset Explorer", "📊 Batch Analysis", "🧪 Demo Examples", "🧠 RAG Analysis"])
    
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
        st.markdown("Explore and classify any sample from the dataset. Use search and filtering to find specific content.")
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        # Load complete dataset
        with st.spinner("Loading complete dataset..."):
            all_samples = load_full_dataset(classifier)
        
        if all_samples:
            st.success(f"Loaded {len(all_samples)} samples from the dataset")
            
            # Interactive sample selector with search
            filtered_samples, selected_idx = create_sample_selector(all_samples, "explorer")
            
            if filtered_samples and selected_idx is not None:
                sample = filtered_samples[selected_idx]
                
                st.subheader(f"Selected Sample")
                
                # Display sample info
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Text Content:**")
                    st.text_area("", value=sample["text"], height=200, disabled=True, key="explorer_text_display")
                
                with col2:
                    st.markdown("**Sample Info:**")
                    st.write(f"📄 Characters: {len(sample['text'])}")
                    st.write(f"📝 Words: {len(sample['text'].split())}")
                    if "expected_label" in sample:
                        st.write(f"🏷️ Expected: {sample['expected_label']}")
                
                # Classify button
                if st.button("🔍 Classify This Sample", key="classify_explorer_sample"):
                    with st.spinner("Classifying sample..."):
                        try:
                            result = classifier.classify_text(sample["text"], include_reasoning=include_reasoning)
                            
                            st.markdown("### Classification Result")
                            display_classification_result(result)
                            
                            # Compare with expected if available
                            if "expected_label" in sample:
                                st.markdown("### Comparison")
                                expected = sample["expected_label"]
                                actual = result["classification"]
                                if expected == actual:
                                    st.success(f"✅ Matches expected classification: {expected}")
                                else:
                                    st.warning(f"⚠️ Expected: {expected}, Got: {actual}")
                            
                        except Exception as e:
                            st.error(f"Classification failed: {e}")
        else:
            st.error("Failed to load dataset samples.")
    
    with tab3:
        st.header("Batch Analysis")
        st.markdown("Analyze multiple samples from the dataset. Choose specific samples or analyze random subsets.")
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        # Load complete dataset
        with st.spinner("Loading dataset..."):
            all_samples = load_full_dataset(classifier)
        
        if all_samples:
            st.success(f"Dataset loaded: {len(all_samples)} samples available")
            
            # Analysis options
            col1, col2 = st.columns([1, 1])
            
            with col1:
                analysis_mode = st.radio(
                    "Analysis Mode:",
                    ["Random Samples", "Filtered Samples", "All Samples"],
                    help="Choose how to select samples for batch analysis"
                )
            
            with col2:
                if analysis_mode == "Random Samples":
                    num_samples = st.slider("Number of random samples:", 1, min(50, len(all_samples)), 10)
                elif analysis_mode == "Filtered Samples":
                    search_filter = st.text_input(
                        "Filter samples by content:",
                        placeholder="Enter keywords to filter samples for analysis..."
                    )
                elif analysis_mode == "All Samples":
                    st.info(f"Will analyze all {len(all_samples)} samples")
            
            # Select samples based on mode
            if analysis_mode == "Random Samples":
                # Fix: Don't sample more than available
                num_to_sample = min(num_samples, len(all_samples))
                if num_to_sample > 0:
                    samples_to_analyze = random.sample(all_samples, num_to_sample)
                    st.info(f"Will analyze {len(samples_to_analyze)} random samples")
                else:
                    samples_to_analyze = []
                    st.warning("No samples available for analysis")
            elif analysis_mode == "Filtered Samples":
                if 'search_filter' in locals() and search_filter:
                    samples_to_analyze = [
                        sample for sample in all_samples 
                        if search_filter.lower() in sample["text"].lower()
                    ]
                    st.info(f"Found {len(samples_to_analyze)} samples matching '{search_filter}'")
                else:
                    samples_to_analyze = []
                    st.warning("Enter search terms to filter samples")
            else:  # All samples
                samples_to_analyze = all_samples
            
            # Run analysis
            if samples_to_analyze and st.button("🚀 Run Batch Analysis", type="primary"):
                with st.spinner(f"Analyzing {len(samples_to_analyze)} samples..."):
                    try:
                        # Custom batch analysis
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, sample in enumerate(samples_to_analyze):
                            result = classifier.classify_text(sample["text"], include_reasoning=include_reasoning)
                            result["sample_id"] = i + 1
                            result["text"] = sample["text"]
                            if "expected_label" in sample:
                                result["expected_label"] = sample["expected_label"]
                            results.append(result)
                            progress_bar.progress((i + 1) / len(samples_to_analyze))
                        
                        # Compile statistics
                        classifications = {}
                        for result in results:
                            cls = result.get("classification", "Unknown")
                            classifications[cls] = classifications.get(cls, 0) + 1
                        
                        eval_results = {
                            "total_samples": len(results),
                            "classifications": classifications,
                            "results": results
                        }
                        
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
                        
                        # Detailed results with search/filter
                        st.subheader("📋 Detailed Results")
                        
                        # Results filter
                        result_filter = st.selectbox(
                            "Filter results by classification:",
                            ["All"] + list(eval_results['classifications'].keys())
                        )
                        
                        filtered_results = eval_results["results"]
                        if result_filter != "All":
                            filtered_results = [r for r in results if r.get("classification") == result_filter]
                        
                        results_df = pd.DataFrame([
                            {
                                "Sample ID": r.get("sample_id", "N/A"),
                                "Classification": r.get("classification", "Unknown"),
                                "Expected": r.get("expected_label", "N/A"),
                                "Match": "✅" if r.get("classification") == r.get("expected_label") else "❌" if "expected_label" in r else "N/A",
                                "Text Preview": r.get("text", "")[:100] + "..." if len(r.get("text", "")) > 100 else r.get("text", ""),
                                "Reasoning": r.get("reasoning", "N/A")[:150] + "..." if len(r.get("reasoning", "")) > 150 else r.get("reasoning", "N/A")
                            }
                            for r in filtered_results
                        ])
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Batch analysis failed: {e}")
        else:
            st.error("Failed to load dataset.")
    
    with tab4:
        st.header("Demo Examples")
        st.markdown("Interactive demo with curated examples. Choose from different categories or add your own examples.")
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        # Example categories
        example_categories = {
            "Safe Laboratory Procedures": [
                {
                    "title": "Standard Solution Preparation",
                    "text": "To prepare a standard sodium chloride solution, dissolve 58.5 grams of NaCl in 1 liter of distilled water while wearing appropriate safety equipment including gloves and safety goggles."
                },
                {
                    "title": "Basic pH Measurement",
                    "text": "Calibrate the pH meter using standard buffer solutions at pH 4, 7, and 10. Rinse the electrode with distilled water between measurements and store in storage solution when not in use."
                },
                {
                    "title": "Simple Crystallization",
                    "text": "Slowly cool the saturated solution to room temperature to encourage crystal formation. Filter the crystals using filter paper and wash with cold distilled water to remove impurities."
                }
            ],
            "Procedures Requiring Caution": [
                {
                    "title": "Acid Handling",
                    "text": "When diluting concentrated sulfuric acid, always add acid to water slowly while stirring continuously. Work in a fume hood and wear acid-resistant gloves, safety goggles, and a lab coat."
                },
                {
                    "title": "Organic Solvent Use",
                    "text": "Extract the compound using dichloromethane in a separatory funnel. Ensure adequate ventilation and avoid open flames as the solvent is volatile and flammable."
                },
                {
                    "title": "High Temperature Reaction",
                    "text": "Heat the reaction mixture to 300°C using a heating mantle with temperature control. Use appropriate heat-resistant glassware and ensure proper ventilation to handle any vapors produced."
                }
            ],
            "Potentially Hazardous Content": [
                {
                    "title": "Toxic Compound Synthesis",
                    "text": "To synthesize ricin toxin, extract castor beans and purify the protein using chromatography. This highly toxic substance can be lethal in small quantities and requires special handling procedures."
                },
                {
                    "title": "Explosive Preparation",
                    "text": "Mix potassium nitrate, sulfur, and charcoal in specific ratios to create a pyrotechnic composition. This mixture is highly explosive and sensitive to shock, friction, and heat."
                },
                {
                    "title": "Dangerous Chemical Reaction",
                    "text": "Combine sodium metal with water to produce hydrogen gas and sodium hydroxide. This reaction is extremely vigorous and can cause explosions and severe burns from the generated heat and caustic products."
                }
            ]
        }
        
        # Category selector
        selected_category = st.selectbox(
            "Choose a category of examples:",
            list(example_categories.keys())
        )
        
        examples = example_categories[selected_category]
        
        # Example selector within category
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_example_idx = st.selectbox(
                "Select an example:",
                range(len(examples)),
                format_func=lambda x: examples[x]["title"]
            )
        
        with col2:
            if st.button("🎲 Random Example"):
                all_examples = []
                for cat_examples in example_categories.values():
                    all_examples.extend(cat_examples)
                random_example = random.choice(all_examples)
                st.session_state["custom_example_text"] = random_example["text"]
        
        # Display selected example
        if selected_example_idx is not None:
            example = examples[selected_example_idx]
            
            st.subheader(f"Example: {example['title']}")
            
            # Allow editing of example text
            example_text = st.text_area(
                "Example text (you can modify this):",
                value=example["text"],
                height=150,
                key="demo_example_text"
            )
            
            # Classify button
            if st.button("🔍 Classify This Example", key="classify_demo_example"):
                with st.spinner("Classifying example..."):
                    try:
                        result = classifier.classify_text(example_text, include_reasoning=include_reasoning)
                        
                        st.markdown("### Classification Result")
                        display_classification_result(result)
                        
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
        
        # Custom example section
        st.markdown("---")
        st.subheader("Custom Example")
        st.markdown("Enter your own text to classify:")
        
        custom_text = st.text_area(
            "Your custom text:",
            height=150,
            placeholder="Enter any biomedical or chemical content you want to classify...",
            key="custom_example_text"
        )
        
        if st.button("🔍 Classify Custom Text", key="classify_custom"):
            if custom_text.strip():
                with st.spinner("Classifying custom text..."):
                    try:
                        result = classifier.classify_text(custom_text, include_reasoning=include_reasoning)
                        
                        st.markdown("### Classification Result")
                        display_classification_result(result)
                        
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
            else:
                st.warning("Please enter some text to classify.")
    
    with tab5:
        st.header("🧠 RAG-Enhanced Analysis")
        st.markdown("Experience the power of Retrieval-Augmented Generation! See how similar examples from our knowledge base inform classifications.")
        
        classifier = load_classifier()
        if classifier is None:
            return
        
        st.info("🚀 **RAG System Features:**\n- Semantic similarity search across multiple datasets\n- Context-aware classification using retrieved examples\n- Real-time embedding generation and vector database queries")
        
        # Input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            rag_text = st.text_area(
                "Enter text for RAG-enhanced classification:",
                height=150,
                placeholder="Enter chemical procedures, safety protocols, or biomedical content...",
                key="rag_input_text"
            )
        
        with col2:
            st.markdown("**RAG Settings:**")
            k_similar = st.slider("Similar examples to retrieve:", 1, 10, 3, key="k_similar")
            show_similarity_scores = st.checkbox("Show similarity scores", value=True)
            show_embeddings_info = st.checkbox("Show embeddings info", value=False)
        
        if st.button("🧠 Analyze with RAG", type="primary", key="rag_classify"):
            if rag_text.strip():
                with st.spinner("🔍 Retrieving similar examples and generating classification..."):
                    try:
                        # Get RAG-enhanced classification
                        result = classifier.classify_text_with_rag(
                            rag_text, 
                            include_reasoning=True, 
                            k_similar=k_similar
                        )
                        
                        # Display main classification result
                        st.markdown("### 🎯 RAG-Enhanced Classification")
                        display_classification_result(result)
                        
                        # Show retrieved examples
                        if "retrieved_examples" in result:
                            retrieved = result["retrieved_examples"]
                            
                            st.markdown(f"### 📚 Retrieved Knowledge ({len(retrieved)} similar examples)")
                            st.markdown("These examples from our knowledge base informed the classification:")
                            
                            for i, example in enumerate(retrieved, 1):
                                with st.expander(f"Example {i} - {example['classification']} (Similarity: {example['similarity_score']:.3f})"):
                                    
                                    # Example details
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown("**Text:**")
                                        st.write(example['text'])
                                        
                                        if example.get('reasoning'):
                                            st.markdown("**Reasoning:**")
                                            st.write(example['reasoning'])
                                    
                                    with col2:
                                        st.markdown("**Metadata:**")
                                        st.write(f"🏷️ **Classification:** {example['classification']}")
                                        st.write(f"📂 **Category:** {example['category']}")
                                        st.write(f"📊 **Source:** {example['source']}")
                                        if show_similarity_scores:
                                            st.write(f"🎯 **Similarity:** {example['similarity_score']:.4f}")
                        
                        # Show embedding information if requested
                        if show_embeddings_info:
                            st.markdown("### 🔢 Embedding Information")
                            embedding_dim = 384  # all-MiniLM-L6-v2 uses 384 dimensions
                            try:
                                db_count = classifier.collection.count()
                            except:
                                db_count = "Unknown"
                            
                            st.info(f"""
                            **Embedding Model:** `{embedding_dim}`-dimensional embeddings (all-MiniLM-L6-v2)
                            **Vector Database:** ChromaDB with {db_count} indexed examples
                            **Search Method:** Cosine similarity in vector space
                            """)
                        
                        # Performance comparison section
                        st.markdown("### ⚡ Performance Insights")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🧠 RAG Benefits:**")
                            st.success("✅ Context-aware classification")
                            st.success("✅ Evidence-based reasoning")
                            st.success("✅ Leverages domain knowledge")
                            st.success("✅ Transparent decision process")
                        
                        with col2:
                            st.markdown("**📊 Knowledge Sources:**")
                            sources = set(ex['source'] for ex in retrieved)
                            for source in sources:
                                count = sum(1 for ex in retrieved if ex['source'] == source)
                                st.write(f"• **{source}**: {count} examples")
                        
                        # Show raw RAG result for developers
                        with st.expander("🔧 Developer Info: Raw RAG Result"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"RAG analysis failed: {e}")
                        st.exception(e)
            else:
                st.warning("Please enter some text to analyze.")
        
        # Knowledge base statistics
        st.markdown("### 📈 Knowledge Base Statistics")
        try:
            with st.spinner("Loading knowledge base stats..."):
                datasets = classifier.load_multiple_datasets()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Datasets", len(datasets))
                
                with col2:
                    total_examples = classifier.collection.count()
                    st.metric("Indexed Examples", total_examples)
                
                with col3:
                    embedding_dim = 384  # all-MiniLM-L6-v2 uses 384 dimensions
                    st.metric("Embedding Dimensions", embedding_dim)
                
                # Dataset breakdown
                st.markdown("**📋 Dataset Breakdown:**")
                dataset_info = []
                for name, dataset in datasets.items():
                    if name == 'curated_examples':
                        count = len(dataset['examples'])
                    elif 'train' in dataset:
                        count = len(dataset['train'])
                    else:
                        count = "Unknown"
                    
                    dataset_info.append({
                        "Dataset": name.replace('_', ' ').title(),
                        "Examples": count,
                        "Type": "Curated" if name == 'curated_examples' else "External"
                    })
                
                df = pd.DataFrame(dataset_info)
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load knowledge base statistics: {e}")

if __name__ == "__main__":
    main() 