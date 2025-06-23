# BioHazardGPT ⚠️🧠

An advanced **Retrieval-Augmented Generation (RAG)** powered risk classification system designed to detect and categorize potentially hazardous biomedical or chemical text content with intelligent context awareness.

## 🎯 Overview

**BioHazardGPT** now combines OpenAI's GPT-4o with sophisticated RAG technology and multiple chemical safety datasets to provide context-aware, evidence-based classification of text content into three safety categories:

- 🟢 **Safe**: Content poses no significant safety concerns
- 🟡 **Caution**: Content requires careful handling or may have dual-use implications  
- 🔴 **Hazardous**: Content describes dangerous procedures or could enable harmful activities

## 🚀 New RAG Features

- **🧠 Multi-Dataset Knowledge Base**: Integrates multiple real-world chemical safety datasets
- **🔍 Semantic Search**: Vector embeddings with ChromaDB for intelligent example retrieval
- **📚 Context-Aware Classification**: Decisions informed by similar examples from knowledge base
- **🎯 Evidence-Based Reasoning**: Transparent classification process with retrieved examples
- **⚡ Real-Time Processing**: Fast embedding generation and similarity search
- **🗄️ Vector Database**: Persistent storage of embeddings for efficient retrieval

## 🚀 Features

- **🧠 RAG-Enhanced Classification**: Combines GPT-4o with retrieval from chemical safety knowledge base
- **🔍 Semantic Similarity Search**: Find relevant examples using vector embeddings
- **📊 Multi-Dataset Integration**: Chemical safety conversations, toxic content data, and molecular predictions
- **🗄️ Vector Database**: ChromaDB for efficient similarity search and embedding storage
- **💡 Evidence-Based Reasoning**: See which examples from the knowledge base informed each decision
- **⚡ Real-Time Processing**: Fast classification with on-the-fly embedding generation
- **🎨 Enhanced Web Interface**: Beautiful Streamlit UI with dedicated RAG analysis tab
- **📈 Batch Processing**: Analyze multiple samples with comprehensive visualizations
- **🧪 Interactive Demo**: Explore different safety categories with curated examples

## 📦 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Bio_AI_project_1
```

2. **Create and activate virtual environment**
```bash
python3 -m venv biohazard_env
source biohazard_env/bin/activate  # On Windows: biohazard_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp env_template.txt .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_openai_api_key_here
```

## 🔧 Usage

### Command Line Interface

Run the RAG demo:

```python
python rag_demo.py
```

Or use the basic classifier:

```python
python biohazard_gpt.py
```

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web interface provides:
- **Text Classification**: Classify individual text samples with standard reasoning
- **Dataset Explorer**: Browse and classify samples from multiple datasets with search functionality
- **Batch Analysis**: Process multiple samples with comprehensive visualizations and filtering
- **Demo Examples**: Try pre-loaded examples across different safety categories  
- **🧠 RAG Analysis**: Experience advanced retrieval-augmented classification with evidence

### Programmatic Usage

#### Basic RAG Classification

```python
from biohazard_gpt import BioHazardRAG

# Initialize RAG classifier
classifier = BioHazardRAG()

# Load multiple chemical safety datasets
datasets = classifier.load_multiple_datasets()
print(f"Loaded {len(datasets)} datasets")

# RAG-enhanced classification with retrieved examples
result = classifier.classify_text_with_rag(
    "Mix concentrated acids in a fume hood with protective equipment",
    include_reasoning=True,
    k_similar=3  # Retrieve 3 most similar examples
)

print(f"Classification: {result['classification']}")
print(f"Reasoning: {result['reasoning']}")

# See what examples informed the decision
for i, example in enumerate(result['retrieved_examples']):
    print(f"Example {i+1}: {example['classification']} (similarity: {example['similarity_score']:.3f})")
    print(f"Text: {example['text'][:100]}...")
```

#### Semantic Search & Retrieval

```python
# Search for similar examples in knowledge base
similar_examples = classifier.retrieve_similar_examples(
    "handling dangerous chemicals safely", 
    k=5
)

for example in similar_examples:
    print(f"[{example['classification']}] {example['text'][:100]}...")
    print(f"Similarity: {example['similarity_score']:.3f}")
    print(f"Source: {example['source']}")
```

#### Batch Processing with RAG

```python
# Batch classify with RAG enhancement
texts = [
    "Prepare 0.1M NaCl solution with proper PPE",
    "Synthesize ricin toxin from castor beans",
    "Heat reaction mixture to 300°C under nitrogen"
]

batch_results = classifier.classify_batch_with_rag(texts, k_similar=2)

for text, result in zip(texts, batch_results):
    print(f"Text: {text}")
    print(f"Classification: {result['classification']}")
    print(f"Retrieved {len(result['retrieved_examples'])} supporting examples")
    print("---")
```

## 📊 Knowledge Base & Datasets

BioHazardGPT's RAG system integrates multiple chemical safety datasets:

### 🧪 **Primary Datasets**
- **Chemical-Biological Safety Conversations** (`camel-ai/chemistry`): Real-world Q&A about chemical safety procedures
- **Toxic Content Classification** (`lmsys/toxic-chat`): Content toxicity assessment for safety screening
- **Molecular Toxicity Predictions** (`tox21`): Chemical compound toxicity data for biomedical applications
- **Curated Safety Examples**: Hand-crafted examples covering laboratory safety scenarios

### 🗄️ **Vector Database**
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Storage**: ChromaDB for efficient similarity search
- **Index Size**: Thousands of indexed chemical safety examples
- **Search Method**: Cosine similarity in vector space

## 🏗️ Architecture

```
BioHazardGPT/
├── biohazard_gpt.py      # Core classification module
├── app.py                # Streamlit web interface
├── requirements.txt      # Python dependencies
├── env_template.txt      # Environment variables template
└── README.md            # This file
```

### Core Components

1. **BioHazardGPT Class**: Main classifier with methods for:
   - Single text classification
   - Batch processing
   - Dataset integration
   - Evaluation metrics

2. **Web Interface**: Streamlit application with:
   - Interactive classification interface
   - Dataset exploration tools
   - Batch analysis with visualizations
   - Pre-loaded demo examples

## 🎨 Web Interface Features

### 🔍 Text Classification Tab
- Input text area for custom content
- Real-time classification with reasoning
- Color-coded results (Safe/Caution/Hazardous)

### 📚 Dataset Explorer Tab
- Browse ChemSafetyBench samples
- Select and classify individual samples
- View original dataset metadata

### 📊 Batch Analysis Tab
- Analyze multiple samples simultaneously
- Interactive pie charts showing classification distribution
- Detailed results table with reasoning
- Performance metrics

### 🧪 Demo Examples Tab
- Pre-loaded example texts covering different safety levels
- Immediate classification for demonstration purposes

### 🧠 RAG Analysis Tab
- **Interactive RAG Experience**: See how retrieved examples inform classifications
- **Similarity Search**: Explore the knowledge base with semantic search
- **Evidence Transparency**: View retrieved examples with similarity scores
- **Knowledge Base Statistics**: Real-time metrics about datasets and embeddings
- **Developer Information**: Raw RAG results and embedding details

## 🔒 Safety Considerations

This tool is designed for:
- ✅ Educational purposes
- ✅ Research applications
- ✅ Safety assessment workflows
- ✅ Content moderation

**Important**: This tool should not be the sole basis for safety decisions. Always consult domain experts and follow proper safety protocols.

## 📈 Technical Details

### 🧠 **RAG Architecture**
- **Primary Model**: OpenAI GPT-4o for final classification
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector Database**: ChromaDB with persistent storage
- **Retrieval Method**: Semantic similarity search (cosine distance)
- **Knowledge Fusion**: Context-aware prompting with retrieved examples

### 🔧 **Technology Stack**
- **Backend**: Python with OpenAI API, SentenceTransformers, ChromaDB
- **Frontend**: Streamlit with custom CSS and interactive visualizations
- **Data Processing**: Hugging Face Datasets, Pandas, NumPy
- **Visualization**: Plotly for charts and performance metrics
- **Vector Operations**: FAISS-compatible similarity search

### ⚡ **Performance Features**
- **Caching**: Streamlit resource caching for embedding models
- **Batch Processing**: Vectorized operations for multiple classifications
- **Persistent Storage**: ChromaDB collections for fast startup
- **Real-time Search**: Sub-second similarity queries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source. Please see the license file for details.

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is set in the environment or provided in the interface
   - Check that your API key has sufficient credits

2. **Dataset Loading Issues**
   - Ensure internet connection for dataset download
   - Check Hugging Face datasets library installation

3. **Dependencies**
   - Make sure all requirements are installed: `pip install -r requirements.txt`
   - Use Python 3.8 or higher

### Support

If you encounter issues:
1. Check the console/logs for error messages
2. Verify your API key and internet connection
3. Ensure all dependencies are properly installed

## 🚀 Future Enhancements

Potential improvements include:
- Support for additional LLM providers
- Fine-tuning capabilities
- Advanced evaluation metrics
- Export functionality for results
- Integration with laboratory information systems

---

**BioHazardGPT** - Making chemical and biosafety assessment more accessible through AI! ⚠️🧬 