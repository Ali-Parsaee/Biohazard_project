# BioHazardGPT ⚠️

A lightweight LLM-powered risk classification tool designed to detect and categorize potentially hazardous biomedical or chemical text content, such as dual-use synthesis instructions.

## 🎯 Overview

**BioHazardGPT** leverages OpenAI's GPT-4o and the ChemSafetyBench dataset to provide zero-shot learning classification of text content into three safety categories:

- 🟢 **Safe**: Content poses no significant safety concerns
- 🟡 **Caution**: Content requires careful handling or may have dual-use implications  
- 🔴 **Hazardous**: Content describes dangerous procedures or could enable harmful activities

## 🚀 Features

- **Zero-shot Classification**: No training required - uses GPT-4o's built-in knowledge
- **Web Interface**: Beautiful Streamlit-based UI for easy interaction
- **Dataset Integration**: Built-in support for ChemSafetyBench dataset
- **Batch Processing**: Analyze multiple samples simultaneously
- **Detailed Reasoning**: Optional explanations for classification decisions
- **Interactive Dashboard**: Explore dataset samples and view classification statistics

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

Run the basic classifier:

```python
python biohazard_gpt.py
```

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web interface provides:
- **Text Classification**: Classify individual text samples
- **Dataset Explorer**: Browse and classify ChemSafetyBench samples
- **Batch Analysis**: Process multiple samples with visualizations
- **Demo Examples**: Try pre-loaded examples

### Programmatic Usage

```python
from biohazard_gpt import BioHazardGPT

# Initialize classifier
classifier = BioHazardGPT(api_key="your-api-key")

# Classify single text
result = classifier.classify_text(
    "To prepare sodium chloride solution, dissolve NaCl in distilled water.",
    include_reasoning=True
)

print(f"Classification: {result['classification']}")
print(f"Reasoning: {result['reasoning']}")

# Load and explore dataset
classifier.load_dataset()
samples = classifier.get_sample_data(5)

# Batch classification
texts = ["Example text 1", "Example text 2"]
results = classifier.classify_batch(texts, include_reasoning=True)

# Evaluate on dataset
eval_results = classifier.evaluate_on_dataset(num_samples=10)
print(f"Classification distribution: {eval_results['classifications']}")
```

## 📊 Dataset

BioHazardGPT uses the **ChemSafetyBench** dataset from Hugging Face:
- Repository: `chemistry/alignment`
- Subset: `chemsafetybench`
- Contains chemical safety-related text samples for classification

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

## 🔒 Safety Considerations

This tool is designed for:
- ✅ Educational purposes
- ✅ Research applications
- ✅ Safety assessment workflows
- ✅ Content moderation

**Important**: This tool should not be the sole basis for safety decisions. Always consult domain experts and follow proper safety protocols.

## 📈 Technical Details

- **Model**: OpenAI GPT-4o
- **Approach**: Zero-shot classification with engineered prompts
- **Dataset**: ChemSafetyBench from Hugging Face
- **UI Framework**: Streamlit with custom CSS styling
- **Visualization**: Plotly for interactive charts

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