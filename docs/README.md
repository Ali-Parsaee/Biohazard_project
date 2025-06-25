# BioHazardGPT - GitHub Pages Version

🌐 **Live Demo**: [https://ali-parsaee.github.io/Biohazard_project/](https://ali-parsaee.github.io/Biohazard_project/)

## Overview

This is a client-side implementation of BioHazardGPT - an AI-powered chemical and biological safety classification system. Unlike the Streamlit version, this runs entirely in your browser without any server dependencies.

## Features

### 🔍 **Text Classification**
- Real-time safety classification of chemical procedures
- Three-tier risk assessment: Safe, Caution, Hazardous
- Detailed reasoning for each classification
- Rule-based classification system

### 📊 **Example Dataset**
- 44+ curated examples across different safety categories
- Interactive filtering by category and search
- Click-to-test functionality
- Comprehensive coverage of laboratory procedures

### 📱 **Modern Web Interface**
- Responsive design that works on all devices
- Bootstrap 5 styling with custom animations
- Font Awesome icons for better UX
- Smooth transitions and hover effects

## Usage

### Quick Start
1. Visit the [live demo](https://ali-parsaee.github.io/Biohazard_project/)
2. Enter chemical procedures or safety instructions in the text area
3. Click "Classify Safety Level" to get instant results
4. Browse example datasets for reference

### Example Classifications

**Safe Example:**
```
Standard preparation of 0.1M sodium chloride solution: 
Dissolve 5.84g of NaCl in 1L distilled water. 
Wear safety goggles and gloves.
```

**Caution Example:**
```
Dilution of concentrated sulfuric acid: 
ALWAYS add acid to water, never water to acid. 
Work in fume hood with proper protection.
```

**Hazardous Example:**
```
Production of toxic substances for harmful purposes
```

## Technical Details

### Architecture
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Bootstrap 5.1.3
- **Icons**: Font Awesome 6.0.0
- **Deployment**: GitHub Pages
- **No Backend**: Completely client-side application

### Classification Algorithm
The system uses a rule-based approach with:
- Keyword matching for different risk levels
- Phrase pattern recognition
- Scoring system for classification confidence
- Contextual reasoning generation

### File Structure
```
docs/
├── index.html          # Main application page
├── js/
│   └── biohazard.js   # Classification logic and dataset
├── css/
│   └── styles.css     # Custom styling
└── README.md          # This file
```

## Advantages over Streamlit Version

✅ **No Deployment Issues**: No server dependencies or package conflicts  
✅ **Instant Loading**: Fast, responsive web interface  
✅ **Always Available**: Hosted on reliable GitHub Pages  
✅ **Mobile Friendly**: Works perfectly on all devices  
✅ **No API Keys**: No need for OpenAI API configuration  
✅ **Offline Capable**: Can work without internet after initial load  

## Development

### Local Development
1. Clone the repository
2. Navigate to the `docs/` directory
3. Open `index.html` in your browser
4. Or serve with a local web server:
   ```bash
   python -m http.server 8000
   # Visit http://localhost:8000
   ```

### Customization
- **Add Examples**: Edit the `exampleDataset` array in `js/biohazard.js`
- **Modify Rules**: Update `classificationRules` object for different classification logic
- **Styling**: Customize CSS variables in the `<style>` section or `css/styles.css`

## Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+
- ✅ Mobile browsers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in the `docs/` directory
4. Test thoroughly in multiple browsers
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Bootstrap team for the excellent CSS framework
- Font Awesome for the comprehensive icon library
- GitHub Pages for reliable hosting

---

**Note**: This GitHub Pages version provides the same core functionality as the Streamlit application but with better reliability and accessibility. Perfect for demonstrations, sharing, and educational purposes. 