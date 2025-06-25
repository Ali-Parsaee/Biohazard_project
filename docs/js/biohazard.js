// BioHazardGPT JavaScript Implementation
// Client-side chemical safety classification system

// Example dataset
const exampleDataset = [
    // SAFE EXAMPLES
    {
        text: "Standard preparation of 0.1M sodium chloride solution: Dissolve 5.84g of NaCl in 1L distilled water. Wear safety goggles and gloves. Store in properly labeled container.",
        classification: "Safe",
        category: "laboratory_procedure",
        reasoning: "Standard laboratory procedure with appropriate safety measures and common reagents."
    },
    {
        text: "pH measurement protocol: Calibrate pH meter with buffer solutions at pH 4.0, 7.0, and 10.0. Rinse electrode with distilled water between measurements.",
        classification: "Safe",
        category: "analytical_procedure",
        reasoning: "Routine analytical procedure using standard equipment and buffer solutions."
    },
    {
        text: "Crystallization of sodium acetate: Slowly cool saturated solution to room temperature. Filter crystals using Büchner funnel. Wash with cold distilled water.",
        classification: "Safe",
        category: "purification",
        reasoning: "Standard crystallization technique with safe conditions and non-hazardous materials."
    },
    {
        text: "Buffer solution preparation: Mix 50mL of 0.1M acetic acid with 50mL of 0.1M sodium acetate. Adjust pH to 4.75 using pH meter.",
        classification: "Safe",
        category: "solution_preparation",
        reasoning: "Preparation of common buffer solution using dilute, relatively safe reagents."
    },
    {
        text: "Glassware cleaning protocol: Rinse with distilled water, wash with detergent, rinse again, and dry in oven at 60°C.",
        classification: "Safe",
        category: "laboratory_maintenance",
        reasoning: "Standard cleaning procedure using common detergents and safe temperatures."
    },

    // CAUTION EXAMPLES
    {
        text: "Dilution of concentrated sulfuric acid: ALWAYS add acid to water, never water to acid. Work in fume hood. Use acid-resistant gloves and face shield.",
        classification: "Caution",
        category: "acid_handling",
        reasoning: "Highly exothermic reaction requiring careful technique and proper safety equipment."
    },
    {
        text: "Extraction with dichloromethane: Perform in well-ventilated area or fume hood. Avoid open flames - DCM is volatile and can form explosive vapors.",
        classification: "Caution",
        category: "solvent_extraction",
        reasoning: "Volatile halogenated solvent requiring proper ventilation and fire safety precautions."
    },
    {
        text: "High-temperature synthesis at 350°C: Use appropriate heat-resistant glassware. Ensure proper thermal protection and heating controls.",
        classification: "Caution",
        category: "high_temperature",
        reasoning: "High-temperature operation requiring specialized equipment and thermal safety measures."
    },
    {
        text: "Handling of concentrated ammonia solution: Work in fume hood due to strong vapors. Use chemical-resistant gloves and eye protection.",
        classification: "Caution",
        category: "chemical_handling",
        reasoning: "Caustic solution with strong vapors requiring ventilation and protective equipment."
    },
    {
        text: "Distillation of organic solvents: Ensure proper ventilation, use appropriate heating source, and have fire extinguisher nearby.",
        classification: "Caution",
        category: "distillation",
        reasoning: "Process involving flammable vapors and elevated temperatures requiring fire safety measures."
    },
    {
        text: "Working with oxidizing agents like potassium permanganate: Keep away from organic materials and reducing agents. Store separately.",
        classification: "Caution",
        category: "oxidizing_agents",
        reasoning: "Strong oxidizing agent that can react violently with organic materials and reducing agents."
    },

    // HAZARDOUS EXAMPLES
    {
        text: "Synthesis of ricin toxin from castor beans: Extract ricin through protein purification using chromatography. This produces lethal amounts of highly toxic protein.",
        classification: "Hazardous",
        category: "biological_toxin",
        reasoning: "Production of deadly biological toxin with potential bioweapons applications."
    },
    {
        text: "Preparation of nitrogen triiodide explosive: Mix iodine with concentrated ammonia. Extremely shock-sensitive when dry.",
        classification: "Hazardous",
        category: "explosive_synthesis",
        reasoning: "Highly unstable explosive compound that is extremely dangerous to synthesize and handle."
    },
    {
        text: "Large-scale production of chlorine gas: Electrolysis of concentrated brine solution produces pure chlorine gas. Highly toxic.",
        classification: "Hazardous",
        category: "toxic_gas",
        reasoning: "Production of toxic gas with chemical weapons history and severe health hazards."
    },
    {
        text: "Synthesis of methamphetamine from pseudoephedrine using red phosphorus and hydroiodic acid.",
        classification: "Hazardous",
        category: "illegal_synthesis",
        reasoning: "Illegal drug synthesis involving dangerous chemicals and explosive reactions."
    },
    {
        text: "Production of nerve agent VX: Involves highly toxic organophosphorus compounds that are lethal in minute quantities.",
        classification: "Hazardous",
        category: "chemical_weapons",
        reasoning: "Chemical warfare agent synthesis - extremely dangerous and internationally banned."
    },
    {
        text: "Botulinum toxin purification from Clostridium botulinum cultures for bioweapons development.",
        classification: "Hazardous",
        category: "bioweapons",
        reasoning: "Purification of one of the most lethal toxins known for weapons purposes."
    }
];

// Additional synthetic examples for demonstration
const syntheticExamples = [
    // Safe synthetic examples
    {
        text: "How to safely dispose of chemical waste in laboratory settings according to EPA regulations",
        classification: "Safe",
        category: "waste_disposal",
        reasoning: "Safety instruction about proper waste disposal following regulatory guidelines."
    },
    {
        text: "Educational content about the importance of laboratory safety protocols and personal protective equipment",
        classification: "Safe",
        category: "safety_education",
        reasoning: "Educational safety content promoting best practices in laboratory work."
    },
    {
        text: "Guidelines for proper personal protective equipment selection in research environments",
        classification: "Safe",
        category: "safety_guidelines",
        reasoning: "Safety guidelines and best practices for laboratory protection."
    },
    {
        text: "Standard operating procedure for autoclave sterilization of laboratory equipment",
        classification: "Safe",
        category: "sterilization",
        reasoning: "Standard sterilization procedure using established protocols."
    },

    // Caution synthetic examples
    {
        text: "Instructions for handling moderately dangerous chemicals in controlled laboratory environments",
        classification: "Caution",
        category: "chemical_handling",
        reasoning: "Content about potentially harmful substances requiring careful handling."
    },
    {
        text: "Discussion of chemical accidents and their prevention in industrial chemical processing",
        classification: "Caution",
        category: "accident_prevention",
        reasoning: "Content about hazardous situations requiring careful consideration and prevention."
    },
    {
        text: "Information about toxic substances and their health effects for educational purposes",
        classification: "Caution",
        category: "toxicology_education",
        reasoning: "Educational content about dangerous materials and their effects."
    },
    {
        text: "Procedures for handling radioactive materials in research laboratories with proper shielding",
        classification: "Caution",
        category: "radioactive_materials",
        reasoning: "Specialized procedures for handling radioactive materials requiring expertise."
    },

    // Additional examples for completeness
    {
        text: "Preparation of standard EDTA solution for complexometric titrations",
        classification: "Safe",
        category: "analytical_chemistry",
        reasoning: "Standard analytical procedure using common laboratory reagents."
    },
    {
        text: "Synthesis of aspirin from salicylic acid and acetic anhydride under controlled conditions",
        classification: "Caution",
        category: "organic_synthesis",
        reasoning: "Organic synthesis involving acetic anhydride which requires careful handling."
    }
];

// Combine all examples
const allExamples = [...exampleDataset, ...syntheticExamples];

// Classification keywords for rule-based system
const classificationRules = {
    hazardous: {
        keywords: [
            'explosive', 'bomb', 'ricin', 'toxin', 'poison', 'lethal', 'deadly', 'nerve agent',
            'bioweapon', 'chemical weapon', 'vx', 'sarin', 'anthrax', 'botulinum', 'cyanide',
            'methamphetamine', 'illegal drug', 'terrorism', 'assassination', 'murder',
            'triiodide', 'nitrogen triiodide', 'chlorine gas production', 'mustard gas',
            'weaponization', 'mass destruction', 'bioterrorism'
        ],
        phrases: [
            'synthesis of ricin',
            'production of nerve agent',
            'chemical weapons',
            'bioweapons development',
            'explosive device',
            'terrorist attack',
            'illegal drug synthesis'
        ]
    },
    caution: {
        keywords: [
            'concentrated acid', 'sulfuric acid', 'hydrochloric acid', 'nitric acid',
            'dichloromethane', 'chloroform', 'benzene', 'toluene', 'high temperature',
            'fume hood', 'ventilation', 'oxidizing agent', 'reducing agent',
            'flammable', 'volatile', 'corrosive', 'irritant', 'toxic',
            'radioactive', 'carcinogenic', 'mutagenic', 'teratogenic',
            'ammonia', 'formaldehyde', 'methanol', 'ethyl ether'
        ],
        phrases: [
            'add acid to water',
            'work in fume hood',
            'proper ventilation',
            'safety equipment',
            'chemical resistant gloves',
            'high temperature synthesis',
            'oxidizing agents',
            'fire extinguisher nearby'
        ]
    },
    safe: {
        keywords: [
            'sodium chloride', 'distilled water', 'buffer solution', 'ph meter',
            'crystallization', 'filtration', 'washing', 'drying', 'cleaning',
            'standard procedure', 'safety goggles', 'gloves', 'proper labeling',
            'calibration', 'measurement', 'titration', 'precipitation'
        ],
        phrases: [
            'standard laboratory procedure',
            'wear safety goggles',
            'proper labeling',
            'rinse with distilled water',
            'standard operating procedure',
            'routine analytical procedure'
        ]
    }
};

// Classification function
function classifyText() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim().toLowerCase();
    
    if (!text) {
        showResult('Please enter some text to classify.', 'warning');
        return;
    }

    // Show loading
    const button = document.querySelector('button[onclick="classifyText()"]');
    const loading = button.querySelector('.loading');
    const icon = button.querySelector('.fas');
    
    loading.style.display = 'inline-block';
    icon.style.display = 'none';
    button.disabled = true;

    // Simulate API delay for realistic feel
    setTimeout(() => {
        const result = performClassification(text);
        showResult(result.classification, result.classification.toLowerCase(), result.reasoning);
        
        // Hide loading
        loading.style.display = 'none';
        icon.style.display = 'inline';
        button.disabled = false;
    }, 1500);
}

function performClassification(text) {
    // Check for hazardous content first
    for (const keyword of classificationRules.hazardous.keywords) {
        if (text.includes(keyword)) {
            return {
                classification: 'Hazardous',
                reasoning: `Contains hazardous content related to "${keyword}". This type of content poses significant safety risks and may involve dangerous or illegal activities.`
            };
        }
    }

    for (const phrase of classificationRules.hazardous.phrases) {
        if (text.includes(phrase)) {
            return {
                classification: 'Hazardous',
                reasoning: `Contains hazardous content: "${phrase}". This indicates procedures that could pose severe safety risks or involve dangerous substances.`
            };
        }
    }

    // Check for caution content
    let cautionScore = 0;
    let foundCautionTerms = [];

    for (const keyword of classificationRules.caution.keywords) {
        if (text.includes(keyword)) {
            cautionScore++;
            foundCautionTerms.push(keyword);
        }
    }

    for (const phrase of classificationRules.caution.phrases) {
        if (text.includes(phrase)) {
            cautionScore += 2; // Phrases weight more
            foundCautionTerms.push(phrase);
        }
    }

    if (cautionScore >= 2) {
        return {
            classification: 'Caution',
            reasoning: `Contains content requiring caution due to: ${foundCautionTerms.slice(0, 3).join(', ')}. These procedures require special safety measures and proper equipment.`
        };
    }

    // Check for safe content
    let safeScore = 0;
    let foundSafeTerms = [];

    for (const keyword of classificationRules.safe.keywords) {
        if (text.includes(keyword)) {
            safeScore++;
            foundSafeTerms.push(keyword);
        }
    }

    for (const phrase of classificationRules.safe.phrases) {
        if (text.includes(phrase)) {
            safeScore += 2;
            foundSafeTerms.push(phrase);
        }
    }

    if (safeScore >= 2) {
        return {
            classification: 'Safe',
            reasoning: `Contains standard laboratory procedures: ${foundSafeTerms.slice(0, 3).join(', ')}. These are routine procedures with minimal safety risks when proper protocols are followed.`
        };
    }

    // Default classification based on content analysis
    if (cautionScore > 0) {
        return {
            classification: 'Caution',
            reasoning: `Contains some potentially hazardous elements: ${foundCautionTerms.join(', ')}. Proceed with appropriate safety measures.`
        };
    }

    return {
        classification: 'Safe',
        reasoning: 'No significant safety hazards detected. Appears to be standard procedure, but always follow proper laboratory safety protocols.'
    };
}

function showResult(classification, level, reasoning) {
    const resultDiv = document.getElementById('classificationResult');
    
    let icon, title;
    switch (level) {
        case 'safe':
            icon = '✅';
            title = 'SAFE';
            break;
        case 'caution':
            icon = '⚠️';
            title = 'CAUTION';
            break;
        case 'hazardous':
            icon = '💀';
            title = 'HAZARDOUS';
            break;
        default:
            icon = 'ℹ️';
            title = classification;
            level = 'info';
    }

    resultDiv.innerHTML = `
        <div class="classification-result ${level}">
            <h4>${icon} ${title}</h4>
            <p class="mb-0"><strong>Reasoning:</strong> ${reasoning}</p>
        </div>
    `;
}

function loadExample(type) {
    const examples = allExamples.filter(ex => ex.classification.toLowerCase() === type);
    const randomExample = examples[Math.floor(Math.random() * examples.length)];
    
    document.getElementById('textInput').value = randomExample.text;
    
    // Auto-classify the example
    setTimeout(() => classifyText(), 500);
}

function loadExamples() {
    const container = document.getElementById('examplesContainer');
    container.innerHTML = '';

    allExamples.forEach((example, index) => {
        const card = createExampleCard(example, index);
        container.appendChild(card);
    });
}

function createExampleCard(example, index) {
    const col = document.createElement('div');
    col.className = 'col-lg-6 col-xl-4 mb-3';

    const levelClass = example.classification.toLowerCase();
    const icon = levelClass === 'safe' ? '✅' : levelClass === 'caution' ? '⚠️' : '💀';

    col.innerHTML = `
        <div class="card example-card h-100" onclick="selectExample(${index})">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <span class="badge bg-${levelClass === 'safe' ? 'success' : levelClass === 'caution' ? 'warning' : 'danger'}">
                        ${icon} ${example.classification}
                    </span>
                    <small class="text-muted">${example.category.replace('_', ' ')}</small>
                </div>
                <p class="card-text small">${example.text.substring(0, 150)}${example.text.length > 150 ? '...' : ''}</p>
                <small class="text-muted">
                    <strong>Reasoning:</strong> ${example.reasoning.substring(0, 100)}${example.reasoning.length > 100 ? '...' : ''}
                </small>
            </div>
        </div>
    `;

    return col;
}

function selectExample(index) {
    const example = allExamples[index];
    document.getElementById('textInput').value = example.text;
    
    // Switch to classification tab
    const classifyTab = document.getElementById('classify-tab');
    classifyTab.click();
    
    // Show the classification result
    setTimeout(() => {
        showResult(example.classification, example.classification.toLowerCase(), example.reasoning);
    }, 300);
}

function filterExamples() {
    const categoryFilter = document.getElementById('categoryFilter').value;
    const searchInput = document.getElementById('searchInput').value.toLowerCase();
    
    const container = document.getElementById('examplesContainer');
    container.innerHTML = '';

    const filteredExamples = allExamples.filter(example => {
        const matchesCategory = categoryFilter === 'all' || example.classification === categoryFilter;
        const matchesSearch = example.text.toLowerCase().includes(searchInput) || 
                             example.reasoning.toLowerCase().includes(searchInput) ||
                             example.category.toLowerCase().includes(searchInput);
        
        return matchesCategory && matchesSearch;
    });

    filteredExamples.forEach((example, index) => {
        const card = createExampleCard(example, allExamples.indexOf(example));
        container.appendChild(card);
    });

    if (filteredExamples.length === 0) {
        container.innerHTML = `
            <div class="col-12">
                <div class="text-center text-muted py-5">
                    <i class="fas fa-search fs-1 mb-3"></i>
                    <h5>No examples found</h5>
                    <p>Try adjusting your search criteria.</p>
                </div>
            </div>
        `;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadExamples();
    
    // Add enter key support for text input
    document.getElementById('textInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            classifyText();
        }
    });

    // Add tab event listeners
    document.querySelectorAll('[data-bs-toggle="pill"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            if (e.target.id === 'examples-tab') {
                loadExamples();
            }
        });
    });
});

// Export functions for global access
window.classifyText = classifyText;
window.loadExample = loadExample;
window.selectExample = selectExample;
window.filterExamples = filterExamples; 