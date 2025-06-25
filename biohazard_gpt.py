"""
BioHazardGPT: A sophisticated RAG-powered risk classification tool
for detecting hazardous biomedical or chemical text content.
"""

# Set environment variables BEFORE any imports to avoid protobuf conflicts
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import openai
from dotenv import load_dotenv
import logging
import hashlib
import time
import pickle
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioHazardRAG:
    """
    RAG-powered biomedical and chemical safety risk classification system.
    Uses FAISS for vector similarity search instead of ChromaDB for better deployment compatibility.
    """
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize BioHazardRAG with OpenAI API and embedding model"""
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Set up vector database using FAISS
        self.setup_vector_database()
        
        # Safety categories
        self.categories = ["Safe", "Caution", "Hazardous"]
        
        # Dataset tracking
        self.datasets = {}
        self.dataset_loaded = False
    
    def clear_vector_database(self):
        """Clear the vector database to reload fresh data"""
        try:
            if hasattr(self, 'faiss_index') and self.faiss_index is not None:
                logger.info(f"Cleared {self.faiss_index.ntotal} examples from vector database")
            
            # Reset FAISS index
            self.faiss_index = None
            self.example_texts = []
            self.example_metadata = []
            
            # Reset dataset loaded flag
            self.dataset_loaded = False
            self.datasets = {}
            logger.info("Created fresh vector database")
                
        except Exception as e:
            logger.warning(f"Failed to clear vector database: {e}")
    
    def setup_vector_database(self):
        """Set up FAISS vector database for similarity search"""
        # Initialize FAISS index (will be created when first examples are added)
        self.faiss_index = None
        self.example_texts = []
        self.example_metadata = []
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Try to load existing index
        try:
            self.load_vector_database()
        except:
            logger.info("No existing vector database found, will create new one")

    def save_vector_database(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.faiss_index is not None and self.faiss_index.ntotal > 0:
                # Create directory if it doesn't exist
                os.makedirs("./faiss_db", exist_ok=True)
                
                # Save FAISS index
                faiss.write_index(self.faiss_index, "./faiss_db/index.faiss")
                
                # Save metadata
                with open("./faiss_db/metadata.pkl", "wb") as f:
                    pickle.dump({
                        'texts': self.example_texts,
                        'metadata': self.example_metadata
                    }, f)
                
                logger.info(f"Saved vector database with {self.faiss_index.ntotal} examples")
        except Exception as e:
            logger.warning(f"Failed to save vector database: {e}")

    def load_vector_database(self):
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index("./faiss_db/index.faiss")
            
            # Load metadata
            with open("./faiss_db/metadata.pkl", "rb") as f:
                data = pickle.load(f)
                self.example_texts = data['texts']
                self.example_metadata = data['metadata']
            
            logger.info(f"Loaded vector database with {self.faiss_index.ntotal} examples")
        except Exception as e:
            logger.info("No existing vector database found")
            self.faiss_index = None
            self.example_texts = []
            self.example_metadata = []
    
    def load_multiple_datasets(self) -> Dict:
        """Load multiple chemical safety datasets for comprehensive coverage"""
        if self.dataset_loaded:
            return self.datasets
            
        logger.info("Loading chemical safety datasets...")
        
        # Curated chemical safety examples
        self.datasets['curated_examples'] = self._create_curated_safety_examples()
        logger.info("✅ Created curated chemical safety examples")
        
        # Use only synthetic datasets for maximum speed (no downloads)
        logger.info("Creating synthetic datasets for fast RAG system...")
        
        # 1. Synthetic toxic content examples
        logger.info("Creating synthetic toxic content examples...")
        self.datasets['toxic_chat'] = self._create_synthetic_toxic_examples()
        logger.info("✅ Created toxic content dataset")
        
        # 2. Synthetic chemistry examples
        logger.info("Creating synthetic chemistry examples...")
        self.datasets['chemistry_qa'] = self._create_synthetic_chemistry_examples()
        logger.info("✅ Created chemistry Q&A dataset")
        
        # 3. Synthetic safety examples
        logger.info("Creating synthetic safety examples...")
        self.datasets['safety_instructions'] = self._create_synthetic_safety_examples()
        logger.info("✅ Created safety instructions dataset")
        
        # Process and index all datasets
        self._process_and_index_datasets()
        self.dataset_loaded = True
        
        return self.datasets
    
    def _create_curated_safety_examples(self):
        """Create high-quality curated examples for different safety categories"""
        examples = [
            # SAFE EXAMPLES
            {
                "text": "Standard preparation of 0.1M sodium chloride solution: Dissolve 5.84g of NaCl in 1L distilled water. Wear safety goggles and gloves. Store in properly labeled container.",
                "classification": "Safe",
                "category": "laboratory_procedure",
                "reasoning": "Standard laboratory procedure with appropriate safety measures"
            },
            {
                "text": "pH measurement protocol: Calibrate pH meter with buffer solutions at pH 4.0, 7.0, and 10.0. Rinse electrode with distilled water between measurements.",
                "classification": "Safe", 
                "category": "analytical_procedure",
                "reasoning": "Routine analytical procedure with standard equipment"
            },
            {
                "text": "Crystallization of sodium acetate: Slowly cool saturated solution to room temperature. Filter crystals using Büchner funnel. Wash with cold distilled water.",
                "classification": "Safe",
                "category": "purification",
                "reasoning": "Standard crystallization technique with safe conditions"
            },
            # CAUTION EXAMPLES  
            {
                "text": "Dilution of concentrated sulfuric acid: ALWAYS add acid to water, never water to acid. Work in fume hood. Use acid-resistant gloves and face shield.",
                "classification": "Caution",
                "category": "acid_handling", 
                "reasoning": "Requires careful handling due to highly exothermic reaction"
            },
            {
                "text": "Extraction with dichloromethane: Perform in well-ventilated area or fume hood. Avoid open flames - DCM is volatile and can form explosive vapors.",
                "classification": "Caution",
                "category": "solvent_extraction",
                "reasoning": "Volatile halogenated solvent requiring proper ventilation"
            },
            {
                "text": "High-temperature synthesis at 350°C: Use appropriate heat-resistant glassware. Ensure proper thermal protection.",
                "classification": "Caution", 
                "category": "high_temperature",
                "reasoning": "High temperature operation requiring specialized equipment"
            },
            # HAZARDOUS EXAMPLES
            {
                "text": "Synthesis of ricin toxin from castor beans: Extract ricin through protein purification using chromatography. This produces lethal amounts of highly toxic protein.",
                "classification": "Hazardous",
                "category": "biological_toxin",
                "reasoning": "Production of deadly biological toxin with potential weapons applications"
            },
            {
                "text": "Preparation of nitrogen triiodide explosive: Mix iodine with concentrated ammonia. Extremely shock-sensitive when dry.",
                "classification": "Hazardous", 
                "category": "explosive_synthesis",
                "reasoning": "Highly unstable explosive compound dangerous to synthesize"
            },
            {
                "text": "Large-scale production of chlorine gas: Electrolysis of concentrated brine solution produces pure chlorine gas. Highly toxic.",
                "classification": "Hazardous",
                "category": "toxic_gas",
                "reasoning": "Production of toxic gas with chemical weapons history"
            }
        ]
        
        return {"examples": examples}
    
    def _load_small_toxic_dataset(self):
        """Load a very small toxic chat dataset quickly"""
        try:
            # Load only 100 examples for speed
            dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train[:100]", trust_remote_code=True)
            
            processed_examples = []
            for example in dataset:
                toxicity = example.get('toxicity', 0)
                if toxicity >= 0.7:
                    classification = "Hazardous"
                elif toxicity >= 0.3:
                    classification = "Caution"
                else:
                    classification = "Safe"
                
                processed_examples.append({
                    'text': example.get('user_input', '')[:500],  # Limit text length
                    'classification': classification,
                    'reasoning': f"Toxicity score: {toxicity:.2f}",
                    'category': 'content_safety',
                    'source': 'toxic_chat'
                })
            
            logger.info(f"Processed {len(processed_examples)} toxic chat examples")
            return {'examples': processed_examples}
            
        except Exception as e:
            logger.error(f"Failed to load toxic chat dataset: {e}")
            return None
    
    def _create_synthetic_toxic_examples(self):
        """Create synthetic toxic content examples"""
        synthetic_examples = [
            # SAFE CONTENT
            {
                'text': "How to safely dispose of chemical waste in laboratory settings according to regulations",
                'classification': "Safe",
                'reasoning': "Safety instruction about proper waste disposal",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            {
                'text': "Educational content about the importance of laboratory safety protocols",
                'classification': "Safe",
                'reasoning': "Educational safety content",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            {
                'text': "Guidelines for proper personal protective equipment in research environments",
                'classification': "Safe",
                'reasoning': "Safety guidelines and best practices",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            
            # CAUTION CONTENT
            {
                'text': "Instructions for handling moderately dangerous chemicals in controlled environments",
                'classification': "Caution", 
                'reasoning': "Content about potentially harmful substances",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            {
                'text': "Discussion of chemical accidents and their prevention in industrial settings",
                'classification': "Caution",
                'reasoning': "Content about hazardous situations requiring careful consideration",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            {
                'text': "Information about toxic substances and their health effects for educational purposes",
                'classification': "Caution",
                'reasoning': "Educational content about dangerous materials",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            
            # HAZARDOUS CONTENT
            {
                'text': "Detailed instructions for creating explosive devices for terrorist activities",
                'classification': "Hazardous",
                'reasoning': "Content about dangerous explosive materials with malicious intent",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            {
                'text': "Step-by-step guide for producing biological weapons and toxins",
                'classification': "Hazardous",
                'reasoning': "Bioweapons production information",
                'category': 'content_safety',
                'source': 'synthetic'
            },
            {
                'text': "Instructions for manufacturing illegal drugs and narcotics",
                'classification': "Hazardous",
                'reasoning': "Illegal drug production content",
                'category': 'content_safety',
                'source': 'synthetic'
            }
        ]
        
        logger.info(f"Created {len(synthetic_examples)} synthetic toxic examples")
        return {'examples': synthetic_examples}
    
    def _load_small_chemistry_dataset(self):
        """Load a very small chemistry dataset quickly"""
        try:
            # Try a lightweight chemistry dataset - only 50 examples for speed
            dataset = load_dataset("camel-ai/chemistry", split="train[:50]")
            
            processed_examples = []
            for example in dataset:
                # Extract text content
                if 'message_1' in example and 'message_2' in example:
                    text = f"{example['message_1']} {example['message_2']}"
                else:
                    text = str(example)[:300]
                
                # Infer safety classification
                classification = self._infer_classification(text)
                
                processed_examples.append({
                    'text': text[:500],  # Limit text length for speed
                    'classification': classification,
                    'reasoning': f"Chemistry content classification",
                    'category': 'chemistry',
                    'source': 'camel_chemistry'
                })
            
            logger.info(f"Processed {len(processed_examples)} chemistry examples")
            return {'examples': processed_examples}
            
        except Exception as e:
            logger.error(f"Failed to load chemistry dataset: {e}")
            return None
    
    def _create_synthetic_safety_examples(self):
        """Create synthetic safety instruction examples"""
        safety_examples = [
            # SAFE SAFETY INSTRUCTIONS
            {
                'text': "Always wear appropriate personal protective equipment when handling chemicals in the laboratory",
                'classification': "Safe",
                'reasoning': "Standard safety instruction",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "How to safely store and label chemical reagents in laboratory settings",
                'classification': "Safe",
                'reasoning': "Standard laboratory safety procedures",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "Proper procedures for washing hands and decontamination after laboratory work",
                'classification': "Safe",
                'reasoning': "Basic hygiene and safety procedures",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "Guidelines for maintaining clean and organized laboratory workspaces",
                'classification': "Safe",
                'reasoning': "Laboratory organization and safety",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            
            # CAUTION SAFETY INSTRUCTIONS
            {
                'text': "Procedures for handling concentrated acids require special ventilation and protective equipment",
                'classification': "Caution",
                'reasoning': "Safety instruction for potentially dangerous procedures",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "Emergency procedures for chemical spills and exposure incidents",
                'classification': "Caution",
                'reasoning': "Emergency safety procedures",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "Safety protocols for working with radioactive materials in research facilities",
                'classification': "Caution",
                'reasoning': "Procedures for handling dangerous radioactive substances",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "Guidelines for handling biological samples that may contain pathogens",
                'classification': "Caution",
                'reasoning': "Biosafety procedures for potentially infectious materials",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            
            # HAZARDOUS SAFETY-RELATED CONTENT
            {
                'text': "Manufacturing processes for military-grade chemical weapons and nerve agents",
                'classification': "Hazardous",
                'reasoning': "Content about dangerous weapons production",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "Instructions for bypassing laboratory safety systems and protocols",
                'classification': "Hazardous",
                'reasoning': "Content that undermines safety measures",
                'category': 'safety_instructions',
                'source': 'synthetic'
            },
            {
                'text': "How to create improvised explosive devices using laboratory chemicals",
                'classification': "Hazardous",
                'reasoning': "Dangerous explosive creation instructions",
                'category': 'safety_instructions',
                'source': 'synthetic'
            }
        ]
        
        logger.info(f"Created {len(safety_examples)} synthetic safety examples")
        return {'examples': safety_examples}
    
    def _create_synthetic_chemistry_examples(self):
        """Create synthetic chemistry examples as fallback"""
        synthetic_examples = [
            # SAFE CHEMISTRY EXAMPLES
            {
                'text': "How do you prepare a standard solution of sodium chloride for analytical chemistry?",
                'classification': "Safe", 
                'reasoning': "Basic chemistry procedure with safe materials",
                'category': 'chemistry_basic',
                'source': 'synthetic'
            },
            {
                'text': "What is the procedure for calibrating a pH meter using buffer solutions?",
                'classification': "Safe",
                'reasoning': "Standard analytical procedure",
                'category': 'chemistry_basic',
                'source': 'synthetic'
            },
            {
                'text': "How do you perform a simple crystallization of salt from solution?",
                'classification': "Safe",
                'reasoning': "Basic purification technique",
                'category': 'chemistry_basic',
                'source': 'synthetic'
            },
            {
                'text': "What are the steps for preparing a buffer solution in the laboratory?",
                'classification': "Safe",
                'reasoning': "Standard laboratory procedure",
                'category': 'chemistry_basic',
                'source': 'synthetic'
            },
            {
                'text': "How do you properly clean glassware for analytical chemistry experiments?",
                'classification': "Safe",
                'reasoning': "Basic laboratory maintenance",
                'category': 'chemistry_basic',
                'source': 'synthetic'
            },
            
            # CAUTION CHEMISTRY EXAMPLES
            {
                'text': "What is the proper way to handle concentrated acids in a laboratory setting?",
                'classification': "Caution",
                'reasoning': "Chemistry safety question about dangerous chemicals",
                'category': 'chemistry_safety',
                'source': 'synthetic'
            },
            {
                'text': "What precautions should be taken when working with organic solvents like benzene?",
                'classification': "Caution",
                'reasoning': "Safety question about potentially harmful solvents",
                'category': 'chemistry_safety',
                'source': 'synthetic'
            },
            {
                'text': "How do you safely perform distillation of volatile organic compounds?",
                'classification': "Caution",
                'reasoning': "Procedure involving potentially dangerous materials",
                'category': 'chemistry_safety',
                'source': 'synthetic'
            },
            {
                'text': "What safety measures are needed when heating reactions above 200°C?",
                'classification': "Caution",
                'reasoning': "High temperature procedures require special precautions",
                'category': 'chemistry_safety',
                'source': 'synthetic'
            },
            {
                'text': "How do you handle and store oxidizing agents in the laboratory?",
                'classification': "Caution",
                'reasoning': "Oxidizing agents can be reactive and dangerous",
                'category': 'chemistry_safety',
                'source': 'synthetic'
            },
            
            # HAZARDOUS CHEMISTRY EXAMPLES
            {
                'text': "Describe the synthesis of explosive compounds for military applications.",
                'classification': "Hazardous",
                'reasoning': "Discussion of explosive synthesis - potentially dangerous",
                'category': 'chemistry_hazardous',
                'source': 'synthetic'
            },
            {
                'text': "How to synthesize ricin toxin from castor bean extract for research purposes?",
                'classification': "Hazardous",
                'reasoning': "Synthesis of deadly biological toxin",
                'category': 'chemistry_hazardous',
                'source': 'synthetic'
            },
            {
                'text': "What are the steps to produce chlorine gas in large quantities?",
                'classification': "Hazardous",
                'reasoning': "Production of toxic gas with weapons potential",
                'category': 'chemistry_hazardous',
                'source': 'synthetic'
            },
            {
                'text': "How do you synthesize methamphetamine from over-the-counter medications?",
                'classification': "Hazardous",
                'reasoning': "Illegal drug synthesis information",
                'category': 'chemistry_hazardous',
                'source': 'synthetic'
            },
            {
                'text': "Procedures for creating nerve agents and chemical warfare compounds.",
                'classification': "Hazardous",
                'reasoning': "Chemical weapons production information",
                'category': 'chemistry_hazardous',
                'source': 'synthetic'
            }
        ]
        
        logger.info(f"Created {len(synthetic_examples)} synthetic chemistry examples")
        return {'examples': synthetic_examples}
    
    def _process_and_index_datasets(self):
        """Process all loaded datasets and index them in vector database"""
        logger.info("Processing and indexing datasets for RAG...")
        
        all_examples = []
        
        # Process all loaded datasets
        for dataset_name, dataset in self.datasets.items():
            if 'examples' in dataset:
                dataset_examples = []
                for example in dataset['examples']:
                    processed_example = {
                        'text': example['text'],
                        'classification': example['classification'], 
                        'reasoning': example.get('reasoning', ''),
                        'category': example.get('category', 'general'),
                        'source': example.get('source', dataset_name),
                        'id': self._generate_id(example['text'], example.get('source', dataset_name), example.get('category', 'general'))
                    }
                    dataset_examples.append(processed_example)
                    all_examples.append(processed_example)
                
                logger.info(f"Processed {len(dataset_examples)} examples from {dataset_name}")
        
        logger.info(f"Total examples for vector database: {len(all_examples)}")
        
        # Index examples in vector database
        if all_examples:
            self._index_examples(all_examples)
        
        logger.info(f"✅ Indexed {len(all_examples)} examples in vector database")
    
    def _generate_id(self, text: str, source: str = "", category: str = "") -> str:
        """Generate unique ID for text"""
        unique_string = f"{text}_{source}_{category}_{len(text)}_{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _infer_classification(self, text: str) -> str:
        """Infer classification based on text content keywords"""
        text_lower = text.lower()
        
        hazardous_keywords = ['weapon', 'explosive', 'toxin', 'poison', 'lethal', 'deadly', 'warfare', 'bomb']
        caution_keywords = ['synthesis', 'chemical', 'acid', 'reactive', 'volatile', 'toxic', 'carcinogen']
        
        if any(keyword in text_lower for keyword in hazardous_keywords):
            return "Hazardous"
        elif any(keyword in text_lower for keyword in caution_keywords):
            return "Caution"
        else:
            return "Safe"
    
    def _index_examples(self, examples: List[Dict]):
        """Index examples in FAISS with embeddings"""
        if not examples:
            return
            
        texts = [ex['text'] for ex in examples]
        
        # Generate embeddings in batches for better performance
        logger.info(f"Generating embeddings for {len(examples)} examples...")
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings.tolist())
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Check if FAISS index is empty or if we should reset it
        if self.faiss_index is None or self.faiss_index.ntotal == 0:  # Reset if empty
            # Clear existing data
            self.faiss_index = None
            self.example_texts = []
            self.example_metadata = []
            
            # Add all examples in batches
            batch_size = 100
            for i in range(0, len(examples), batch_size):
                batch_examples = examples[i:i+batch_size]
                batch_embeddings = all_embeddings[i:i+batch_size]
                batch_texts = texts[i:i+batch_size]
                
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                self.faiss_index.add(np.array(batch_embeddings))
                self.example_texts.extend(batch_texts)
                self.example_metadata.extend([{
                    'classification': ex['classification'],
                    'reasoning': ex['reasoning'],
                    'category': ex['category'], 
                    'source': ex['source']
                } for ex in batch_examples])
            
            self.save_vector_database()
            logger.info(f"✅ Successfully indexed {len(examples)} examples in vector database")
        else:
            logger.info(f"Vector database already contains {self.faiss_index.ntotal} examples - skipping reindexing")
    
    def retrieve_similar_examples(self, query_text: str, k: int = 5) -> List[Dict]:
        """Retrieve similar examples from vector database"""
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query_text])
        
        # Search for similar examples
        distances, indices = self.faiss_index.search(np.array(query_embedding).reshape(1, -1), k)
        
        similar_examples = []
        for i in range(len(indices[0])):
            similar_examples.append({
                'text': self.example_texts[indices[0][i]],
                'classification': self.example_metadata[indices[0][i]]['classification'],
                'reasoning': self.example_metadata[indices[0][i]]['reasoning'],
                'category': self.example_metadata[indices[0][i]]['category'],
                'source': self.example_metadata[indices[0][i]]['source'],
                'similarity_score': 1 - distances[0][i]
            })
        
        return similar_examples
    
    def get_rag_enhanced_prompt(self, text: str, similar_examples: List[Dict]) -> str:
        """Create RAG-enhanced system prompt with retrieved examples"""
        
        base_prompt = """You are BioHazardGPT, a specialized AI assistant for chemical and biosafety risk assessment.

Your task is to classify text content based on potential safety risks:
- "Safe": Content poses no significant safety concerns
- "Caution": Content contains information that requires careful handling
- "Hazardous": Content describes dangerous procedures or substances

Use the following similar examples from our safety database:

RETRIEVED EXAMPLES:
"""
        
        # Add retrieved examples
        for i, example in enumerate(similar_examples, 1):
            base_prompt += f"\nExample {i} (Similarity: {example['similarity_score']:.3f}):\n"
            base_prompt += f"Text: {example['text'][:200]}{'...' if len(example['text']) > 200 else ''}\n"
            base_prompt += f"Classification: {example['classification']}\n"
            base_prompt += f"Reasoning: {example['reasoning']}\n"
        
        base_prompt += f"\nNow classify this text:\n\nText to classify: {text}\n\nRespond with only one word: Safe, Caution, or Hazardous."
        
        return base_prompt
    
    def classify_text_with_rag(self, text: str, include_reasoning: bool = False, k_similar: int = 3) -> Dict:
        """Classify text using RAG"""
        try:
            # Load datasets if not already loaded
            if not self.dataset_loaded:
                self.load_multiple_datasets()
            
            # Retrieve similar examples
            similar_examples = self.retrieve_similar_examples(text, k_similar)
            
            # Create RAG-enhanced prompt
            rag_prompt = self.get_rag_enhanced_prompt(text, similar_examples)
            
            messages = [
                {"role": "system", "content": rag_prompt},
                {"role": "user", "content": f"Classify: {text}"}
            ]
            
            if include_reasoning:
                messages[0]["content"] += "\n\nAfter your classification, provide a brief explanation on a new line starting with 'Reasoning:'."
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=200 if include_reasoning else 10
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse response
            if include_reasoning and "Reasoning:" in content:
                parts = content.split("Reasoning:")
                classification = parts[0].strip()
                reasoning = parts[1].strip() if len(parts) > 1 else ""
            else:
                classification = content
                reasoning = ""
            
            # Validate classification
            if classification not in self.categories:
                logger.warning(f"Unexpected classification: {classification}")
                classification = "Caution"
            
            result = {
                "classification": classification,
                "text": text,
                "model": "gpt-4o-rag",
                "retrieved_examples": similar_examples,
                "rag_enhanced": True
            }
            
            if include_reasoning:
                result["reasoning"] = reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"RAG classification failed: {e}")
            return {
                "classification": "Caution",
                "text": text,
                "error": str(e),
                "model": "gpt-4o-rag"
            }
    
    def classify_text(self, text: str, include_reasoning: bool = False) -> Dict:
        """Basic classification without RAG"""
        
        messages = [
            {"role": "system", "content": """You are BioHazardGPT, a specialized AI assistant for chemical and biosafety risk assessment.

Your task is to classify text content based on potential safety risks:
- "Safe": Content poses no significant safety concerns
- "Caution": Content contains information that requires careful handling
- "Hazardous": Content describes dangerous procedures or substances

Respond with only one word: Safe, Caution, or Hazardous."""},
            {"role": "user", "content": f"Classify: {text}"}
        ]
        
        if include_reasoning:
            messages[0]["content"] += "\n\nAfter your classification, provide a brief explanation on a new line starting with 'Reasoning:'."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=200 if include_reasoning else 10
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse response
            if include_reasoning and "Reasoning:" in content:
                parts = content.split("Reasoning:")
                classification = parts[0].strip()
                reasoning = parts[1].strip() if len(parts) > 1 else ""
            else:
                classification = content
                reasoning = ""
            
            # Validate classification
            if classification not in self.categories:
                logger.warning(f"Unexpected classification: {classification}")
                classification = "Caution"
            
            result = {
                "classification": classification,
                "text": text,
                "model": "gpt-4o"
            }
            
            if include_reasoning:
                result["reasoning"] = reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "classification": "Caution",
                "text": text,
                "error": str(e),
                "model": "gpt-4o"
            }

    def get_sample_data(self, num_samples: int = 10) -> List[Dict]:
        """Get sample data for testing"""
        if not self.dataset_loaded:
            self.load_multiple_datasets()
        
        samples = []
        if 'curated_examples' in self.datasets:
            curated = self.datasets['curated_examples']['examples'][:num_samples]
            for example in curated:
                samples.append({
                    'text': example['text'],
                    'expected_label': example['classification'],
                    'category': example.get('category', 'general'),
                    'source': 'curated'
                })
        
        return samples

def main():
    """Test the RAG system"""
    classifier = BioHazardRAG()
    
    test_text = "Prepare 0.1M sodium chloride solution with proper safety equipment"
    result = classifier.classify_text(test_text, include_reasoning=True)
    
    print(f"Text: {test_text}")
    print(f"Classification: {result['classification']}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")

if __name__ == "__main__":
    main() 