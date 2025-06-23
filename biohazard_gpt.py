"""
BioHazardGPT: A sophisticated RAG-powered risk classification tool
for detecting hazardous biomedical or chemical text content.
"""

import os
import openai
from dotenv import load_dotenv
import logging
import hashlib
import time
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioHazardRAG:
    """
    RAG-powered biomedical and chemical safety risk classification system.
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
        
        # Set up vector database
        self.setup_vector_database()
        
        # Safety categories
        self.categories = ["Safe", "Caution", "Hazardous"]
        
        # Dataset tracking
        self.datasets = {}
        self.dataset_loaded = False
    
    def setup_vector_database(self):
        """Set up ChromaDB vector database for similarity search"""
        # Create persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        collection_name = "chemical_safety_examples"
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Chemical and biological safety examples for RAG"}
            )
    
    def load_multiple_datasets(self) -> Dict:
        """Load multiple chemical safety datasets for comprehensive coverage"""
        if self.dataset_loaded:
            return self.datasets
            
        logger.info("Loading chemical safety datasets...")
        
        # Curated chemical safety examples
        self.datasets['curated_examples'] = self._create_curated_safety_examples()
        logger.info("✅ Created curated chemical safety examples")
        
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
    
    def _process_and_index_datasets(self):
        """Process all loaded datasets and index them in vector database"""
        logger.info("Processing and indexing datasets for RAG...")
        
        all_examples = []
        
        # Process curated examples
        if 'curated_examples' in self.datasets:
            for example in self.datasets['curated_examples']['examples']:
                all_examples.append({
                    'text': example['text'],
                    'classification': example['classification'], 
                    'reasoning': example.get('reasoning', ''),
                    'category': example.get('category', 'general'),
                    'source': 'curated',
                    'id': self._generate_id(example['text'], 'curated', example.get('category', 'general'))
                })
        
        logger.info(f"Using {len(all_examples)} curated examples for vector database")
        
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
        """Index examples in ChromaDB with embeddings"""
        if not examples:
            return
            
        texts = [ex['text'] for ex in examples]
        
        # Generate embeddings
        logger.info("Generating embeddings for examples...")
        embeddings = self.embedding_model.encode(texts)
        
        # Check if collection is empty
        collection_count = self.collection.count()
        
        if collection_count == 0:
            # Add all examples
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[{
                    'classification': ex['classification'],
                    'reasoning': ex['reasoning'],
                    'category': ex['category'], 
                    'source': ex['source']
                } for ex in examples],
                ids=[ex['id'] for ex in examples]
            )
            logger.info(f"Added {len(examples)} examples to vector database")
        else:
            logger.info(f"Vector database already contains {collection_count} examples")
    
    def retrieve_similar_examples(self, query_text: str, k: int = 5) -> List[Dict]:
        """Retrieve similar examples from vector database"""
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query_text])
        
        # Search for similar examples
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        similar_examples = []
        for i in range(len(results['documents'][0])):
            similar_examples.append({
                'text': results['documents'][0][i],
                'classification': results['metadatas'][0][i]['classification'],
                'reasoning': results['metadatas'][0][i]['reasoning'],
                'category': results['metadatas'][0][i]['category'],
                'source': results['metadatas'][0][i]['source'],
                'similarity_score': 1 - results['distances'][0][i]
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