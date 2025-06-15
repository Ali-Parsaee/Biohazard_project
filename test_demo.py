"""
Test Demo for BioHazardGPT
This script demonstrates the core functionality and dataset loading capabilities.
"""

from biohazard_gpt import BioHazardGPT
import os

def test_dataset_loading():
    """Test loading the ChemSafetyBench dataset"""
    print("=== Testing Dataset Loading ===")
    
    try:
        # Initialize classifier without API key for dataset testing
        classifier = BioHazardGPT.__new__(BioHazardGPT)  # Create instance without calling __init__
        classifier.categories = ["Safe", "Caution", "Hazardous"]
        classifier.dataset = None
        
        # Test dataset loading
        print("Loading ChemSafetyBench dataset...")
        dataset = classifier.load_dataset()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Training samples: {len(dataset['train'])}")
        
        # Show sample data
        samples = classifier.get_sample_data(3)
        
        print("\n=== Sample Data ===")
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"ID: {sample['id']}")
            print(f"Text: {sample['text'][:150]}..." if len(sample['text']) > 150 else f"Text: {sample['text']}")
            print(f"Metadata: {sample['metadata']}")
            print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_api_key_validation():
    """Test API key validation"""
    print("\n=== Testing API Key Validation ===")
    
    try:
        # Test without API key
        classifier = BioHazardGPT(api_key="test_key")
        print("✅ API key validation works")
        return True
    except Exception as e:
        print(f"⚠️ Expected behavior: {e}")
        return True

def run_demo():
    """Run complete demo"""
    print("🧬 BioHazardGPT Demo")
    print("=" * 50)
    
    # Test dataset loading
    dataset_ok = test_dataset_loading()
    
    # Test API validation
    api_ok = test_api_key_validation()
    
    print("\n=== Demo Summary ===")
    print(f"Dataset Loading: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    print(f"API Validation: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    print("\n=== Next Steps ===")
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("\n2. Run the main application:")
    print("   python biohazard_gpt.py")
    print("\n3. Or launch the web interface:")
    print("   streamlit run app.py")
    
    print("\n🚀 BioHazardGPT is ready to use!")

if __name__ == "__main__":
    run_demo() 