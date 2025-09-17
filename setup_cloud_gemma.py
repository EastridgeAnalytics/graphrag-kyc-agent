#!/usr/bin/env python3
"""
Cloud Gemma Setup Script

This script helps you set up cloud-based Gemma models for Text-to-Cypher translation.
No local installation required!
"""

import os
import sys

def setup_google_ai_studio():
    """Setup Google AI Studio (Easiest option)"""
    print("\n🌐 GOOGLE AI STUDIO SETUP")
    print("=" * 30)
    print("✅ Free tier available")
    print("✅ Official Google Gemma models")
    print("✅ Fast and reliable")
    print("✅ Easy setup")
    
    print("\n📝 Steps:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the API key")
    
    api_key = input("\n🔑 Paste your Google AI API key here: ").strip()
    
    if api_key:
        env_content = f"""# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Google AI Studio Configuration
GOOGLE_API_KEY={api_key}
GEMMA_MODEL=gemini-pro

# Agent Configuration
LOG_LEVEL=INFO
EVOLUTION_THRESHOLD=0.6
PERFORMANCE_WINDOW=10

# Text-to-Cypher Configuration
CYPHER_CONFIDENCE_THRESHOLD=0.6
CYPHER_MAX_TOKENS=2048
CYPHER_TEMPERATURE=0.2
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Google AI Studio configuration saved!")
        return True
    else:
        print("❌ No API key provided")
        return False

def setup_huggingface():
    """Setup Hugging Face"""
    print("\n🤗 HUGGING FACE SETUP")
    print("=" * 21)
    print("✅ Free tier available")
    print("✅ Direct access to Gemma models")
    print("✅ Good for development")
    
    print("\n📝 Steps:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Sign up/login to Hugging Face")
    print("3. Create a new token")
    print("4. Copy the token")
    
    api_key = input("\n🔑 Paste your Hugging Face API token here: ").strip()
    
    if api_key:
        env_content = f"""# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Hugging Face Configuration
HUGGINGFACE_API_KEY={api_key}
GEMMA_MODEL=google/gemma-2-9b-it

# Agent Configuration
LOG_LEVEL=INFO
EVOLUTION_THRESHOLD=0.6
PERFORMANCE_WINDOW=10

# Text-to-Cypher Configuration
CYPHER_CONFIDENCE_THRESHOLD=0.6
CYPHER_MAX_TOKENS=2048
CYPHER_TEMPERATURE=0.2
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Hugging Face configuration saved!")
        return True
    else:
        print("❌ No API key provided")
        return False

def setup_openai_compatible():
    """Setup OpenAI-compatible providers"""
    print("\n⚡ OPENAI-COMPATIBLE PROVIDERS")
    print("=" * 32)
    print("Various cloud providers offer Gemma through OpenAI-compatible APIs:")
    print("• Together AI")
    print("• Fireworks AI")
    print("• Groq")
    print("• DeepInfra")
    print("• And many others...")
    
    print("\n📝 Choose your provider:")
    providers = {
        "1": {"name": "Together AI", "url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-2-7b-chat-hf"},
        "2": {"name": "Fireworks AI", "url": "https://api.fireworks.ai/inference/v1", "model": "accounts/fireworks/models/gemma-7b-it"},
        "3": {"name": "Groq", "url": "https://api.groq.com/openai/v1", "model": "gemma-7b-it"},
        "4": {"name": "Custom", "url": "", "model": ""}
    }
    
    for key, provider in providers.items():
        print(f"{key}. {provider['name']}")
    
    choice = input("\nChoose provider (1-4): ").strip()
    
    if choice in providers:
        provider = providers[choice]
        
        if choice == "4":
            base_url = input("Enter API base URL: ").strip()
            model_name = input("Enter model name: ").strip()
        else:
            base_url = provider["url"]
            model_name = provider["model"]
        
        api_key = input(f"\n🔑 Enter your {provider['name']} API key: ").strip()
        
        if api_key and base_url and model_name:
            env_content = f"""# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# OpenAI-Compatible Provider Configuration
OPENAI_API_KEY={api_key}
OPENAI_BASE_URL={base_url}
GEMMA_MODEL={model_name}

# Agent Configuration
LOG_LEVEL=INFO
EVOLUTION_THRESHOLD=0.6
PERFORMANCE_WINDOW=10

# Text-to-Cypher Configuration
CYPHER_CONFIDENCE_THRESHOLD=0.6
CYPHER_MAX_TOKENS=2048
CYPHER_TEMPERATURE=0.2
"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print(f"✅ {provider['name']} configuration saved!")
            return True
    
    print("❌ Configuration incomplete")
    return False

def test_configuration():
    """Test the configuration"""
    print("\n🧪 TESTING CONFIGURATION")
    print("=" * 25)
    
    try:
        from gemma_text_to_cypher import gemma_text_to_cypher, CypherRequest
        import asyncio
        
        async def test():
            request = CypherRequest(
                question="Show me all agents in the system",
                context="Simple test query"
            )
            
            response = await gemma_text_to_cypher.generate_cypher(request)
            
            print(f"✅ Test successful!")
            print(f"Generated query: {response.cypher_query[:100]}...")
            print(f"Confidence: {response.confidence_score:.2%}")
            return True
        
        return asyncio.run(test())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 This might be because dependencies aren't installed yet.")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function"""
    print("☁️ CLOUD GEMMA SETUP (No Local Installation)")
    print("=" * 50)
    
    print("\n🎯 Available Cloud Options:")
    print("1. 🌐 Google AI Studio (Recommended - Free & Easy)")
    print("2. 🤗 Hugging Face (Good for Development)")
    print("3. ⚡ OpenAI-Compatible Providers (Various Options)")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    success = False
    
    if choice == "1":
        success = setup_google_ai_studio()
    elif choice == "2":
        success = setup_huggingface()
    elif choice == "3":
        success = setup_openai_compatible()
    else:
        print("❌ Invalid choice")
        return
    
    if success:
        print(f"\n🎉 SETUP COMPLETE!")
        print(f"✅ Configuration saved to .env file")
        print(f"✅ No local installation required")
        
        print(f"\n📋 Next Steps:")
        print(f"1. Install dependencies: pip install -r requirements.txt")
        print(f"2. Test the system: python enhanced_demo.py")
        print(f"3. Start using natural language queries!")
        
        test_now = input(f"\n🧪 Test configuration now? (y/n): ").strip().lower()
        if test_now == 'y':
            test_configuration()
    
    else:
        print(f"\n❌ Setup incomplete. Please try again.")

if __name__ == "__main__":
    main()
