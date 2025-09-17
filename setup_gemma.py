#!/usr/bin/env python3
"""
Gemma Setup Script

This script helps you set up Gemma3-4B for Text-to-Cypher translation.
It provides multiple options depending on your preferences.
"""

import os
import subprocess
import sys
import requests
from typing import Optional

def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama():
    """Guide user to install Ollama"""
    print("\n🤖 OLLAMA INSTALLATION REQUIRED")
    print("=" * 40)
    print("Ollama is the easiest way to run Gemma locally (no API key needed!)")
    print("\nInstallation options:")
    print("1. Windows: Download from https://ollama.ai/download")
    print("2. macOS: brew install ollama")
    print("3. Linux: curl -fsSL https://ollama.ai/install.sh | sh")
    print("\nAfter installation, restart this script.")

def check_ollama_running() -> bool:
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama_service():
    """Start Ollama service"""
    print("\n🚀 Starting Ollama service...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['ollama', 'serve'], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Unix-like
            subprocess.Popen(['ollama', 'serve'])
        
        import time
        time.sleep(3)
        
        if check_ollama_running():
            print("✅ Ollama service started successfully!")
            return True
        else:
            print("⚠️ Ollama service may still be starting...")
            return False
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def download_gemma_model(model_name: str = "gemma2:9b") -> bool:
    """Download Gemma model through Ollama"""
    print(f"\n📥 Downloading Gemma model: {model_name}")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download model: {e.stderr}")
        return False

def test_gemma_model(model_name: str = "gemma2:9b") -> bool:
    """Test if Gemma model is working"""
    print(f"\n🧪 Testing Gemma model: {model_name}")
    
    try:
        import ollama
        
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "Generate a simple Cypher query to find all nodes with label 'Agent'"
            }]
        )
        
        result = response['message']['content']
        print(f"✅ Model test successful!")
        print(f"Sample response: {result[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def setup_environment_file():
    """Setup environment configuration"""
    print("\n📝 Setting up environment configuration...")
    
    env_content = """# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Gemma Configuration (Local Ollama - No API Key Needed!)
GEMMA_MODEL=gemma2:9b

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
    
    print("✅ Created .env file with Gemma configuration")

def setup_alternative_providers():
    """Show setup for alternative providers"""
    print("\n🔄 ALTERNATIVE GEMMA PROVIDERS")
    print("=" * 40)
    
    print("\n1. 🌐 Google AI Studio (Official)")
    print("   - Get API key: https://makersuite.google.com/app/apikey")
    print("   - Set GOOGLE_API_KEY in .env")
    print("   - Set GEMMA_MODEL=gemini-pro")
    
    print("\n2. 🤗 Hugging Face")
    print("   - Get API key: https://huggingface.co/settings/tokens")
    print("   - Set HUGGINGFACE_API_KEY in .env")
    print("   - Set GEMMA_MODEL=google/gemma-2-9b-it")
    
    print("\n3. ☁️ Other Cloud Providers")
    print("   - Various providers offer Gemma through OpenAI-compatible APIs")
    print("   - Set OPENAI_API_KEY and OPENAI_BASE_URL in .env")

def main():
    """Main setup function"""
    print("🧠 GEMMA3-4B SETUP FOR TEXT-TO-CYPHER")
    print("=" * 50)
    
    print("\n🎯 Setup Options:")
    print("1. Local Ollama (Recommended - Free, Private, No API Key)")
    print("2. Google AI Studio (Requires API Key)")
    print("3. Other Providers (Various API Keys)")
    
    choice = input("\nChoose setup option (1-3): ").strip()
    
    if choice == "1":
        # Local Ollama setup
        print("\n🏠 SETTING UP LOCAL OLLAMA")
        print("=" * 30)
        
        if not check_ollama_installed():
            print("❌ Ollama not found")
            install_ollama()
            return
        
        if not check_ollama_running():
            print("⚠️ Ollama service not running")
            if not start_ollama_service():
                print("Please start Ollama manually: 'ollama serve'")
                return
        
        # Download model
        model_choice = input("\nChoose Gemma model (2b/9b/27b) [9b]: ").strip() or "9b"
        model_name = f"gemma2:{model_choice}"
        
        if download_gemma_model(model_name):
            if test_gemma_model(model_name):
                setup_environment_file()
                print(f"\n🎉 SUCCESS! Gemma setup complete!")
                print(f"   Model: {model_name}")
                print(f"   No API key required!")
                print(f"   Ready to use with your AI system!")
            else:
                print("❌ Model test failed. Please check Ollama installation.")
        else:
            print("❌ Model download failed. Please check your internet connection.")
    
    elif choice == "2":
        print("\n🌐 GOOGLE AI STUDIO SETUP")
        print("=" * 25)
        setup_alternative_providers()
        
        api_key = input("\nEnter your Google AI API key (or press Enter to skip): ").strip()
        if api_key:
            env_content = f"""# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Google AI Studio Configuration
GEMMA_MODEL=gemini-pro
GOOGLE_API_KEY={api_key}

# Agent Configuration
LOG_LEVEL=INFO
EVOLUTION_THRESHOLD=0.6
PERFORMANCE_WINDOW=10
"""
            with open('.env', 'w') as f:
                f.write(env_content)
            print("✅ Google AI Studio configuration saved!")
    
    elif choice == "3":
        print("\n☁️ OTHER PROVIDERS SETUP")
        print("=" * 23)
        setup_alternative_providers()
        print("\n💡 Tip: Local Ollama is usually the best option for privacy and cost!")
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
