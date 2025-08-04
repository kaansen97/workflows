#!/usr/bin/env python3
"""
Local setup and testing script for AI/ML Weekly Post Generator
"""

import os
import sys
from dotenv import load_dotenv

def setup_environment():
    """Setup local environment for testing"""
    
    # Load environment variables from .env file if it exists
    if os.path.exists('.env'):
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    else:
        print("⚠️  No .env file found. Please create one with your API keys.")
        print("Example .env file content:")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        print("SERPAPI_KEY=your_serpapi_key_here")
        print("GITHUB_TOKEN=your_github_token_here")
        
    # Check for required dependencies
    required_packages = [
        'requests', 'openai', 'feedparser', 'beautifulsoup4', 
        'arxiv', 'google-search-results', 'pytz'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
    
    # Check API keys
    api_keys = {
        'GEMINI_API_KEY': 'Google Gemini API (for AI analysis and post generation)',
        'SERPAPI_KEY': 'SerpAPI (for news search)',
        'GITHUB_TOKEN': 'GitHub Token (optional, for higher rate limits)'
    }
    
    for key, description in api_keys.items():
        if os.getenv(key):
            print(f"✅ {key} is set")
        else:
            if key == 'GITHUB_TOKEN':
                print(f"⚠️  {key} not set - {description}")
            else:
                print(f"❌ {key} not set - {description}")
    
    return True

def test_generation():
    """Test the post generation locally"""
    print("\n🧪 Testing post generation...")
    
    try:
        # Import and run the generator
        sys.path.append('scripts')
        from generate_weekly_post import AIMLPostGenerator
        
        generator = AIMLPostGenerator()
        generator.generate_weekly_post()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 AI/ML Weekly Post Generator - Local Setup")
    print("=" * 50)
    
    if setup_environment():
        choice = input("\n🧪 Would you like to test the post generation? (y/n): ").lower().strip()
        if choice == 'y':
            test_generation()
    
    print("\n📝 Next steps:")
    print("1. Set up your API keys in GitHub repository secrets:")
    print("   - GEMINI_API_KEY")
    print("   - SERPAPI_KEY")
    print("   - GITHUB_TOKEN (optional)")
    print("2. Push your code to GitHub")
    print("3. The workflow will run every Monday at 08:00 UTC")
    print("4. Check the 'posts' folder for generated content")

if __name__ == "__main__":
    main()
