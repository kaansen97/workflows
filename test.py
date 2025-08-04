#!/usr/bin/env python3
"""
Test script for AI/ML Weekly Post Generator
"""

import os
import sys
import json
from datetime import datetime

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    packages = {
        'requests': 'requests',
        'google-generativeai': 'google.generativeai', 
        'feedparser': 'feedparser',
        'beautifulsoup4': 'bs4',
        'arxiv': 'arxiv',
        'google-search-results': 'serpapi',
        'pytz': 'pytz'
    }
    
    failed_imports = []
    
    for package, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully")
    return True

def test_api_connections():
    """Test API connections"""
    print("\n🌐 Testing API connections...")
    
    # Test arXiv (no API key needed)
    try:
        import arxiv
        search = arxiv.Search(query="machine learning", max_results=1)
        next(search.results())
        print("  ✅ arXiv API connection")
    except Exception as e:
        print(f"  ❌ arXiv API connection: {e}")
    
    # Test Gemini (if API key is available)
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-pro')
            # Simple test call
            response = model.generate_content("Test")
            print("  ✅ Gemini API connection")
        except Exception as e:
            print(f"  ❌ Gemini API connection: {e}")
    else:
        print("  ⚠️  Gemini API key not found")
    
    # Test SerpAPI (if API key is available)
    serpapi_key = os.getenv('SERPAPI_KEY')
    if serpapi_key:
        try:
            from serpapi import GoogleSearch
            search = GoogleSearch({
                "q": "test",
                "api_key": serpapi_key
            })
            results = search.get_dict()
            print("  ✅ SerpAPI connection")
        except Exception as e:
            print(f"  ❌ SerpAPI connection: {e}")
    else:
        print("  ⚠️  SerpAPI key not found")

def test_post_generation():
    """Test basic post generation functionality"""
    print("\n📝 Testing post generation...")
    
    try:
        # Add scripts directory to path
        sys.path.append('scripts')
        from generate_weekly_post import AIMLPostGenerator
        
        # Create generator instance
        generator = AIMLPostGenerator()
        
        # Test individual components
        print("  Testing arXiv papers fetch...")
        papers = generator.fetch_arxiv_papers(days_back=1)  # Short period for testing
        print(f"    Found {len(papers)} papers")
        
        print("  Testing GitHub trending fetch...")
        repos = generator.fetch_github_trending()
        print(f"    Found {len(repos)} repositories")
        
        # Create a simple test post
        test_developments = [
            {
                'title': 'Test AI Development',
                'summary': 'This is a test development for validation purposes.',
                'url': 'https://example.com',
                'source': 'Test',
                'type': 'test',
                'published': datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        print("  Testing post generation...")
        post = generator.generate_linkedin_post(test_developments)
        
        if post and len(post) > 50:
            print("  ✅ Post generation successful")
            print(f"  Generated post length: {len(post)} characters")
        else:
            print("  ❌ Post generation failed or too short")
            
        print(f"  ✅ Core functionality test completed")
        
    except Exception as e:
        print(f"  ❌ Post generation test failed: {e}")
        import traceback
        traceback.print_exc()

def create_sample_env():
    """Create a sample .env file if it doesn't exist"""
    if not os.path.exists('.env') and os.path.exists('.env.template'):
        print("\n📁 Creating sample .env file...")
        import shutil
        shutil.copy('.env.template', '.env')
        print("  ✅ Created .env file from template")
        print("  📝 Please edit .env file with your actual API keys")

def main():
    """Main test function"""
    print("🚀 AI/ML Weekly Post Generator - Test Suite")
    print("=" * 60)
    
    # Load environment variables if .env exists
    try:
        from dotenv import load_dotenv
        if os.path.exists('.env'):
            load_dotenv()
            print("✅ Loaded environment variables from .env")
        else:
            print("⚠️  No .env file found")
    except ImportError:
        print("⚠️  python-dotenv not installed (optional)")
    
    # Run tests
    success = True
    
    success &= test_imports()
    test_api_connections()
    
    if success:
        test_post_generation()
    
    create_sample_env()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test suite completed successfully!")
        print("\n📝 Next steps:")
        print("1. Set up API keys in .env file or GitHub secrets")
        print("2. Push to GitHub to enable automated workflow")
        print("3. Test manual workflow trigger from GitHub Actions")
    else:
        print("❌ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()
