# Configuration for AI/ML Weekly Post Generator

# Data Sources Configuration
SOURCES = {
    "arxiv": {
        "enabled": True,
        "categories": ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE"],
        "max_results": 50
    },
    "news": {
        "enabled": True,
        "max_results": 15
    },
    "github": {
        "enabled": True,
        "max_results": 10
    }
}

# LinkedIn Post Configuration
POST_CONFIG = {
    "min_developments": 3,
    "max_developments": 5,
    "max_post_length": 3000,
    "include_hashtags": True,
    "hashtags": [
        "#AI", "#MachineLearning", "#DeepLearning", 
        "#ArtificialIntelligence", "#Tech", "#Innovation", 
        "#Research", "#OpenSource"
    ]
}

# API Keys (set as environment variables)
# GEMINI_API_KEY=your_gemini_api_key_here
# SERPAPI_KEY=your_serpapi_key_here
# GITHUB_TOKEN=your_github_token_here
