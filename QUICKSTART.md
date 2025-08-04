# Quick Start Guide 🚀

## 5-Minute Setup

### 1. Get API Keys (2 minutes)

**Google Gemini API Key** (Required):
1. Go to https://makersuite.google.com/app/apikey
2. Create new API key
3. Copy the key

**SerpAPI Key** (Required):
1. Go to https://serpapi.com/
2. Sign up for free account (100 searches/month free)
3. Get your API key from dashboard

### 2. Configure GitHub Secrets (1 minute)

1. Go to your GitHub repository
2. Settings → Secrets and Variables → Actions
3. Add these secrets:
   - `GEMINI_API_KEY`: Your Gemini key
   - `SERPAPI_KEY`: Your SerpAPI key

**Important**: Make sure GitHub Actions has write permissions:
- Go to Settings → Actions → General
- Under "Workflow permissions", select "Read and write permissions"
- Click "Save"

### 3. Test the Setup (2 minutes)

**Option A: Manual Trigger**
1. Go to Actions tab in GitHub
2. Select "Weekly AI/ML LinkedIn Post Generator"
3. Click "Run workflow"

**Option B: Local Test**
```bash
# Clone and setup
git clone https://github.com/kaansen97/workflows.git
cd workflows
pip install -r requirements.txt

# Create .env file with your keys
cp .env.template .env
# Edit .env with your actual API keys

# Run test
python test.py
```

## What Happens Next?

- ✅ Workflow runs every Monday at 08:00 UTC
- ✅ Generates LinkedIn post with 3-5 AI/ML developments
- ✅ Saves post to `posts/` directory
- ✅ You get professional content ready to post

## Sample Output

```
🚀 AI/ML Weekly Roundup - January 15, 2025

This week's most significant developments:

🧠 GPT-5 Preview Released with Enhanced Reasoning
   Major breakthrough in AI reasoning capabilities...
   🔗 https://openai.com/blog/gpt-5

⚡ New Transformer Architecture Achieves SOTA on Vision Tasks
   Revolutionary approach to computer vision...
   🔗 https://arxiv.org/abs/2501.xxxxx

#AI #MachineLearning #DeepLearning #Tech
```

## Need Help?

- 📖 Read the full [README.md](README.md)
- 🧪 Run `python test.py` to diagnose issues
- 🐛 Check GitHub Actions logs for errors
- 💬 Open an issue for support