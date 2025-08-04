# AI/ML Weekly LinkedIn Post Generator 🚀

Automated GitHub Actions workflow that generates weekly LinkedIn posts summarizing the latest AI/ML developments, research papers, and industry news.

## 🎯 Features

- **Automated Scheduling**: Runs every Monday at 08:00 UTC
- **Multi-Source Data**: Fetches content from arXiv papers, AI news, and GitHub trending repos
- **AI-Powered Curation**: Uses OpenAI GPT to select and summarize the most important developments
- **Professional Format**: Generates LinkedIn-ready posts with proper formatting, hashtags, and links
- **Version Control**: Saves all generated posts to the repository with timestamps

## 📁 Project Structure

```
workflows/
├── .github/workflows/
│   └── weekly-ai-ml-post.yml    # GitHub Actions workflow
├── scripts/
│   └── generate_weekly_post.py  # Main post generation script  
├── posts/                       # Generated posts storage
├── requirements.txt             # Python dependencies
├── config.py                   # Configuration settings
├── setup.py                    # Local testing setup
└── README.md                   # This file
```

## 🛠️ Setup Instructions

### 1. Clone and Setup Repository

```bash
git clone https://github.com/kaansen97/workflows.git
cd workflows
pip install -r requirements.txt
```

### 2. Configure API Keys

You'll need the following API keys for full functionality:

#### Required:
- **Google Gemini API Key**: For AI-powered content analysis and post generation
  - Get it from: https://makersuite.google.com/app/apikey
- **SerpAPI Key**: For searching recent AI/ML news
  - Get it from: https://serpapi.com/

#### Optional:
- **GitHub Token**: For higher API rate limits when fetching trending repositories
  - Generate at: https://github.com/settings/tokens

### 3. Set GitHub Repository Secrets

In your GitHub repository, go to Settings > Secrets and Variables > Actions, and add:

- `GEMINI_API_KEY`: Your Google Gemini API key
- `SERPAPI_KEY`: Your SerpAPI key  
- `GITHUB_TOKEN`: Your GitHub personal access token (optional)

### 4. Test Locally (Optional)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SERPAPI_KEY=your_serpapi_key_here
GITHUB_TOKEN=your_github_token_here
```

Run the setup script:

```bash
python setup.py
```

## 🔄 How It Works

### Data Sources

1. **arXiv Papers**: Fetches recent AI/ML research papers from categories:
   - cs.AI (Artificial Intelligence)  
   - cs.LG (Machine Learning)
   - cs.CV (Computer Vision)
   - cs.CL (Computational Linguistics)
   - cs.NE (Neural and Evolutionary Computing)

2. **AI News**: Searches for recent AI/ML news articles and announcements

3. **GitHub Trending**: Finds popular AI/ML repositories updated in the past week

### AI-Powered Curation

The script uses Google's Gemini Pro model to:
- Analyze all collected developments
- Select the 3-5 most significant and impactful ones
- Generate engaging LinkedIn post content with professional commentary
- Ensure diverse coverage across different AI/ML domains

### Post Generation

Each generated post includes:
- Engaging introduction
- 3-5 selected developments with:
  - Clear title
  - One-sentence expert commentary  
  - Direct URL link
- Relevant hashtags and call-to-action
- Professional formatting optimized for LinkedIn

## 📅 Scheduling

The workflow runs automatically:
- **Schedule**: Every Monday at 08:00 UTC
- **Manual Trigger**: Can be triggered manually from GitHub Actions tab
- **Output**: Generated posts are saved in the `posts/` directory

## 🎨 Customization

### Modify Sources

Edit `config.py` to:
- Enable/disable specific data sources
- Adjust the number of results fetched
- Modify search categories and keywords

### Customize Post Format

Edit the `generate_linkedin_post()` method in `generate_weekly_post.py` to:
- Change post structure and tone
- Modify hashtags and formatting
- Adjust content length and style

### Change Schedule

Modify the cron expression in `.github/workflows/weekly-ai-ml-post.yml`:
```yaml
schedule:
  - cron: '0 8 * * 1'  # Every Monday at 08:00 UTC
```

## 📝 Generated Post Example

```
🚀 AI/ML Weekly Roundup - January 15, 2025

This week's most significant developments in AI and Machine Learning:

🧠 Attention Is All You Need: Transformer Advances in Multi-Modal Learning
   Revolutionary approach combining vision and language understanding in a single architecture.
   🔗 https://arxiv.org/abs/2501.xxxxx

⚡ OpenAI Announces GPT-5 Preview with Enhanced Reasoning Capabilities  
   Major leap forward in AI reasoning and complex problem-solving abilities.
   🔗 https://openai.com/blog/gpt-5-preview

🔬 Microsoft Open-Sources Florence-2: Unified Vision-Language Model
   Democratizing access to state-of-the-art multi-modal AI capabilities.
   🔗 https://github.com/microsoft/Florence-2

💡 Google DeepMind's AlphaFold 3 Predicts Protein-Drug Interactions
   Breakthrough in computational biology with massive healthcare implications.
   🔗 https://deepmind.google/research/alphafold-3

What development interests you most? Share your thoughts below! 👇

#AI #MachineLearning #DeepLearning #ArtificialIntelligence #Tech #Innovation #Research #OpenSource
```

## 🚨 Troubleshooting

### Common Issues

1. **No posts generated**: Check API keys and internet connectivity
2. **Empty developments**: Verify data sources are accessible
3. **Workflow fails**: Check GitHub Actions logs and repository secrets

### Fallback Mode

If API keys are missing, the script runs in fallback mode:
- Uses simple heuristics for content selection
- Generates basic post format without AI enhancement
- Still collects and presents developments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [arXiv](https://arxiv.org/) for providing access to research papers
- [Google AI](https://ai.google.dev/) for Gemini Pro models
- [SerpAPI](https://serpapi.com/) for news search capabilities
- [GitHub](https://github.com/) for hosting and automation infrastructure

---

**Made with ❤️ for the AI/ML community**