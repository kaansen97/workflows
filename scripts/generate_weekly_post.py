#!/usr/bin/env python3
"""
Weekly AI/ML LinkedIn Post Generator

This script fetches recent AI/ML developments from various sources and generates
a LinkedIn post summarizing 3-5 major developments from the past week.
"""

import os
import json
import requests
import feedparser
import arxiv
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytz
import google.generativeai as genai
from serpapi import GoogleSearch

class AIMLPostGenerator:
    def __init__(self):
        # Configure Gemini API
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
            
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.developments = []
        
    def fetch_arxiv_papers(self, days_back=7) -> List[Dict[str, Any]]:
        """Fetch recent AI/ML papers from arXiv"""
        print("Fetching papers from arXiv...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        papers = []
        try:
            # Use individual category searches to avoid URL encoding issues
            categories = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE"]
            
            for category in categories:
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=20,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for result in search.results():
                    if result.published.replace(tzinfo=None) >= start_date:
                        papers.append({
                            'title': result.title.strip(),
                            'summary': result.summary.strip()[:300] + "...",
                            'url': result.entry_id,
                            'published': result.published.strftime('%Y-%m-%d'),
                            'authors': [author.name for author in result.authors[:3]],
                            'source': 'arXiv',
                            'type': 'paper',
                            'category': category
                        })
                        
                if len(papers) >= 15:  # Limit total papers
                    break
                    
        except Exception as e:
            print(f"Error fetching arXiv papers: {e}")
            
        # Remove duplicates based on title
        seen_titles = set()
        unique_papers = []
        for paper in papers:
            if paper['title'] not in seen_titles:
                seen_titles.add(paper['title'])
                unique_papers.append(paper)
                
        return unique_papers[:10]  # Limit to top 10
    
    def fetch_ai_news(self) -> List[Dict[str, Any]]:
        """Fetch AI/ML news from various sources using web search"""
        print("Fetching AI/ML news...")
        
        news_items = []
        
        if not self.serpapi_key:
            print("No SERPAPI key found, skipping news search")
            return news_items
            
        try:
            # More specific search queries for individual news items
            search_queries = [
                '"OpenAI" OR "Anthropic" OR "Google AI" announcement last week',
                '"new AI model" OR "AI breakthrough" OR "AI research" 2025',
                '"ChatGPT" OR "Claude" OR "Gemini" update news',
                '"Meta AI" OR "Microsoft AI" OR "Apple AI" announcement',
                '"AI startup" funding OR acquisition news',
                '"artificial intelligence" regulation OR policy news'
            ]
            
            for query in search_queries:
                search = GoogleSearch({
                    "q": query,
                    "tbm": "nws",  # News search
                    "tbs": "qdr:w",  # Past week
                    "num": 10,
                    "api_key": self.serpapi_key
                })
                
                results = search.get_dict()
                
                if "news_results" in results:
                    for item in results["news_results"]:
                        title = item.get('title', '')
                        link = item.get('link', '')
                        
                        # Filter out generic news aggregation pages
                        if any(skip_word in title.lower() for skip_word in [
                            'weekly update', 'news roundup', 'this week in', 
                            'daily digest', 'news summary', 'weekly digest',
                            'newsletter', 'roundup', 'weekly wrap'
                        ]):
                            continue
                            
                        # Filter out certain domains that are typically aggregators
                        if any(domain in link for domain in [
                            'marketingprofs.com/opinions',
                            'solutionsreview.com/artificial-intelligence-news-for-the-week',
                            'martech.org/the-latest-ai-powered-martech-news-and-releases'
                        ]):
                            continue
                        
                        news_items.append({
                            'title': title,
                            'summary': item.get('snippet', ''),
                            'url': link,
                            'published': item.get('date', ''),
                            'source': item.get('source', ''),
                            'type': 'news'
                        })
                        
        except Exception as e:
            print(f"Error fetching news: {e}")
            
        # Remove duplicates based on title similarity
        unique_news = []
        seen_titles = set()
        
        for item in news_items:
            title_lower = item['title'].lower()
            # Simple deduplication based on title
            if not any(seen_title in title_lower or title_lower in seen_title 
                      for seen_title in seen_titles):
                seen_titles.add(title_lower)
                unique_news.append(item)
                
        return unique_news[:12]  # Limit results
    
    def fetch_tech_news_feeds(self) -> List[Dict[str, Any]]:
        """Fetch AI/ML news from RSS feeds of tech publications"""
        print("Fetching news from RSS feeds...")
        
        news_items = []
        
        # Tech news RSS feeds that often cover AI/ML
        rss_feeds = [
            {
                'url': 'https://feeds.feedburner.com/venturebeat/SZYF',
                'source': 'VentureBeat AI'
            },
            {
                'url': 'https://techcrunch.com/category/artificial-intelligence/feed/',
                'source': 'TechCrunch AI'
            },
            {
                'url': 'https://www.theverge.com/ai-artificial-intelligence/rss/index.xml',
                'source': 'The Verge AI'
            },
            {
                'url': 'https://feeds.feedburner.com/oreilly/radar',
                'source': "O'Reilly Radar"
            }
        ]
        
        for feed_info in rss_feeds:
            try:
                feed = feedparser.parse(feed_info['url'])
                
                for entry in feed.entries[:5]:  # Limit per feed
                    # Check if published in the last week
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_date = datetime.now() - timedelta(days=1)  # Assume recent
                            
                        if pub_date >= datetime.now() - timedelta(days=7):
                            # Filter for AI/ML related content
                            title_lower = entry.title.lower()
                            summary_lower = getattr(entry, 'summary', '').lower()
                            
                            ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 
                                         'deep learning', 'neural network', 'chatgpt', 'openai',
                                         'anthropic', 'claude', 'gemini', 'llm', 'generative']
                            
                            if any(keyword in title_lower or keyword in summary_lower 
                                  for keyword in ai_keywords):
                                news_items.append({
                                    'title': entry.title,
                                    'summary': getattr(entry, 'summary', '')[:300] + "...",
                                    'url': entry.link,
                                    'published': pub_date.strftime('%Y-%m-%d'),
                                    'source': feed_info['source'],
                                    'type': 'news'
                                })
                    except Exception as e:
                        print(f"Error parsing entry from {feed_info['source']}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error fetching RSS feed {feed_info['source']}: {e}")
                continue
        
        return news_items
    
    def fetch_github_trending(self) -> List[Dict[str, Any]]:
        """Fetch trending AI/ML repositories from GitHub"""
        print("Fetching trending GitHub repositories...")
        
        repos = []
        try:
            # GitHub API to get trending AI/ML repositories
            url = "https://api.github.com/search/repositories"
            params = {
                'q': 'machine-learning OR artificial-intelligence OR deep-learning language:Python',
                'sort': 'updated',
                'order': 'desc',
                'per_page': 15
            }
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AI-ML-Post-Generator'
            }
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                if 'items' in data and data['items']:
                    for repo in data['items']:
                        # Check if updated in the last week
                        updated_at = datetime.strptime(repo['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
                        if updated_at >= datetime.now() - timedelta(days=7):
                            repos.append({
                                'title': f"{repo['full_name']} - {repo['description'][:100] if repo['description'] else 'No description'}",
                                'summary': repo['description'] or "No description available",
                                'url': repo['html_url'],
                                'published': repo['updated_at'][:10],
                                'source': 'GitHub',
                                'type': 'repository',
                                'stars': repo['stargazers_count'],
                                'language': repo.get('language', 'Unknown')
                            })
                else:
                    print("No repository items found in GitHub response")
            else:
                print(f"GitHub API error: {response.status_code} - {response.text}")
                        
        except Exception as e:
            print(f"Error fetching GitHub repos: {e}")
            
        return repos[:8]  # Limit to top 8
    
    def analyze_and_select_top_developments(self, all_developments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use AI to analyze and select the top 3-5 developments"""
        print("Analyzing developments with AI...")
        
        if not self.gemini_model:
            print("No Gemini API key found, using simple selection")
            # Simple fallback: select based on recency and source diversity
            selected = []
            sources_used = set()
            
            # Sort by date and select diverse sources
            sorted_devs = sorted(all_developments, 
                               key=lambda x: x.get('published', '2025-01-01'), 
                               reverse=True)
            
            for dev in sorted_devs:
                if len(selected) >= 5:
                    break
                source = dev.get('source', 'unknown')
                if source not in sources_used or len(selected) < 3:
                    selected.append(dev)
                    sources_used.add(source)
                    
            return selected[:5]
        
        try:
            # Prepare context for AI analysis
            context = "Recent AI/ML developments:\n\n"
            for i, dev in enumerate(all_developments[:20], 1):  # Limit to avoid token limits
                context += f"{i}. {dev['title']}\n   Source: {dev['source']}\n   Summary: {dev['summary'][:200]}...\n\n"
            
            prompt = f"""
            You are an AI/ML expert curator. From the following developments, select the 3-5 most significant and impactful ones for a LinkedIn post targeting AI/ML professionals.
            
            Consider:
            - Impact on the field
            - Novelty and innovation
            - Practical applications
            - Industry relevance
            - Diversity of topics
            - Avoid generic news aggregation sources

            {context}

            Return only the numbers (1-based) of the selected developments, separated by commas.
            Example: 1, 3, 7, 12
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse the response to get selected indices
            selected_indices = []
            try:
                indices_str = response.text.strip()
                selected_indices = [int(x.strip()) - 1 for x in indices_str.split(',') if x.strip().isdigit()]
            except:
                # Fallback to first 4 if parsing fails
                selected_indices = [0, 1, 2, 3]
            
            # Return selected developments
            return [all_developments[i] for i in selected_indices if i < len(all_developments)]
            
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            # Fallback selection
            return all_developments[:4]
    
    def generate_linkedin_post(self, selected_developments: List[Dict[str, Any]]) -> str:
        """Generate the final LinkedIn post"""
        print("Generating LinkedIn post...")
        
        # Get current date for the post
        current_date = datetime.now().strftime('%B %d, %Y')
        
        if not self.gemini_model:
            # Fallback manual post generation
            post = f"🚀 AI/ML Weekly Update - {current_date}\n\n"
            post += "Here are the top AI/ML developments from this week:\n\n"
            
            for i, dev in enumerate(selected_developments, 1):
                post += f"{i}. {dev['title']}\n"
                post += f"   💡 {dev['summary'][:150]}...\n"
                post += f"   🔗 {dev['url']}\n\n"
            
            post += "#AI #MachineLearning #DeepLearning #ArtificialIntelligence #Tech #Innovation"
            return post
        
        try:
            # Prepare context for AI post generation
            developments_text = ""
            for i, dev in enumerate(selected_developments, 1):
                developments_text += f"{i}. Title: {dev['title']}\n"
                developments_text += f"   Summary: {dev['summary']}\n"
                developments_text += f"   URL: {dev['url']}\n"
                developments_text += f"   Source: {dev['source']}\n\n"
            
            prompt = f"""
            Create an engaging LinkedIn post for AI/ML professionals summarizing these weekly developments.

            Requirements:
            - Start with an engaging hook
            - Include exactly {len(selected_developments)} developments
            - For each development: title, one-sentence insightful commentary, and URL
            - Use relevant emojis
            - End with appropriate hashtags
            - Keep it professional but engaging
            - Total length should be suitable for LinkedIn (under 3000 characters)

            Developments:
            {developments_text}

            Current date: {current_date}
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating LinkedIn post: {e}")
            # Fallback to manual generation
            return self.generate_linkedin_post_fallback(selected_developments, current_date)
    
    def generate_linkedin_post_fallback(self, selected_developments: List[Dict[str, Any]], current_date: str) -> str:
        """Fallback method for generating LinkedIn post without AI"""
        post = f"🚀 AI/ML Weekly Roundup - {current_date}\n\n"
        post += "This week's most significant developments in AI and Machine Learning:\n\n"
        
        emojis = ["🧠", "⚡", "🔬", "💡", "🚀"]
        
        for i, dev in enumerate(selected_developments):
            emoji = emojis[i % len(emojis)]
            post += f"{emoji} {dev['title']}\n"
            
            # Generate simple commentary based on source type
            if dev['type'] == 'paper':
                commentary = "Advancing the theoretical foundations of AI research."
            elif dev['type'] == 'news':
                commentary = "Another step forward in practical AI applications."
            elif dev['type'] == 'repository':
                commentary = "Open-source innovation driving AI accessibility."
            else:
                commentary = "Significant development in the AI landscape."
            
            post += f"   {commentary}\n"
            post += f"   🔗 {dev['url']}\n\n"
        
        post += "What development interests you most? Share your thoughts below! 👇\n\n"
        post += "#AI #MachineLearning #DeepLearning #ArtificialIntelligence #Tech #Innovation #Research #OpenSource"
        
        return post
    
    def save_post(self, post_content: str):
        """Save the generated post to a file"""
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f"posts/ai-ml-weekly-{timestamp}.md"
        
        os.makedirs('posts', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# AI/ML Weekly Post - {timestamp}\n\n")
            f.write("Generated by AI/ML LinkedIn Post Generator\n\n")
            f.write("## LinkedIn Post Content\n\n")
            f.write(post_content)
            f.write("\n\n---\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n")
        
        print(f"Post saved to: {filename}")
        return filename
    
    def generate_weekly_post(self):
        """Main method to generate the weekly LinkedIn post"""
        print("🚀 Starting AI/ML Weekly Post Generation...")
        
        # Fetch developments from various sources
        print("\n📊 Fetching data from sources...")
        arxiv_papers = self.fetch_arxiv_papers()
        ai_news = self.fetch_ai_news()
        rss_news = self.fetch_tech_news_feeds()
        github_repos = self.fetch_github_trending()
        
        # Combine all developments
        all_developments = arxiv_papers + ai_news + rss_news + github_repos
        print(f"Found {len(all_developments)} total developments")
        print(f"  - ArXiv papers: {len(arxiv_papers)}")
        print(f"  - News (SerpAPI): {len(ai_news)}")
        print(f"  - News (RSS): {len(rss_news)}")
        print(f"  - GitHub repos: {len(github_repos)}")
        
        if not all_developments:
            print("❌ No developments found. Please check your API keys and internet connection.")
            return
        
        # Select top developments
        selected_developments = self.analyze_and_select_top_developments(all_developments)
        print(f"📝 Selected {len(selected_developments)} top developments")
        
        # Generate LinkedIn post
        linkedin_post = self.generate_linkedin_post(selected_developments)
        
        # Save the post
        filename = self.save_post(linkedin_post)
        
        print(f"\n✅ Weekly post generated successfully!")
        print(f"📁 Saved to: {filename}")
        print("\n📋 Post preview:")
        print("-" * 50)
        print(linkedin_post)
        print("-" * 50)

def main():
    """Main entry point"""
    generator = AIMLPostGenerator()
    generator.generate_weekly_post()

if __name__ == "__main__":
    main()
