# app.py
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any
import requests
import time
import hashlib
import os
from dateutil import parser as dateparser
from dotenv import load_dotenv
import google.generativeai as genai
import tweepy
import openai

# Load environment variables
load_dotenv()

# Optional: improved summarisation with transformers
USE_TRANSFORMERS = False
try:
    from transformers import pipeline
    summarizer_pipeline = pipeline("summarization")
    USE_TRANSFORMERS = True
except Exception:
    USE_TRANSFORMERS = False

# Optional: newspaper for fetching full article text
try:
    from newspaper import Article
    HAVE_NEWSPAPER = True
except Exception:
    HAVE_NEWSPAPER = False

# -------------------------
# Agent implementations
# -------------------------
@dataclass
class ArticleItem:
    id: str
    source: str
    title: str
    description: str
    url: str
    published_at: str
    content: str = ""
    summary: str = ""
    category: str = ""
    impact: str = ""
    impact_score: float = 0.0
    why: str = ""

class FetcherAgent:
    """
    Fetches latest news from NewsAPI if key provided, otherwise returns sample/mock items.
    """
    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')

    def fetch(self, q: str = None, page_size: int = 20) -> List[ArticleItem]:
        if self.newsapi_key:
            return self._fetch_newsapi(q, page_size)
        else:
            return self._mock_data()

    def _fetch_newsapi(self, q: str = None, page_size: int = 20) -> List[ArticleItem]:
        url = "https://newsapi.org/v2/top-headlines"
        params = {"pageSize": page_size, "language": "en"}
        if q:
            params["q"] = q
        resp = requests.get(url, params=params, headers={"X-API-Key": self.newsapi_key})
        if resp.status_code != 200:
            st.warning(f"NewsAPI error: {resp.status_code} {resp.text}")
            return self._mock_data()
        data = resp.json()
        items = []
        for a in data.get("articles", []):
            source = a.get("source", {}).get("name", "unknown")
            title = a.get("title") or ""
            desc = a.get("description") or ""
            url = a.get("url") or ""
            pub = a.get("publishedAt") or ""
            content = ""
            # try to fetch full article if possible
            if HAVE_NEWSPAPER and url:
                try:
                    art = Article(url)
                    art.download()
                    art.parse()
                    content = art.text
                except Exception:
                    content = ""
            id_hash = hashlib.sha1((source + title + pub).encode()).hexdigest()[:12]
            items.append(ArticleItem(id=id_hash, source=source, title=title, description=desc, url=url, published_at=pub, content=content))
        return items

    def _mock_data(self) -> List[ArticleItem]:
        samples = [
            {
                "source": "Global News",
                "title": "Major bank suffers ransomware attack disrupting transactions",
                "description": "A major bank reported systems outages after a suspected ransomware attack.",
                "url": "https://example.com/bank-ransomware",
                "published_at": "2025-09-27T08:22:00Z",
                "content": "A major international bank experienced a widespread ransomware attack that has disrupted online and branch transactions for several hours. Customers report inability to access accounts, and internal systems are being taken offline as part of containment."
            },
            {
                "source": "Science Daily",
                "title": "Scientists announce promising new malaria vaccine trial",
                "description": "Early trials show high efficacy in preventing malaria in children.",
                "url": "https://example.com/malaria-vaccine",
                "published_at": "2025-09-27T06:00:00Z",
                "content": "Researchers report a new candidate vaccine showing strong immune response in early-stage trials. Larger trials are needed, and regulatory approvals will take time."
            },
            {
                "source": "TechCrunch",
                "title": "Small startup claims breakthrough in battery tech",
                "description": "A small startup has announced a battery that charges in 5 minutes.",
                "url": "https://example.com/battery-breakthrough",
                "published_at": "2025-09-26T23:10:00Z",
                "content": "Startup claims significant improvements in charging speed, but independent reviewers caution that prototype data is limited and peer review is pending."
            },
            {
                "source": "Celebrity Buzz",
                "title": "Viral hoax: celebrity X declares world ending",
                "description": "A viral post claimed celebrity declared world ending; debunked by major outlets.",
                "url": "https://example.com/celebrity-hoax",
                "published_at": "2025-09-26T21:00:00Z",
                "content": "A viral post circulated claiming that a celebrity announced doomsday; fact checking sites show the claim is false."
            },
        ]
        items = []
        for s in samples:
            id_hash = hashlib.sha1((s["source"] + s["title"] + s["published_at"]).encode()).hexdigest()[:12]
            items.append(ArticleItem(id=id_hash, source=s["source"], title=s["title"], description=s["description"], url=s["url"], published_at=s["published_at"], content=s["content"]))
        return items

class SummariserAgent:
    """
    Summarises content using OpenAI GPT if available, Gemini AI as fallback, transformers as second fallback, else heuristic.
    """
    def __init__(self):
        self.use_transformers = USE_TRANSFORMERS
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_client = None
        self.gemini_model = None
        
        # Initialize OpenAI if API key is available
        if self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                self.openai_client = openai
            except Exception as e:
                st.warning(f"Failed to initialize OpenAI: {e}")
                self.openai_client = None
        
        # Initialize Gemini if API key is available
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Use the most basic available model
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                st.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None

    def summarise(self, article: ArticleItem) -> str:
        text = article.content.strip() or (article.title + ". " + article.description)
        if not text:
            return ""
        
        # Try OpenAI GPT first if available
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a news summarization expert. Summarize the following news article in 2-3 sentences, focusing on the key facts and impact."},
                        {"role": "user", "content": text[:2000]}  # Limit text length
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                return response.choices[0].message.content[:400]  # Limit response length
            except Exception as e:
                st.warning(f"OpenAI summarization failed: {e}")
                # Disable OpenAI for this session if API key is invalid
                if "invalid" in str(e).lower() or "expired" in str(e).lower():
                    st.error("⚠️ OpenAI API key is invalid or expired. Using fallback summarization.")
                    self.openai_client = None
        
        # Try Gemini AI as fallback if available
        if self.gemini_model:
            try:
                prompt = f"Summarize this news article in 2-3 sentences, focusing on the key facts and impact:\n\n{text[:2000]}"  # Limit text length
                response = self.gemini_model.generate_content(prompt)
                return response.text[:400]  # Limit response length
            except Exception as e:
                st.warning(f"Gemini summarization failed: {e}")
                # Disable Gemini for this session if API key is invalid
                if "API_KEY_INVALID" in str(e) or "expired" in str(e).lower():
                    st.error("⚠️ Gemini API key is invalid or expired. Using fallback summarization.")
                    self.gemini_model = None
        
        # If transformers available, use model but keep short
        if self.use_transformers:
            try:
                # keep it short
                s = summarizer_pipeline(text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
                return s
            except Exception:
                pass
        
        # fallback: take first 1-2 sentences
        parts = text.split(".")
        if len(parts) >= 2:
            return (parts[0].strip() + ". " + parts[1].strip())[:400]
        return text[:400]

class ClassifierAgent:
    """
    Very light-weight category classifier via keyword heuristics.
    You can replace with an LLM classifier later.
    """
    CATEGORY_KEYWORDS = {
        "Cybersecurity": ["ransomware", "breach", "hack", "malware", "cyber"],
        "Finance": ["bank", "stock", "market", "finance", "economy", "cryptocurrency", "bitcoin"],
        "Health": ["vaccine", "virus", "health", "WHO", "disease", "dengue", "covid", "malaria"],
        "Tech": ["startup", "ai", "battery", "software", "tech", "chip", "semiconductor"],
        "Climate": ["heatwave", "flood", "drought", "climate", "emissions"],
        "Politics": ["election", "government", "sanctions", "president", "parliament"],
    }

    def classify(self, article: ArticleItem) -> str:
        text = " ".join([article.title, article.description, article.content]).lower()
        scores = {}
        for cat, kws in self.CATEGORY_KEYWORDS.items():
            score = sum(text.count(k) for k in kws)
            if score > 0:
                scores[cat] = score
        if not scores:
            # fallback: general
            return "General"
        # return highest score
        return max(scores.items(), key=lambda x: x[1])[0]

class ImpactScorerAgent:
    """
    Produces an impact score and short explanation.
    Strategy:
      - Score from source credibility mapping (higher for reputable outlets)
      - Keyword severity (ransomware, mass casualty, sanctions => high)
      - Recency / reach (if published recently + contains 'global' or 'millions' => higher)
    """
    SOURCE_SCORES = {
        # higher = more trusted / likely to matter
        "Reuters": 0.95, "BBC": 0.93, "AP": 0.92, "Al Jazeera": 0.85, "CNN": 0.88,
        "TechCrunch": 0.7, "Global News": 0.7, "Science Daily": 0.8, "Unknown": 0.4,
        "Celebrity Buzz": 0.2,
    }
    SEVERE_KEYWORDS = ["ransomware", "attack", "explosion", "dead", "killed", "sanction", "collapse", "mass", "outage", "breach"]
    MEDIUM_KEYWORDS = ["trial", "research", "study", "announce", "release", "prototype"]

    def score(self, article: ArticleItem) -> Dict[str, Any]:
        base = 0.0
        src = article.source or "Unknown"
        # source cred
        src_score = self.SOURCE_SCORES.get(src, 0.5)
        base += src_score * 0.5  # contribute up to 0.5

        text = " ".join([article.title, article.description, article.content]).lower()
        severe_hits = sum(text.count(k) for k in self.SEVERE_KEYWORDS)
        medium_hits = sum(text.count(k) for k in self.MEDIUM_KEYWORDS)
        base += min(severe_hits * 0.2, 0.4)  # severe keywords bump up
        base += min(medium_hits * 0.05, 0.1)

        # recency boost (newer articles slightly higher)
        try:
            if article.published_at:
                dt = dateparser.parse(article.published_at)
                age_seconds = (time.time() - dt.timestamp())
                # less than 6 hours -> small boost
                if age_seconds < 6 * 3600:
                    base += 0.05
        except Exception:
            pass

        # clamp 0..1
        score = max(0.0, min(1.0, base))
        # map to categories
        if score > 0.7:
            impact = "High"
        elif score > 0.4:
            impact = "Medium"
        else:
            impact = "Low"

        # simple why text
        reasons = []
        if severe_hits:
            reasons.append(f"Contains severe keywords ({severe_hits})")
        if medium_hits:
            reasons.append(f"Related to ongoing research/announcements")
        if src_score > 0.85:
            reasons.append(f"Source ({src}) is reputable")
        if not reasons:
            reasons.append("No major severity keywords found; source moderate")

        return {"score": score, "impact": impact, "why": "; ".join(reasons)}

class TwitterAgent:
    """
    Monitors Twitter for trending topics and social sentiment related to news.
    """
    def __init__(self):
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.api = None
        
        # Initialize Twitter API if key is available
        if self.twitter_api_key:
            try:
                # Note: This is a simplified implementation
                # In production, you'd need proper Twitter API v2 setup with bearer token
                self.api = tweepy.Client(bearer_token=self.twitter_api_key)
            except Exception as e:
                st.warning(f"Failed to initialize Twitter API: {e}")
                self.api = None
    
    def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending topics.
        Returns mock data if Twitter API is not available.
        """
        if not self.api:
            return self._mock_twitter_data(limit)
        
        try:
            # This would be the actual Twitter API call
            # For now, returning mock data
            return self._mock_twitter_data(limit)
        except Exception as e:
            st.warning(f"Twitter API error: {e}")
            return self._mock_twitter_data(limit)
    
    def _mock_twitter_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock Twitter data for demonstration"""
        mock_tweets = [
            {
                "text": "Breaking: Major cybersecurity incident reported by financial institutions",
                "author": "TechNews",
                "retweets": 150,
                "likes": 300,
                "sentiment": "negative"
            },
            {
                "text": "New vaccine trial shows promising results for malaria prevention",
                "author": "HealthWatch",
                "retweets": 89,
                "likes": 200,
                "sentiment": "positive"
            },
            {
                "text": "Startup claims revolutionary battery technology breakthrough",
                "author": "InnovationHub",
                "retweets": 45,
                "likes": 120,
                "sentiment": "neutral"
            }
        ]
        
        return mock_tweets[:limit]

class ReporterAgent:
    """
    Turns enriched ArticleItem list into displayable dicts / notifications.
    """
    def __init__(self, notifier_enabled=False):
        self.notifier_enabled = notifier_enabled

    def make_report(self, items: List[ArticleItem]) -> List[Dict[str,Any]]:
        rows = []
        for a in items:
            rows.append({
                "id": a.id,
                "source": a.source,
                "title": a.title,
                "summary": a.summary,
                "category": a.category,
                "impact": a.impact,
                "impact_score": round(a.impact_score, 3),
                "why": a.why,
                "url": a.url,
                "published_at": a.published_at,
            })
        return rows

# -------------------------
# Tiny Crew orchestrator
# -------------------------
class Crew:
    def __init__(self, fetcher: FetcherAgent, summariser: SummariserAgent, classifier: ClassifierAgent, scorer: ImpactScorerAgent, reporter: ReporterAgent, twitter: TwitterAgent = None):
        self.fetcher = fetcher
        self.summariser = summariser
        self.classifier = classifier
        self.scorer = scorer
        self.reporter = reporter
        self.twitter = twitter

    def run(self, limit: int = 20) -> List[Dict[str,Any]]:
        st.info("Fetcher Agent running...")
        items = self.fetcher.fetch(page_size=limit)
        st.success(f"Fetched {len(items)} items")
        
        # Get Twitter data if available
        twitter_data = []
        if self.twitter:
            st.info("Twitter Agent running...")
            twitter_data = self.twitter.get_trending_topics(limit=5)
            st.success(f"Fetched {len(twitter_data)} Twitter trends")
        
        # process each article
        for art in items:
            # summarise
            art.summary = self.summariser.summarise(art)
            # classify
            art.category = self.classifier.classify(art)
            # score
            sc = self.scorer.score(art)
            art.impact_score = sc["score"]
            art.impact = sc["impact"]
            art.why = sc["why"]
        
        # reporter
        report = self.reporter.make_report(items)
        
        # Add Twitter data to report if available
        if twitter_data:
            report.append({
                "id": "twitter_trends",
                "source": "Twitter",
                "title": "Social Media Trends",
                "summary": f"Found {len(twitter_data)} trending topics",
                "category": "Social",
                "impact": "Medium",
                "impact_score": 0.5,
                "why": "Social media sentiment analysis",
                "url": "",
                "published_at": "2025-01-27T00:00:00Z",
                "twitter_data": twitter_data
            })
        
        return report

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="News Impact Radar", layout="wide", initial_sidebar_state="expanded")
st.title("🛰️ News Impact Radar")
st.markdown("Fetch trending news / claims (mock or via NewsAPI), summarise, classify, and score impact.")

# Sidebar settings
st.sidebar.header("API Configuration")
st.sidebar.info("API keys loaded from .env file")

# Show API key status
newsapi_status = "✅ Configured" if os.getenv('NEWSAPI_KEY') else "❌ Missing"
openai_status = "✅ Configured" if os.getenv('OPENAI_API_KEY') else "❌ Missing"
gemini_status = "✅ Configured" if os.getenv('GEMINI_API_KEY') else "❌ Missing"
twitter_status = "✅ Configured" if os.getenv('TWITTER_API_KEY') else "❌ Missing"

st.sidebar.markdown(f"**NewsAPI:** {newsapi_status}")
st.sidebar.markdown(f"**OpenAI GPT:** {openai_status}")
st.sidebar.markdown(f"**Gemini AI:** {gemini_status}")
st.sidebar.markdown(f"**Twitter API:** {twitter_status}")

st.sidebar.header("Settings")
limit = st.sidebar.slider("Max items to fetch", min_value=1, max_value=50, value=8)
run_button = st.sidebar.button("Fetch & Analyse")

# Create agents
fetcher = FetcherAgent()
summariser = SummariserAgent()
classifier = ClassifierAgent()
scorer = ImpactScorerAgent()
reporter = ReporterAgent()
twitter = TwitterAgent()

crew = Crew(fetcher, summariser, classifier, scorer, reporter, twitter)

if run_button:
    with st.spinner("Running CrewAI pipeline..."):
        report = crew.run(limit=limit)
    # Show radar-ish summary: counts per impact
    highs = [r for r in report if r["impact"] == "High"]
    mediums = [r for r in report if r["impact"] == "Medium"]
    lows = [r for r in report if r["impact"] == "Low"]

    # Impact Summary Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔥 High Impact", len(highs), delta=None)
    with col2:
        st.metric("⚡ Medium Impact", len(mediums), delta=None)
    with col3:
        st.metric("📰 Low Impact", len(lows), delta=None)
    
    st.markdown("---")
    
    # Sort by impact score and display all articles
    filtered = sorted(report, key=lambda x: x["impact_score"], reverse=True)
    
    for r in filtered:
        color = {"High": "#ff4b4b", "Medium": "#ff9f1c", "Low": "#2ecc71"}.get(r["impact"], "#888")
        
        # Special handling for Twitter data
        if r.get("twitter_data"):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
                        padding:20px; border-radius:15px; margin-bottom:20px; 
                        border-left: 6px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin:0; color:#fff; font-size:18px; font-weight:600">🐦 {r['title']}</h3>
                <div style="color:#e0e0e0; font-size:16px; margin-top:8px; line-height:1.5">{r['summary']}</div>
                <div style="margin-top:12px; font-size:14px">
                    <span style="background:{color}; color:#fff; padding:6px 12px; border-radius:20px; font-weight:600; margin-right:10px">{r['impact']}</span>
                    <span style="color:#bbb; margin-right:10px">📂 {r['category']}</span>
                    <span style="color:#999; float:right">Score: {r['impact_score']}</span>
                </div>
                <div style="margin-top:8px; color:#aaa; font-size:13px">💡 {r['why']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Display Twitter data
            for tweet in r["twitter_data"]:
                sentiment_color = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}.get(tweet["sentiment"], "#95a5a6")
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding:12px; border-radius:10px; margin:8px 0; 
                            border-left: 4px solid {sentiment_color}; margin-left:20px;">
                    <div style="font-size:14px; color:#e0e0e0; line-height:1.4">{tweet['text']}</div>
                    <div style="font-size:12px; color:#999; margin-top:6px">
                        👤 @{tweet['author']} • 🔄 {tweet['retweets']} • ❤️ {tweet['likes']} • 
                        <span style="color:{sentiment_color}; font-weight:500">{tweet['sentiment'].title()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
                        padding:20px; border-radius:15px; margin-bottom:20px; 
                        border-left: 6px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin:0; color:#fff; font-size:18px; font-weight:600">{r['title']}</h3>
                <div style="color:#e0e0e0; font-size:16px; margin-top:8px; line-height:1.5">{r['summary']}</div>
                <div style="margin-top:12px; font-size:14px">
                    <span style="background:{color}; color:#fff; padding:6px 12px; border-radius:20px; font-weight:600; margin-right:10px">{r['impact']}</span>
                    <span style="color:#bbb; margin-right:10px">📂 {r['category']}</span>
                    <span style="color:#999; float:right">Score: {r['impact_score']}</span>
                </div>
                <div style="margin-top:8px; color:#aaa; font-size:13px">💡 {r['why']}</div>
                <div style="margin-top:12px; font-size:13px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:8px">
                    <a href="{r['url']}" target="_blank" style="color:#4fc3f7; text-decoration:none; font-weight:500">📖 Read Full Article</a> • 
                    <span style="color:#bbb">🏢 {r['source']}</span> • 
                    <span style="color:#999">🕒 {r['published_at'][:10]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Remove raw JSON for cleaner interface
    # st.markdown("### Raw JSON (for debugging)")
    # st.json(filtered)

else:
    st.info("Press 'Fetch & Analyse' from the sidebar to run the CrewAI pipeline with integrated APIs.")
    st.markdown("""
    **Enhanced Features**
    - **NewsAPI**: Fetches real-time news headlines and articles
    - **OpenAI GPT**: Primary AI summarization and analysis
    - **Gemini AI**: Fallback AI summarization
    - **Twitter API**: Monitors social media trends and sentiment
    - **Fallback Systems**: Uses mock data if APIs are unavailable
    
    **API Status**
    - All API keys are loaded from the `.env` file
    - The system gracefully falls back to mock data if APIs fail
    - Enhanced summarization with OpenAI GPT (primary) and Gemini AI (fallback)
    - Social media monitoring with Twitter integration
    
    **Fail-Safe Options**
    - Mock data available for all services
    - Heuristic fallbacks for classification and scoring
    - Transformers-based summarization as backup
    - Graceful error handling throughout the pipeline
    """)
