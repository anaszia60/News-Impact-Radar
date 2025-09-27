# 🛰️ News Impact Radar - CrewAI Demo

A sophisticated news monitoring and analysis system built with CrewAI framework, featuring real-time news fetching, AI-powered summarization, social media monitoring, and impact scoring.

## 🚀 Features

- **Real-time News Fetching**: Integrates with NewsAPI for live news headlines
- **AI-Powered Summarization**: Uses Google Gemini AI for intelligent article summarization
- **Social Media Monitoring**: Twitter API integration for trending topics and sentiment analysis
- **Impact Scoring**: Advanced algorithm to assess news impact and relevance
- **CrewAI Architecture**: Multi-agent system with specialized roles
- **Fail-Safe Design**: Graceful fallbacks when APIs are unavailable

## 📋 Prerequisites

- Python 3.8 or higher
- API keys for:
  - NewsAPI (free tier available)
  - Google Gemini AI
  - Twitter API v2

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   cd crewaistreamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   The `.env` file is already configured with your API keys:
   ```
   GEMINI_API_KEY=AIzaSyA3QWMgz6P3Yu1Yr88onDM72jRe0KBSMAQ
   TWITTER_API_KEY=hSwBJyUfAPBpO2kFgMCgp7Ozr
   NEWSAPI_KEY=6ccdc3c34bd4429ebafe857e01e1d99b
   ```

4. **Run the application**
   ```bash
   streamlit run file.py
   ```

## 🏗️ Architecture

### CrewAI Agents

1. **FetcherAgent**: Retrieves news articles from NewsAPI
2. **SummariserAgent**: Uses Gemini AI for intelligent summarization
3. **ClassifierAgent**: Categorizes articles by topic
4. **ImpactScorerAgent**: Scores articles based on impact and credibility
5. **TwitterAgent**: Monitors social media trends and sentiment
6. **ReporterAgent**: Formats and presents the final results

### Data Flow

```
NewsAPI → FetcherAgent → SummariserAgent → ClassifierAgent → ImpactScorerAgent
                                                                    ↓
Twitter API → TwitterAgent → ReporterAgent ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

## 🔧 Configuration

### API Keys

All API keys are loaded from the `.env` file. The system will show the status of each API in the sidebar:

- ✅ **Configured**: API key is available and working
- ❌ **Missing**: API key is not configured

### Fail-Safe Options

The system includes multiple fail-safe mechanisms:

1. **Mock Data**: If APIs are unavailable, the system uses realistic mock data
2. **Fallback Summarization**: If Gemini AI fails, uses transformers or heuristic methods
3. **Error Handling**: Graceful degradation when services are down
4. **Offline Mode**: Can run entirely with mock data for testing

## 📊 Usage

1. **Start the application**: Run `streamlit run file.py`
2. **Configure search**: Use the sidebar to set search parameters
3. **Run analysis**: Click "Fetch & Analyse" to start the pipeline
4. **View results**: Results are displayed with impact scores, categories, and summaries
5. **Filter data**: Use the filter options to focus on specific categories or impact levels

## 🎯 Key Features

### News Analysis
- Real-time news fetching from NewsAPI
- AI-powered summarization with Gemini
- Automatic categorization (Cybersecurity, Finance, Health, Tech, etc.)
- Impact scoring based on source credibility and content severity

### Social Media Integration
- Twitter trend monitoring
- Sentiment analysis
- Social engagement metrics
- Real-time social media insights

### Advanced Filtering
- Filter by category (Cybersecurity, Finance, Health, Tech, Climate, Politics)
- Filter by impact level (High, Medium, Low)
- Sort by impact score
- Search by keywords

## 🔒 Security & Privacy

- API keys are stored in `.env` file (not committed to version control)
- All API calls include proper error handling
- No sensitive data is logged or stored
- Graceful fallbacks prevent API key exposure

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys in `.env` file
   - Check API key permissions and quotas
   - Ensure internet connectivity

2. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Performance Issues**
   - Reduce the number of articles fetched
   - Check API rate limits
   - Use mock data for testing

### Fallback Options

If you encounter issues:

1. **Use Mock Data**: The system automatically falls back to mock data if APIs fail
2. **Disable APIs**: Comment out API keys in `.env` to force mock mode
3. **Check Logs**: Streamlit shows detailed error messages in the interface

## 📈 Performance Optimization

- **Batch Processing**: Articles are processed in batches for efficiency
- **Caching**: API responses are cached to reduce redundant calls
- **Rate Limiting**: Built-in rate limiting to respect API quotas
- **Async Processing**: Non-blocking operations for better user experience

## 🔮 Future Enhancements

- [ ] Real-time notifications for high-impact news
- [ ] Advanced sentiment analysis with multiple models
- [ ] Custom alert rules and thresholds
- [ ] Export functionality (PDF, CSV, JSON)
- [ ] Multi-language support
- [ ] Advanced visualization and dashboards

## 📝 License

This project is for demonstration purposes. Please ensure you comply with the terms of service for all integrated APIs.

## 🤝 Contributing

This is a demo project showcasing CrewAI capabilities. Feel free to fork and modify for your own use cases.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify API key configuration
3. Test with mock data first
4. Check Streamlit logs for detailed error messages

---

**Note**: This application is designed with fail-safe mechanisms to ensure it works even when external APIs are unavailable. The mock data provides a realistic demonstration of the system's capabilities.

