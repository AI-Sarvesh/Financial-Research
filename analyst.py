import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import tweepy
import yfinance as yf
from datetime import datetime, timedelta
import os
import praw
from dotenv import load_dotenv
import feedparser
from bs4 import BeautifulSoup
import requests
from newspaper import Article
import time
from lxml_html_clean import clean_html 


load_dotenv()


class AdvancedEquityResearchSystem:
    def __init__(self, anthropic_api_key: str, serper_api_key:str, reddit_config: Dict = None):
        self.search_tool = SerperDevTool()
        self.scrape_tool = ScrapeWebsiteTool()
        self.setup_environment(anthropic_api_key, serper_api_key)
        self.setup_models()
        self.reddit_config = reddit_config or {}
        
    def setup_environment(self, anthropic_api_key: str, serper_api_key:str):
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        os.environ["Serper_API_KEY"] = serper_api_key

        
    def setup_models(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
    def create_agents(self):
        # Fundamental Analysis Agents
        fundamental_analyst = Agent(
            role="Senior Fundamental Analyst",
            goal="Conduct comprehensive fundamental analysis integrating sentiment signals",
            backstory="""Expert financial analyst combining traditional fundamental analysis 
                        with sentiment indicators. Specializes in identifying value drivers 
                        and growth opportunities while considering market sentiment impact.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        financial_modeler = Agent(
            role="Financial Modeling Expert",
            goal="Build sophisticated financial models incorporating sentiment-adjusted scenarios",
            backstory="""Expert in building complex financial models that integrate 
                        sentiment indicators into valuation assumptions. Specializes in 
                        sentiment-adjusted scenario analysis and forecasting.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        market_intelligence = Agent(
            role="Market Intelligence Specialist",
            goal="Monitor market dynamics and integrate multiple data sources",
            backstory="""Specialist in synthesizing market research, competitive intelligence, 
                        and sentiment indicators. Expert in identifying market trends and 
                        sentiment-driven opportunities.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        # Sentiment Analysis Agents
        news_sentiment_analyst = Agent(
            role="News & Social Sentiment Analyst",
            goal="Analyze sentiment across news and social media channels",
            backstory="""Expert in analyzing sentiment patterns across various media channels. 
                        Specializes in detecting subtle sentiment shifts and their potential 
                        impact on market movements.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        earnings_sentiment_analyst = Agent(
            role="Corporate Communications Analyst",
            goal="Analyze sentiment in corporate communications and earnings materials",
            backstory="""Specialist in analyzing language patterns in corporate communications. 
                        Expert in detecting management sentiment and forward-looking indicators.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        technical_analyst = Agent(
            role="Technical & Sentiment Integration Specialist",
            goal="Integrate technical analysis with sentiment indicators",
            backstory="""Expert in combining technical analysis with sentiment signals. 
                        Specializes in identifying pattern convergence between price 
                        action and sentiment trends.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        risk_analyst = Agent(
            role="Risk & Sentiment Assessment Specialist",
            goal="Evaluate risks considering both fundamental and sentiment factors",
            backstory="""Expert in comprehensive risk assessment incorporating sentiment 
                        indicators. Specializes in identifying potential risks from both 
                        fundamental and sentiment perspectives.""",
            tools=[self.scrape_tool, self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        return [fundamental_analyst, financial_modeler, market_intelligence,
                news_sentiment_analyst, earnings_sentiment_analyst, 
                technical_analyst, risk_analyst]
    
    def finbert_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment using FinBERT model"""
        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_scores = predictions.detach().numpy()[0]
            
            sentiment_dict = {
                'positive': float(sentiment_scores[0]),
                'negative': float(sentiment_scores[1]),
                'neutral': float(sentiment_scores[2]),
                'dominant_sentiment': ['positive', 'negative', 'neutral'][np.argmax(sentiment_scores)]
            }
            results.append(sentiment_dict)
        return results
    
    def create_tasks(self, agents: list, inputs: Dict) -> list:
        fundamental_analysis_task = Task(
            description=f"""Conduct fundamental analysis for {inputs['ticker']} integrating sentiment:
                          1. Analyze financial statements and metrics
                          2. Evaluate competitive position
                          3. Assess growth drivers and risks
                          4. Consider sentiment impact on fundamentals""",
            expected_output="""Comprehensive fundamental analysis including:
                             - Financial health assessment
                             - Competitive analysis
                             - Growth evaluation
                             - Sentiment-adjusted insights""",
            agent=agents[0]
        )
        
        financial_modeling_task = Task(
            description=f"""Build sentiment-adjusted financial models for {inputs['ticker']}:
                          1. Create DCF with sentiment scenarios
                          2. Adjust multiples for sentiment
                          3. Build sentiment-based scenarios
                          4. Calculate sentiment-adjusted valuations""",
            expected_output="""Financial modeling analysis including:
                             - Sentiment-adjusted DCF
                             - Multiple analysis
                             - Scenario results
                             - Final valuation range""",
            agent=agents[1]
        )
        
        market_intelligence_task = Task(
            description=f"""Analyze market context for {inputs['ticker']}:
                          1. Monitor news and announcements
                          2. Track industry trends
                          3. Analyze competitive landscape
                          4. Evaluate macro factors""",
            expected_output="""Market intelligence report including:
                             - News analysis
                             - Industry assessment
                             - Competitive updates
                             - Macro evaluation""",
            agent=agents[2]
        )
        
        sentiment_analysis_task = Task(
            description=f"""Conduct sentiment analysis for {inputs['ticker']}:
                          1. Analyze news and social sentiment
                          2. Track sentiment trends
                          3. Identify sentiment drivers
                          4. Monitor sentiment shifts""",
            expected_output="""Sentiment analysis report including:
                             - Overall sentiment score
                             - Trend analysis
                             - Key drivers
                             - Signal alerts""",
            agent=agents[3]
        )
        
        earnings_sentiment_task = Task(
            description=f"""Analyze corporate communications for {inputs['ticker']}:
                          1. Review earnings materials
                          2. Analyze management tone
                          3. Track narrative changes
                          4. Evaluate forward guidance""",
            expected_output="""Corporate communications analysis including:
                             - Management tone assessment
                             - Narrative analysis
                             - Guidance evaluation
                             - Key messages""",
            agent=agents[4]
        )
        
        technical_analysis_task = Task(
            description=f"""Perform technical and sentiment analysis for {inputs['ticker']}:
                          1. Analyze price patterns
                          2. Integrate sentiment indicators
                          3. Identify convergence/divergence
                          4. Generate trading signals""",
            expected_output="""Technical analysis report including:
                             - Pattern analysis
                             - Sentiment integration
                             - Signal identification
                             - Trading recommendations""",
            agent=agents[5]
        )
        
        risk_assessment_task = Task(
            description=f"""Evaluate risks for {inputs['ticker']}:
                          1. Assess fundamental risks
                          2. Analyze sentiment risks
                          3. Identify potential catalysts
                          4. Suggest risk mitigation strategies""",
            expected_output="""Risk assessment report including:
                             - Fundamental risks
                             - Sentiment risks
                             - Catalyst analysis
                             - Mitigation strategies""",
            agent=agents[6]
        )
        
        return [fundamental_analysis_task, financial_modeling_task, 
                market_intelligence_task, sentiment_analysis_task,
                earnings_sentiment_task, technical_analysis_task,
                risk_assessment_task]
    
    def get_reddit_sentiment(self, ticker: str) -> List[Dict]:
        """Fetch and analyze Reddit posts about the stock"""
        if not self.reddit_config:
            return []
            
        reddit = praw.Reddit(
            client_id=self.reddit_config.get('client_id'),
            client_secret=self.reddit_config.get('client_secret'),
            user_agent=self.reddit_config.get('user_agent', 'EquityResearchBot/1.0')
        )
        
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        posts = []
        
        for subreddit in subreddits:
            try:
                for post in reddit.subreddit(subreddit).search(f"{ticker}", limit=20):
                    posts.append({
                        'text': f"{post.title} {post.selftext}",
                        'created_at': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'source': f"reddit/{subreddit}"
                    })
            except Exception as e:
                print(f"Error fetching from r/{subreddit}: {e}")
                
        return posts

    def get_market_data(self, ticker: str, timeframe: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fetch market and social data"""
        # Get stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(timeframe.split()[0]))
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date)
        
        # Get sentiment data from multiple sources
        reddit_data = self.get_reddit_sentiment(ticker)
        news_data = self.get_news_sentiment(ticker)
        stocktwits_data = self.get_stocktwits_sentiment(ticker)
        
        # Combine all sentiment data
        sentiment_data = []
        for data in [reddit_data, news_data, stocktwits_data]:
            sentiment_data.extend(data)
            
        # Sort by date
        sentiment_data.sort(key=lambda x: x['created_at'], reverse=True)
        
        return stock_data, sentiment_data
        
    def get_stocktwits_sentiment(self, ticker: str) -> List[Dict]:
        """Fetch public StockTwits messages (no API key required)"""
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return [{
                    'text': message['body'],
                    'created_at': datetime.strptime(message['created_at'], "%Y-%m-%dT%H:%M:%SZ"),
                    'source': 'stocktwits',
                    'sentiment': message.get('entities', {}).get('sentiment', {}).get('basic', 'neutral')
                } for message in data['messages']]
        except Exception as e:
            print(f"Error fetching StockTwits data: {e}")
        return []    
    
    def get_news_sentiment(self, ticker: str) -> List[Dict]:
        """Fetch and analyze news articles about the stock"""
        news_sources = [
            f"https://finance.yahoo.com/rss/headline?s={ticker}",
            f"https://seekingalpha.com/api/sa/combined/{ticker}.xml",
            # Add more RSS feeds as needed
        ]
        
        news_items = []
        for source in news_sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:10]:  # Get latest 10 articles
                    try:
                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        
                        news_items.append({
                            'text': f"{article.title} {article.text}",
                            'created_at': datetime(*entry.published_parsed[:6]),
                            'source': source,
                            'url': entry.link
                        })
                    except Exception as e:
                        print(f"Error processing article {entry.link}: {e}")
                        
                    # Respect rate limits
                    time.sleep(1)
            except Exception as e:
                print(f"Error fetching from {source}: {e}")
                
        return news_items

    def run_analysis(self, inputs: Dict) -> Dict:
        """Execute comprehensive analysis"""
        agents = self.create_agents()
        tasks = self.create_tasks(agents, inputs)
        
        # Initialize crew with hierarchical process
        crew = Crew(
        agents=agents,
        tasks=tasks,
        manager_llm=ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.5,
        openai_api_key=os.getenv('OPENAI_API_KEY')  

        ),
    process=Process.hierarchical,
    verbose=True
    )
        # Get market and social data
        stock_data, social_data = self.get_market_data(
            inputs['ticker'], 
            inputs['timeframe']
        )
        
        # Analyze social sentiment if available
        social_sentiment = []
        if social_data:
            social_sentiment = [self.analyze_text_sentiment(post['text']) 
                              for post in social_data]
        
        # Run crew analysis
        crew_analysis = crew.kickoff(inputs=inputs)
        
        # Combine all analyses
        return {
            'crew_analysis': crew_analysis,
            'stock_data': stock_data.to_dict(),
            'social_sentiment': social_sentiment,
            'analysis_timestamp': datetime.now().isoformat(),
            'metadata': {
                'ticker': inputs['ticker'],
                'timeframe': inputs['timeframe'],
                'analysis_type': inputs.get('analysis_type', 'comprehensive')
            }
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze text sentiment using multiple methods"""
        finbert_result = self.finbert_sentiment([text])[0]
        blob_analysis = TextBlob(text)
        
        return {
            'finbert': finbert_result,
            'textblob_polarity': blob_analysis.sentiment.polarity,
            'text': text
        }

# Example usage
if __name__ == "__main__":
    analysis_inputs = {
        'ticker': 'AAPL',
        'timeframe': '30 days',
        'analysis_type': 'comprehensive',
        'risk_profile': 'moderate',
        'special_considerations': ['ai_capabilities', 'market_sentiment']
    }
    
    reddit_config = None
    
    research_system = AdvancedEquityResearchSystem(
        anthropic_api_key="your-anthropic-key",
        reddit_config=reddit_config,  # Optional
        serper_api_key="your-serper-key"
    )
    result = research_system.run_analysis(analysis_inputs)

     # Print results
    print("Analysis Results:")
    print(result)
