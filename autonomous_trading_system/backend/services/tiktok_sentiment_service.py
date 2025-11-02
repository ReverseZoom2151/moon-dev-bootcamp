"""
TikTok Sentiment Analysis Service
Native implementation for the Autonomous Trading System

Features:
- Real-time TikTok content analysis using browser automation
- AI-powered sentiment extraction from videos and comments
- Trading signal generation based on social sentiment
- Integration with existing ATS social sentiment infrastructure
- Enterprise-grade async architecture with proper error handling
"""

import asyncio
import logging
import traceback
import base64
import json
import re
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Browser automation imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# AI and analysis imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# ATS imports
from core.config import get_settings
from services.social_sentiment_service import SentimentData, TrendingToken

logger = logging.getLogger(__name__)

@dataclass
class TikTokVideoData:
    """Represents a TikTok video with analysis data"""
    video_id: str
    url: str
    username: str
    description: str
    view_count: int
    like_count: int
    comment_count: int
    share_count: int
    timestamp: datetime
    is_live: bool = False
    hashtags: List[str] = None
    mentions: List[str] = None
    
    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []

@dataclass
class TikTokComment:
    """Represents a TikTok comment"""
    comment_id: str
    video_id: str
    username: str
    text: str
    like_count: int
    timestamp: datetime
    is_reply: bool = False
    parent_comment_id: Optional[str] = None

@dataclass
class TikTokAnalysisResult:
    """Results from TikTok content analysis"""
    video_data: TikTokVideoData
    comments: List[TikTokComment]
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    crypto_mentions: List[str]
    trading_signals: List[str]
    meme_score: int        # 1-10
    viral_potential: float # 0-1
    analysis_timestamp: datetime
    raw_ai_analysis: str
    screenshot_path: Optional[str] = None

@dataclass
class SentimentData:
    """Sentiment data compatible with ATS social sentiment service"""
    platform: str
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    mention_count: int
    engagement_score: float
    timestamp: datetime
    raw_text: Optional[str] = None
    metadata: Optional[Dict] = None

class TikTokSentimentService:
    """
    Enterprise TikTok Sentiment Analysis Service
    
    Provides real-time TikTok content analysis for trading insights:
    - Automated browser-based TikTok scraping
    - AI-powered content and comment analysis
    - Crypto mention detection and sentiment scoring
    - Trading signal generation
    - Integration with ATS social sentiment pipeline
    """
    
    def __init__(self, config=None):
        self.settings = config or get_settings()
        
        # Check dependencies
        if not SELENIUM_AVAILABLE:
            logger.warning("⚠️ Selenium not available. Install with: pip install selenium")
        if not OPENAI_AVAILABLE:
            logger.warning("⚠️ OpenAI not available. Install with: pip install openai")
        
        # Service configuration
        self.enabled = getattr(self.settings, 'TIKTOK_SENTIMENT_ENABLED', True)
        self.max_videos_per_session = getattr(self.settings, 'TIKTOK_MAX_VIDEOS_PER_SESSION', 50)
        self.analysis_interval = getattr(self.settings, 'TIKTOK_ANALYSIS_INTERVAL', 1800)  # 30 minutes
        self.screenshot_enabled = getattr(self.settings, 'TIKTOK_SCREENSHOT_ENABLED', True)
        
        # Browser configuration
        self.headless_mode = getattr(self.settings, 'TIKTOK_HEADLESS_MODE', False)
        self.browser_timeout = getattr(self.settings, 'TIKTOK_BROWSER_TIMEOUT', 30)
        self.page_load_timeout = getattr(self.settings, 'TIKTOK_PAGE_LOAD_TIMEOUT', 15)
        
        # AI configuration
        self.openai_api_key = getattr(self.settings, 'OPENAI_API_KEY', '')
        self.ai_model = getattr(self.settings, 'TIKTOK_AI_MODEL', 'gpt-4o-mini')
        self.ai_max_tokens = getattr(self.settings, 'TIKTOK_AI_MAX_TOKENS', 1000)
        
        # Data storage configuration
        self.data_dir = Path(getattr(self.settings, 'BASE_DIR', '.')) / 'data' / 'tiktok_sentiment'
        self.screenshots_dir = self.data_dir / 'screenshots'
        self.analysis_csv = self.data_dir / 'tiktok_analysis.csv'
        self.live_videos_file = self.data_dir / 'live_videos.txt'
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Trading signal configuration
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'solana', 'sol', 'trading', 'pump', 'dump', 'moon', 'bullish', 'bearish',
            'defi', 'nft', 'altcoin', 'hodl', 'diamond hands', 'paper hands',
            'to the moon', 'rug pull', 'whale', 'degen', 'ape', 'fomo', 'lambo',
            'meme coin', 'shitcoin', 'gem', 'moonshot', 'x100', 'x1000'
        ]
        
        # Token patterns for detection
        self.token_patterns = [
            r'\$([A-Z]{2,10})',  # $BTC, $ETH format
            r'#([A-Z]{2,10})',   # #BTC, #ETH format
            r'\b([A-Z]{2,10})\s*(?:coin|token|crypto)\b',  # BTC coin, ETH token
            r'\b([A-Z]{2,10})\s*(?:to|the|moon)\b',  # BTC to moon
        ]
        
        # State management
        self.driver = None
        self.is_running = False
        self.analysis_results = []
        self.session_stats = {
            'videos_analyzed': 0,
            'crypto_mentions_found': 0,
            'trading_signals_generated': 0,
            'session_start': None,
            'last_analysis': None
        }
        
        # Initialize OpenAI client
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("⚠️ OpenAI API key not configured or OpenAI not available. AI analysis will be disabled.")
        
        logger.info("🎬 TikTok Sentiment Service initialized")
        logger.info(f"📊 Max videos per session: {self.max_videos_per_session}")
        logger.info(f"🤖 AI model: {self.ai_model}")
        logger.info(f"📁 Data directory: {self.data_dir}")
    
    async def start(self):
        """Start the TikTok sentiment analysis service"""
        if not self.enabled:
            logger.warning("⚠️ TikTok sentiment service is disabled")
            return
        
        if not SELENIUM_AVAILABLE:
            logger.error("❌ Cannot start TikTok service: Selenium not available")
            return
        
        self.is_running = True
        self.session_stats['session_start'] = datetime.utcnow()
        
        logger.info("🚀 Starting TikTok Sentiment Analysis Service...")
        
        try:
            # Initialize browser
            await self._initialize_browser()
            
            # Start main analysis loop
            await self._analysis_loop()
            
        except Exception as e:
            logger.error(f"❌ Error starting TikTok sentiment service: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.cleanup()
    
    async def stop(self):
        """Stop the service"""
        self.is_running = False
        logger.info("🛑 TikTok sentiment service stopped")
    
    async def _initialize_browser(self):
        """Initialize Selenium WebDriver for TikTok scraping"""
        try:
            logger.info("🌐 Initializing browser for TikTok scraping...")
            
            chrome_options = Options()
            
            if self.headless_mode:
                chrome_options.add_argument('--headless')
            
            # Browser optimization
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # Disable notifications and popups
            prefs = {
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_settings.popups": 0,
                "profile.managed_default_content_settings.images": 2
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.page_load_timeout)
            self.driver.implicitly_wait(10)
            
            logger.info("✅ Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing browser: {e}")
            raise
    
    async def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_running:
            try:
                logger.info("🔄 Starting new TikTok analysis session...")
                
                # Navigate to TikTok For You page
                await self._navigate_to_tiktok()
                
                # Analyze videos
                session_results = await self._analyze_videos()
                
                # Process results
                await self._process_session_results(session_results)
                
                # Wait before next session
                logger.info(f"😴 Waiting {self.analysis_interval} seconds before next analysis session...")
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in analysis loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _navigate_to_tiktok(self):
        """Navigate to TikTok For You page"""
        try:
            tiktok_url = "https://www.tiktok.com/foryou"
            logger.info(f"🌐 Navigating to {tiktok_url}")
            
            self.driver.get(tiktok_url)
            
            # Wait for page to load
            await asyncio.sleep(5)
            
            # Handle any popups or cookie banners
            await self._handle_popups()
            
            logger.info("✅ Successfully navigated to TikTok")
            
        except Exception as e:
            logger.error(f"❌ Error navigating to TikTok: {e}")
            raise
    
    async def _handle_popups(self):
        """Handle TikTok popups and cookie banners"""
        try:
            # Common popup selectors
            popup_selectors = [
                '[data-e2e="close-icon"]',
                '[data-e2e="modal-close-inner-button"]',
                'button[aria-label="Close"]',
                '.tiktok-modal-close-button',
                '[data-testid="close-button"]'
            ]
            
            for selector in popup_selectors:
                try:
                    element = WebDriverWait(self.driver, 2).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    element.click()
                    logger.info(f"✅ Closed popup with selector: {selector}")
                    await asyncio.sleep(1)
                except TimeoutException:
                    continue
                except Exception as e:
                    logger.debug(f"Could not close popup with selector {selector}: {e}")
                    continue
            
        except Exception as e:
            logger.debug(f"No popups to handle: {e}")
    
    async def _analyze_videos(self) -> List[TikTokAnalysisResult]:
        """Analyze TikTok videos in the current session"""
        results = []
        videos_processed = 0
        
        try:
            while videos_processed < self.max_videos_per_session and self.is_running:
                try:
                    # Check if current video is live
                    is_live = await self._is_live_video()
                    
                    if is_live:
                        logger.info("⏭️ Skipping live video")
                        await self._scroll_to_next_video()
                        continue
                    
                    # Extract video data
                    video_data = await self._extract_video_data()
                    
                    if not video_data:
                        logger.warning("⚠️ Could not extract video data, skipping...")
                        await self._scroll_to_next_video()
                        continue
                    
                    # Take screenshot if enabled
                    screenshot_path = None
                    if self.screenshot_enabled:
                        screenshot_path = await self._take_screenshot(video_data.video_id)
                    
                    # Extract comments
                    comments = await self._extract_comments(video_data.video_id)
                    
                    # Analyze content with AI
                    analysis_result = await self._analyze_content_with_ai(
                        video_data, comments, screenshot_path
                    )
                    
                    if analysis_result:
                        results.append(analysis_result)
                        videos_processed += 1
                        
                        logger.info(f"✅ Analyzed video {videos_processed}/{self.max_videos_per_session}")
                        logger.info(f"   Sentiment: {analysis_result.sentiment_score:.2f}")
                        logger.info(f"   Crypto mentions: {len(analysis_result.crypto_mentions)}")
                    
                    # Scroll to next video
                    await self._scroll_to_next_video()
                    
                    # Random delay to avoid detection
                    await asyncio.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing video {videos_processed + 1}: {e}")
                    await self._scroll_to_next_video()
                    await asyncio.sleep(3)
                    continue
            
            logger.info(f"🎬 Session complete: analyzed {videos_processed} videos")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in video analysis: {e}")
            return results
    
    async def _is_live_video(self) -> bool:
        """Check if current video is a live stream"""
        try:
            # Check URL for live indicators
            current_url = self.driver.current_url
            live_indicators = ['/live', 'live_room_mode']
            
            if any(indicator in current_url.lower() for indicator in live_indicators):
                return True
            
            # Check for live badges or indicators in the DOM
            live_selectors = [
                '[data-e2e="live-badge"]',
                '.live-indicator',
                '[data-testid="live-badge"]'
            ]
            
            for selector in live_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element and element.is_displayed():
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if video is live: {e}")
            return False
    
    async def _extract_video_data(self) -> Optional[TikTokVideoData]:
        """Extract video metadata from current TikTok video"""
        try:
            # Generate video ID from URL or timestamp
            video_id = f"tiktok_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Extract URL
            url = self.driver.current_url
            
            # Extract username
            username = "unknown"
            try:
                username_element = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="video-author-uniqueid"]')
                username = username_element.text.strip('@')
            except:
                pass
            
            # Extract description
            description = ""
            try:
                desc_element = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="video-desc"]')
                description = desc_element.text
            except:
                pass
            
            # Extract engagement metrics (these might not be available)
            view_count = 0
            like_count = 0
            comment_count = 0
            share_count = 0
            
            # Extract hashtags and mentions
            hashtags = re.findall(r'#(\w+)', description)
            mentions = re.findall(r'@(\w+)', description)
            
            return TikTokVideoData(
                video_id=video_id,
                url=url,
                username=username,
                description=description,
                view_count=view_count,
                like_count=like_count,
                comment_count=comment_count,
                share_count=share_count,
                timestamp=datetime.utcnow(),
                hashtags=hashtags,
                mentions=mentions
            )
            
        except Exception as e:
            logger.error(f"❌ Error extracting video data: {e}")
            return None
    
    async def _take_screenshot(self, video_id: str) -> Optional[str]:
        """Take screenshot of current video"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.screenshots_dir / f"tiktok_{video_id}_{timestamp}.png"
            
            self.driver.save_screenshot(str(screenshot_path))
            
            logger.debug(f"📸 Screenshot saved: {screenshot_path}")
            return str(screenshot_path)
            
        except Exception as e:
            logger.error(f"❌ Error taking screenshot: {e}")
            return None
    
    async def _extract_comments(self, video_id: str) -> List[TikTokComment]:
        """Extract comments from current video"""
        comments = []
        
        try:
            # Try to open comments section
            try:
                comment_button = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="comment-icon"]')
                comment_button.click()
                await asyncio.sleep(2)
            except:
                logger.debug("Could not open comments section")
                return comments
            
            # Extract visible comments
            comment_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-e2e="comment-item"]')
            
            for i, element in enumerate(comment_elements[:10]):  # Limit to first 10 comments
                try:
                    # Extract comment text
                    text_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="comment-text"]')
                    text = text_element.text
                    
                    # Extract username
                    username_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="comment-username"]')
                    username = username_element.text
                    
                    comment = TikTokComment(
                        comment_id=f"{video_id}_comment_{i}",
                        video_id=video_id,
                        username=username,
                        text=text,
                        like_count=0,  # Not easily extractable
                        timestamp=datetime.utcnow()
                    )
                    
                    comments.append(comment)
                    
                except Exception as e:
                    logger.debug(f"Error extracting comment {i}: {e}")
                    continue
            
            # Close comments section
            try:
                close_button = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="comment-close"]')
                close_button.click()
                await asyncio.sleep(1)
            except:
                pass
            
            logger.debug(f"📝 Extracted {len(comments)} comments")
            return comments
            
        except Exception as e:
            logger.error(f"❌ Error extracting comments: {e}")
            return comments
    
    async def _analyze_content_with_ai(
        self, 
        video_data: TikTokVideoData, 
        comments: List[TikTokComment],
        screenshot_path: Optional[str]
    ) -> Optional[TikTokAnalysisResult]:
        """Analyze video content and comments using AI"""
        try:
            if not self.openai_client:
                logger.warning("⚠️ OpenAI client not available, using fallback analysis")
                return self._fallback_analysis(video_data, comments, screenshot_path)
            
            # Prepare content for analysis
            content_text = f"""
            Video Description: {video_data.description}
            Username: @{video_data.username}
            Hashtags: {', '.join(video_data.hashtags)}
            
            Comments:
            {chr(10).join([f"@{c.username}: {c.text}" for c in comments[:5]])}
            """
            
            # Prepare AI prompt
            prompt = f"""
            Analyze this TikTok content for cryptocurrency and trading sentiment:
            
            {content_text}
            
            Please provide:
            1. Overall sentiment score (-1 to 1, where -1 is very bearish, 1 is very bullish)
            2. Confidence level (0 to 1)
            3. List of cryptocurrency mentions found
            4. Trading signals or insights
            5. Meme potential score (1-10)
            6. Viral potential (0-1)
            
            Respond in JSON format:
            {{
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "crypto_mentions": [],
                "trading_signals": [],
                "meme_score": 0,
                "viral_potential": 0.0,
                "analysis": "detailed analysis text"
            }}
            """
            
            # Call OpenAI API
            response = await self._call_openai_api(prompt, screenshot_path)
            
            if not response:
                return self._fallback_analysis(video_data, comments, screenshot_path)
            
            # Parse AI response
            try:
                ai_result = json.loads(response)
            except json.JSONDecodeError:
                logger.error("❌ Failed to parse AI response as JSON")
                # Fallback: create basic analysis
                ai_result = {
                    "sentiment_score": 0.0,
                    "confidence": 0.5,
                    "crypto_mentions": self._extract_crypto_mentions(content_text),
                    "trading_signals": [],
                    "meme_score": 5,
                    "viral_potential": 0.5,
                    "analysis": response
                }
            
            # Create analysis result
            analysis_result = TikTokAnalysisResult(
                video_data=video_data,
                comments=comments,
                sentiment_score=float(ai_result.get('sentiment_score', 0.0)),
                confidence=float(ai_result.get('confidence', 0.5)),
                crypto_mentions=ai_result.get('crypto_mentions', []),
                trading_signals=ai_result.get('trading_signals', []),
                meme_score=int(ai_result.get('meme_score', 5)),
                viral_potential=float(ai_result.get('viral_potential', 0.5)),
                analysis_timestamp=datetime.utcnow(),
                raw_ai_analysis=ai_result.get('analysis', ''),
                screenshot_path=screenshot_path
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {e}")
            return self._fallback_analysis(video_data, comments, screenshot_path)
    
    def _fallback_analysis(
        self, 
        video_data: TikTokVideoData, 
        comments: List[TikTokComment],
        screenshot_path: Optional[str]
    ) -> TikTokAnalysisResult:
        """Fallback analysis when AI is not available"""
        try:
            # Combine all text for analysis
            all_text = video_data.description + " " + " ".join([c.text for c in comments])
            
            # Extract crypto mentions
            crypto_mentions = self._extract_crypto_mentions(all_text)
            
            # Basic sentiment analysis using TextBlob if available
            sentiment_score = 0.0
            confidence = 0.3
            
            if TEXTBLOB_AVAILABLE and all_text.strip():
                try:
                    blob = TextBlob(all_text)
                    sentiment_score = blob.sentiment.polarity
                    confidence = min(abs(sentiment_score) + 0.3, 1.0)
                except:
                    pass
            
            # Basic meme score based on hashtags and keywords
            meme_keywords = ['meme', 'funny', 'lol', 'viral', 'trending']
            meme_score = min(
                sum(1 for keyword in meme_keywords if keyword in all_text.lower()) + 
                len(video_data.hashtags), 
                10
            )
            
            # Viral potential based on engagement indicators
            viral_potential = min(
                (len(video_data.hashtags) * 0.1 + 
                 len(comments) * 0.05 + 
                 len(crypto_mentions) * 0.2), 
                1.0
            )
            
            return TikTokAnalysisResult(
                video_data=video_data,
                comments=comments,
                sentiment_score=sentiment_score,
                confidence=confidence,
                crypto_mentions=crypto_mentions,
                trading_signals=[],
                meme_score=meme_score,
                viral_potential=viral_potential,
                analysis_timestamp=datetime.utcnow(),
                raw_ai_analysis="Fallback analysis - AI not available",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            logger.error(f"❌ Error in fallback analysis: {e}")
            return None
    
    async def _call_openai_api(self, prompt: str, screenshot_path: Optional[str] = None) -> Optional[str]:
        """Call OpenAI API for content analysis"""
        try:
            messages = [
                {"role": "system", "content": "You are an expert cryptocurrency and social media analyst."},
                {"role": "user", "content": prompt}
            ]
            
            # Add image if screenshot is available
            if screenshot_path and Path(screenshot_path).exists():
                try:
                    with open(screenshot_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    messages[1]["content"] = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                except Exception as e:
                    logger.warning(f"⚠️ Could not include screenshot in analysis: {e}")
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model=self.ai_model,
                messages=messages,
                max_tokens=self.ai_max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Error calling OpenAI API: {e}")
            return None
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text"""
        mentions = []
        
        try:
            text_lower = text.lower()
            
            # Check for crypto keywords
            for keyword in self.crypto_keywords:
                if keyword in text_lower:
                    mentions.append(keyword.upper())
            
            # Check for token patterns
            for pattern in self.token_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                mentions.extend([match.upper() for match in matches])
            
            # Remove duplicates and return
            return list(set(mentions))
            
        except Exception as e:
            logger.error(f"❌ Error extracting crypto mentions: {e}")
            return []
    
    async def _scroll_to_next_video(self):
        """Scroll to the next TikTok video"""
        try:
            # Use arrow key to scroll to next video
            self.driver.find_element(By.TAG_NAME, 'body').send_keys('ArrowDown')
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error scrolling to next video: {e}")
    
    async def _process_session_results(self, results: List[TikTokAnalysisResult]):
        """Process and store analysis results"""
        try:
            if not results:
                logger.info("📊 No results to process")
                return
            
            # Update session stats
            self.session_stats['videos_analyzed'] += len(results)
            self.session_stats['last_analysis'] = datetime.utcnow()
            
            crypto_mentions = sum(len(r.crypto_mentions) for r in results)
            trading_signals = sum(len(r.trading_signals) for r in results)
            
            self.session_stats['crypto_mentions_found'] += crypto_mentions
            self.session_stats['trading_signals_generated'] += trading_signals
            
            # Save to CSV
            await self._save_results_to_csv(results)
            
            # Generate sentiment data for ATS integration
            await self._generate_sentiment_data(results)
            
            # Log summary
            logger.info(f"📊 Session Results Summary:")
            logger.info(f"   Videos analyzed: {len(results)}")
            logger.info(f"   Crypto mentions: {crypto_mentions}")
            logger.info(f"   Trading signals: {trading_signals}")
            logger.info(f"   Average sentiment: {np.mean([r.sentiment_score for r in results]):.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error processing session results: {e}")
    
    async def _save_results_to_csv(self, results: List[TikTokAnalysisResult]):
        """Save analysis results to CSV file"""
        try:
            # Prepare data for CSV
            csv_data = []
            
            for result in results:
                row = {
                    'timestamp': result.analysis_timestamp.isoformat(),
                    'video_id': result.video_data.video_id,
                    'url': result.video_data.url,
                    'username': result.video_data.username,
                    'description': result.video_data.description,
                    'hashtags': ','.join(result.video_data.hashtags),
                    'sentiment_score': result.sentiment_score,
                    'confidence': result.confidence,
                    'crypto_mentions': ','.join(result.crypto_mentions),
                    'trading_signals': ','.join(result.trading_signals),
                    'meme_score': result.meme_score,
                    'viral_potential': result.viral_potential,
                    'comment_count': len(result.comments),
                    'screenshot_path': result.screenshot_path or '',
                    'ai_analysis': result.raw_ai_analysis
                }
                csv_data.append(row)
            
            # Save to CSV
            df = pd.DataFrame(csv_data)
            
            if self.analysis_csv.exists():
                # Append to existing file
                df.to_csv(self.analysis_csv, mode='a', header=False, index=False)
            else:
                # Create new file with headers
                df.to_csv(self.analysis_csv, index=False)
            
            logger.info(f"💾 Saved {len(results)} results to {self.analysis_csv}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results to CSV: {e}")
    
    async def _generate_sentiment_data(self, results: List[TikTokAnalysisResult]):
        """Generate SentimentData objects for ATS integration"""
        try:
            sentiment_data_list = []
            
            for result in results:
                # Create sentiment data for each crypto mention
                for crypto_symbol in result.crypto_mentions:
                    sentiment_data = SentimentData(
                        platform='tiktok',
                        symbol=crypto_symbol,
                        sentiment_score=result.sentiment_score,
                        confidence=result.confidence,
                        mention_count=1,  # Each video counts as 1 mention
                        engagement_score=result.viral_potential,
                        timestamp=result.analysis_timestamp,
                        raw_text=result.video_data.description,
                        metadata={
                            'video_id': result.video_data.video_id,
                            'username': result.video_data.username,
                            'meme_score': result.meme_score,
                            'trading_signals': result.trading_signals,
                            'hashtags': result.video_data.hashtags
                        }
                    )
                    sentiment_data_list.append(sentiment_data)
            
            # Store for potential integration with social sentiment service
            self.analysis_results.extend(sentiment_data_list)
            
            # Keep only recent data (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.analysis_results = [
                data for data in self.analysis_results 
                if data.timestamp > cutoff_time
            ]
            
            logger.info(f"📈 Generated {len(sentiment_data_list)} sentiment data points")
            
        except Exception as e:
            logger.error(f"❌ Error generating sentiment data: {e}")
    
    async def get_recent_sentiment(self, hours: int = 24) -> List[SentimentData]:
        """Get recent TikTok sentiment data"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_data = [
                data for data in self.analysis_results 
                if data.timestamp > cutoff_time
            ]
            return recent_data
            
        except Exception as e:
            logger.error(f"❌ Error getting recent sentiment: {e}")
            return []
    
    async def get_trending_tokens(self) -> List[Dict[str, Any]]:
        """Get trending tokens from TikTok analysis"""
        try:
            # Aggregate mentions by symbol
            symbol_stats = {}
            
            for data in self.analysis_results:
                symbol = data.symbol
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'symbol': symbol,
                        'mention_count': 0,
                        'total_sentiment': 0,
                        'total_confidence': 0,
                        'total_engagement': 0,
                        'first_seen': data.timestamp,
                        'last_seen': data.timestamp,
                        'videos': []
                    }
                
                stats = symbol_stats[symbol]
                stats['mention_count'] += 1
                stats['total_sentiment'] += data.sentiment_score
                stats['total_confidence'] += data.confidence
                stats['total_engagement'] += data.engagement_score
                stats['last_seen'] = max(stats['last_seen'], data.timestamp)
                stats['videos'].append(data.metadata.get('video_id', ''))
            
            # Calculate averages and sort by trending score
            trending_tokens = []
            
            for symbol, stats in symbol_stats.items():
                if stats['mention_count'] >= 2:  # Minimum mentions threshold
                    avg_sentiment = stats['total_sentiment'] / stats['mention_count']
                    avg_confidence = stats['total_confidence'] / stats['mention_count']
                    avg_engagement = stats['total_engagement'] / stats['mention_count']
                    
                    # Calculate trending score
                    trending_score = (
                        stats['mention_count'] * 0.4 +
                        abs(avg_sentiment) * 0.3 +
                        avg_engagement * 0.3
                    )
                    
                    trending_tokens.append({
                        'symbol': symbol,
                        'mention_count': stats['mention_count'],
                        'avg_sentiment': avg_sentiment,
                        'avg_confidence': avg_confidence,
                        'avg_engagement': avg_engagement,
                        'trending_score': trending_score,
                        'first_seen': stats['first_seen'],
                        'last_seen': stats['last_seen'],
                        'unique_videos': len(set(stats['videos']))
                    })
            
            # Sort by trending score
            trending_tokens.sort(key=lambda x: x['trending_score'], reverse=True)
            
            return trending_tokens[:10]  # Top 10 trending
            
        except Exception as e:
            logger.error(f"❌ Error getting trending tokens: {e}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        return {
            'service_name': 'TikTok Sentiment Analysis',
            'enabled': self.enabled,
            'is_running': self.is_running,
            'dependencies': {
                'selenium': SELENIUM_AVAILABLE,
                'openai': OPENAI_AVAILABLE,
                'textblob': TEXTBLOB_AVAILABLE
            },
            'session_stats': self.session_stats.copy(),
            'total_analysis_results': len(self.analysis_results),
            'data_directory': str(self.data_dir),
            'screenshots_enabled': self.screenshot_enabled,
            'ai_model': self.ai_model,
            'max_videos_per_session': self.max_videos_per_session,
            'analysis_interval_minutes': self.analysis_interval // 60
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            logger.info("🧹 TikTok sentiment service cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# Export for use in other modules
__all__ = ['TikTokSentimentService', 'TikTokVideoData', 'TikTokComment', 'TikTokAnalysisResult', 'SentimentData']