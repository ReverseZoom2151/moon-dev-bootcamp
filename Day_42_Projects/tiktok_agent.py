'''
This AI agent will be able to watch TikTok and pull alpha from the comment section. 

I was inspired by the dumb money guys and the social arbitrage trading style that they approach. 

Essentially, social arbitrage is getting information prior to Wall Street. Wall Street is typically very slow at getting information. 

TikTok has all the information, and in the comment section of TikTok, there are some golden opportunities. 

The problem is, if you watch TikTok all day, you're gonna go brain dead. So I don't want to watch TikTok all day, but I do want the alpha. 

This agent will be able to see the trends and connect them with different Boomer Tokens (stocks). 

## 🔎 SEARCH TERMS TO IMPROVE TIKTOK ALGORITHM 🔎
Use these search terms to train your TikTok algorithm to show more relevant content for investment insights:

### Consumer Behavior & Shopping
1. "Product hauls 2024" - People showing off recent purchases gives insight into consumer spending patterns
2. "TikTok made me buy it" - Viral products experiencing sudden demand spikes
3. "Dupes for expensive brands" - Consumer price sensitivity and alternative product markets
4. "Amazon finds under $20" - Budget consumer trends and high-value products
5. "Luxury shopping experience" - Premium consumer market insights
6. "Viral products worth the hype" - Validation of trending products
7. "Brand switching stories" - Consumer loyalty shifts
8. "Shopping regrets" - Failed products and declining brands
9. "Small business finds" - Emerging brands with growth potential
10. "Shopping hacks 2024" - Price sensitivity and consumer behavior

### Technology & Innovation
11. "Tech that changed my life" - Emerging consumer tech with high adoption potential
12. "AI tools for [specific industry]" - Industry-specific AI applications gaining traction
13. "Next big tech trend" - Early adopters showcasing emerging technologies
14. "Tech CEO news" - Leadership changes and company direction
15. "Startup success stories" - Emerging companies gaining traction
16. "Tech fails to avoid" - Products losing market share
17. "New app everyone's using" - Early app adoption trends
18. "Tech founder interviews" - Insights into company strategy
19. "Tech industry layoffs" - Early signals of company contractions
20. "Product launch reactions" - Initial consumer sentiment on new releases

### Financial & Investment
21. "Stock market analysis" - Find creators who discuss market trends and stock analysis
22. "Crypto updates daily" - Cryptocurrency discussions, news, and sentiment
23. "Financial red flags" - Consumer sentiment about economic conditions or companies
24. "Recession proof businesses" - Companies with strong economic moats
25. "Dividend stock picks" - Income investment trends
26. "Financial literacy tips" - Growing investment demographics
27. "Housing market updates" - Real estate trends and sentiment
28. "Side hustle trends 2024" - Alternative income sources growing in popularity
29. "Money mistakes to avoid" - Financial caution indicators
30. "Portfolio diversification" - Investment allocation trends

### Industry-Specific
31. "Healthcare innovation" - Medical technology and healthcare disruption
32. "Green energy solutions" - Renewable energy adoption and innovation
33. "Future of transportation" - Mobility trends and EV adoption
34. "Work from home setups" - Remote work sustainability
35. "Retail store experiences" - Brick and mortar transformation
36. "Factory automation" - Manufacturing technology trends
37. "Restaurant industry changes" - Food service trends and challenges

### Meta-Search
38. "Brands going viral" - Track which companies are gaining social momentum
39. "Products with cult following" - Items with strong consumer loyalty
40. "This company is done" - Early warnings of declining brands

🌙 Moon Dev tip: Search 3-5 terms daily and interact extensively with relevant content to rapidly train the algorithm! 💰
'''

"""
🚀 Moon Dev's TikTok Alpha Scraper
Navigates TikTok, captures video and comment screenshots for trading insights

1. Opens TikTok URL in browser
2. Detects share button position once at the start (to determine comment button position)
3. For each video:
   a. First checks if the video is LIVE by examining the URL
   b. If LIVE, skips processing and moves to next video with down arrow
   c. If not LIVE, continues with screenshot and analysis
4. For regular videos:
   a. Moves to comment button and clicks it
   b. Takes screenshot of the video and comments
   c. Analyzes screenshot with GPT-4o-mini to extract content and comments
   d. Saves analysis to CSV file for trading insights
   e. Uses double-click to ensure browser activation
   f. Presses down arrow to move to next video
5. Repeats steps 3-4 to collect data from regular videos only

The alpha is in the comments! 💰


TODO -
- this works great, til it hits a live.... and then the double click takes us to more live.
the doublclick was to activate the screen to always scroll because it would break sometimes
but those sometimes were AFTER the lives... so the lives really kill this.... i can 
probably grab the link... 
✅ FIXED: Now detects live videos by checking URL and skips processing them entirely!
"""

import time
import logging
import sys
import traceback
import random
import webbrowser
import pyautogui

# Import configuration
from src import config

# Import DESKTOP interaction functions (Replaces mac_interaction)
from src.desktop_interaction import (
    move_mouse_smooth,
    click,
    press_down_arrow,
    copy_current_url,
    press_escape
)

# Import Screenshot functions (Now uses pyautogui)
from src.screenshot import capture_screenshot, detect_share_button

# Import Analysis function
from src.analysis import analyze_screenshot_content

# Import TikTok specific utility functions
from src.tiktok_utils import (
    is_live_video,
    save_live_video_url,
    safe_navigate_from_live
)

# ===== CONFIGURATION (LOADED FROM src/config.py) =====
# The constants previously defined here are now in src/config.py
# Example access: config.TIKTOK_URL, config.COMMENT_BUTTON_X, etc.
# ** REMEMBER TO RECALIBRATE ALL COORDINATES IN src/config.py FOR WINDOWS **

# Setup logging (moved from __main__)
log_format = '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# Suppress overly verbose logs from http connections if needed
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)

def scrape_tiktok():
    """Main function to scrape TikTok videos and comments"""
    initial_pos = None # Initialize initial_pos
    try:
        logging.info("\n🚀 Moon Dev's TikTok Alpha Scraper Starting (Windows Mode)...")
        logging.warning("🚨 ENSURE ALL COORDINATES IN src/config.py ARE CALIBRATED FOR YOUR WINDOWS SETUP! 🚨")

        # Store initial position using pyautogui
        try:
            initial_pos = pyautogui.position()
            logging.info(f"🖱️ Initial mouse position: {initial_pos}")
        except Exception as e:
            logging.warning(f"⚠️ Could not get initial mouse position: {e}")

        # Create screenshot directory (using config)
        config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        config.SHARE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"\n📁 Created screenshot directories: {config.SCREENSHOT_DIR} and {config.SHARE_DETECTION_DIR}")
        
        # Open TikTok in browser
        logging.info(f"\n🌐 Opening TikTok URL: {config.TIKTOK_URL}")
        webbrowser.open(config.TIKTOK_URL)
        
        # Wait for browser to load
        logging.info(f"\n⏳ Waiting {config.BROWSER_LOAD_WAIT} seconds for browser to load...")
        time.sleep(config.BROWSER_LOAD_WAIT)
        
        # IMPORTANT: Detect share button ONCE at the start to determine comment button position
        # This now uses the function from src.screenshot
        logging.info("\n🔍 Detecting share button position (one-time check)...")
        has_share_button = detect_share_button() # Imported from screenshot.py
        
        # Determine which comment button Y coordinate to use based on share button detection
        # Using coordinates from config
        comment_y = config.COMMENT_BUTTON_Y_WITH_SHARE if has_share_button else config.COMMENT_BUTTON_Y
        logging.info(f"\n🎯 Using comment button coordinates: ({config.COMMENT_BUTTON_X}, {comment_y}) for all videos")
        
        # Main scraping loop
        for video_number in range(1, config.MAX_VIDEOS + 1):
            try:
                logging.info(f"\n🎬 Processing TikTok video #{video_number}/{config.MAX_VIDEOS}...")

                # First get the current video URL using desktop_interaction version
                logging.info("\n🔗 Getting current video URL...")
                current_url = copy_current_url() # Imported from desktop_interaction.py
                logging.info(f"🔗 Current video URL: {current_url if current_url else 'Unknown'}")

                # Check if current video is LIVE using the utility function
                is_live, checked_url = is_live_video(current_url)
                
                # If it's a live video, skip processing and move to next video
                if is_live:
                    logging.info("\n⏭️ LIVE video detected - skipping processing and moving to next video")
                    
                    # Save the live video URL if available
                    if checked_url:
                        save_live_video_url(checked_url, video_number)
                    
                    # Use safe_navigate_from_live (which uses desktop_interaction functions)
                    if not safe_navigate_from_live(video_number):
                        logging.warning("⚠️ Failed to safely navigate from live video, trying fallback method...")

                        # Fallback: Try Escape first
                        try:
                            logging.info("⌨️ Pressing Escape key...")
                            press_escape() # Use desktop_interaction version
                        except Exception as e:
                            logging.warning(f"⚠️ Error pressing Escape: {e}")

                        # Then try a simple down arrow
                        press_down_arrow() # Use desktop_interaction version
                        time.sleep(config.SCROLL_PAUSE + 0.5)

                    # Check if we're still on TikTok after navigating
                    post_live_nav_url = copy_current_url() # Use desktop_interaction version
                    if post_live_nav_url and "tiktok.com" in post_live_nav_url:
                        logging.info("✅ Successfully stayed on TikTok after live video")
                    else:
                        logging.warning("⚠️ May have navigated away from TikTok - attempting to return")
                        webbrowser.open(config.TIKTOK_URL)
                        time.sleep(config.BROWSER_LOAD_WAIT + 1.0)
                    
                    # Skip the rest of the processing for this video
                    logging.info("⏩ Moving to next video...")
                    continue
                
                # REGULAR VIDEO PROCESSING

                # Move to comment button and click
                logging.info(f"\n🖱️ Moving to comment button ({config.COMMENT_BUTTON_X}, {comment_y})")
                if not move_mouse_smooth(config.COMMENT_BUTTON_X, comment_y, duration=config.MOVEMENT_SPEED):
                    logging.error("❌ Failed to move to comment button!")
                    continue
                    
                time.sleep(config.CLICK_PAUSE)
                # Use desktop_interaction click(x, y)
                if not click(config.COMMENT_BUTTON_X, comment_y):
                    logging.error("❌ Failed to click comment button!")
                    continue
                    
                # Wait for comments to load
                logging.info(f"\n⏳ Waiting {config.COMMENT_LOAD_WAIT} seconds for comments to load...")
                time.sleep(config.COMMENT_LOAD_WAIT)
                
                # 1. Take screenshot (uses updated screenshot.py)
                logging.info(f"\n📸 Taking screenshot of video #{video_number}...")
                url_for_analysis = checked_url if checked_url else current_url
                screenshot_path = capture_screenshot(video_number)
                if not screenshot_path:
                    logging.warning(f"⚠️ Failed to capture screenshot for video #{video_number}, continuing...")
                    # Don't necessarily skip, maybe analysis can still happen without screenshot?
                    # Or maybe skip depending on desired behavior.
                    # For now, let's log and continue, analysis will likely fail or be empty.

                else:
                     logging.info(f"✅ Screenshot saved for video #{video_number}")

                # 2. Analyze screenshot with AI
                logging.info(f"\n🧠 Analyzing screenshot content...")
                analysis_result = analyze_screenshot_content(screenshot_path, video_number, url_for_analysis)
                if analysis_result is None:
                    logging.warning(f"⚠️ Failed to analyze screenshot for video #{video_number}, continuing...")

                # 3. Activate browser and scroll to next video (if not the last one)
                if video_number < config.MAX_VIDEOS:
                    logging.info(f"\n🎮 Activating browser window with double-click...")
                    if not move_mouse_smooth(config.BROWSER_CLICK_X, config.BROWSER_CLICK_Y, duration=config.MOVEMENT_SPEED):
                        logging.error("❌ Failed to move to browser area!")
                        continue
                        
                    # Double click using desktop_interaction click
                    if not click(config.BROWSER_CLICK_X, config.BROWSER_CLICK_Y, clicks=2, interval=0.2):
                        logging.error("❌ Failed to double-click browser area!")
                        continue
                        
                    logging.info("✅ Browser window activated with double-click! 🖱️🖱️")
                    time.sleep(config.ACTIVATION_PAUSE)

                    # Now press down arrow (uses desktop_interaction)
                    logging.info(f"\n⬇️ Scrolling to next video...")
                    if not press_down_arrow():
                        logging.error("❌ Failed to press down arrow!")
                        continue
                    
                    # Wait for next video to load
                    logging.info(f"\n⏳ Waiting {config.SCROLL_PAUSE} seconds for next video...")
                    time.sleep(config.SCROLL_PAUSE)
                    
                    # Add randomness
                    random_wait = random.uniform(0.5, 2.0)
                    logging.info(f"⏳ Adding random delay: {random_wait:.2f}s")
                    time.sleep(random_wait)
                
            except Exception as e:
                logging.error(f"\n❌ Error processing video #{video_number}: {str(e)}")
                logging.error(traceback.format_exc())
                # Attempt to recover by scrolling down?
                try:
                    logging.warning("⚠️ Attempting recovery scroll...")
                    press_down_arrow()
                    time.sleep(config.SCROLL_PAUSE * 2)
                except Exception as scroll_e:
                    logging.error(f"❌ Recovery scroll failed: {scroll_e}")
                continue # Continue to next video

        logging.info("\n✅ TikTok scraping completed successfully!")
        logging.info(f"📊 Scraped {config.MAX_VIDEOS} TikTok videos (attempted)")
        logging.info(f"📁 Screenshots saved to: {config.SCREENSHOT_DIR}")
        logging.info(f"📊 Analysis saved to: {config.ANALYSIS_CSV}")
        logging.info("\n🌙 Moon Dev says: All the alpha has been collected! Time to find those trading opportunities! 💰")

        # Return to initial mouse position if captured
        if initial_pos:
            try:
                move_mouse_smooth(initial_pos.x, initial_pos.y, duration=0.5)
            except Exception as e:
                logging.warning(f"⚠️ Could not return mouse to initial position: {e}")
        
    except KeyboardInterrupt:
        logging.warning("\n👋 Scraping cancelled by user")
        if initial_pos:
            try:
                move_mouse_smooth(initial_pos.x, initial_pos.y, duration=0.5)
            except:
                pass # Ignore errors during cleanup
    except Exception as e:
        logging.error(f"\n❌ Fatal error in scrape_tiktok: {str(e)}")
        logging.error(traceback.format_exc())
        if initial_pos:
            try:
                move_mouse_smooth(initial_pos.x, initial_pos.y, duration=0.5)
            except:
                pass # Ignore errors during cleanup

# Update find_coordinates to use pyautogui
def find_coordinates():
    """Helper function to find screen coordinates using pyautogui"""
    try:
        logging.info("\n🔍 Starting coordinate finder (Windows Mode)...")
        logging.info("Move your mouse to the desired position and press Ctrl+C in the terminal to capture coordinates.")
        logging.info("Suggested positions to find and update in src/config.py:")
        logging.info("1. COMMENT_BUTTON_X, COMMENT_BUTTON_Y")
        logging.info("2. COMMENT_BUTTON_Y_WITH_SHARE")
        logging.info("3. SHARE_DETECT_CENTER_X, SHARE_DETECT_CENTER_Y")
        logging.info("4. BROWSER_CLICK_X, BROWSER_CLICK_Y")
        logging.info("5. LIVE_SAFE_CLICK_X, LIVE_SAFE_CLICK_Y")
        logging.info("6. URL_BAR_CLICK_X, URL_BAR_CLICK_Y")
        logging.info("7. SCREENSHOT_REGION_X, SCREENSHOT_REGION_Y (Top-left corner of screenshot area)")

        print("\nPress Ctrl+C to capture the current mouse coordinates...") # Use print for interactive part
        
        while True:
            # Get position using pyautogui
            x, y = pyautogui.position()

            # Use print for real-time feedback, clearing the line
            print(f"\r📍 Current position: ({x}, {y})     ", end="")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        # Capture final position on Ctrl+C
        x, y = pyautogui.position()
        print("\n") # Newline after Ctrl+C

        logging.info(f"\n✅ Captured coordinates: X={x}, Y={y}")
        logging.info("Update these coordinates in the src/config.py file.")
        logging.info("Example:")
        logging.info(f"SOME_X_COORDINATE = {x}")
        logging.info(f"SOME_Y_COORDINATE = {y}")

    except Exception as e:
        logging.error(f"\n❌ Error in coordinate finder: {e}")


if __name__ == "__main__":
    try:
        # Logging is configured near the top now

        # REMOVED macOS dependency check block
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "--find-coordinates":
            find_coordinates()
        else:
            # Run the main scraping function
            scrape_tiktok()
            
    except KeyboardInterrupt:
        logging.warning("\n👋 Execution cancelled by user (main block)")
    except Exception as e:
        logging.error(f"\n❌ Fatal error in main execution block: {str(e)}")
        logging.error(traceback.format_exc())



