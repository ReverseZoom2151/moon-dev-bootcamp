import logging
import time
import random
import traceback
from datetime import datetime

# Import config and interaction functions
from . import config
# Update imports to use desktop_interaction and remove Quartz
from .desktop_interaction import (
    copy_current_url, 
    move_mouse_smooth, 
    click, 
    press_down_arrow, 
    press_escape
)
# from Quartz import CoreGraphics as CG # Removed macOS dependency

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_live_video(url: str | None = None) -> tuple[bool, str | None]:
    """Check if the video is LIVE based on URL. Returns (is_live, checked_url)."""
    try:
        logging.info("\n🔍 Checking if current video is a LIVE video...")

        # Use provided URL or copy it if not provided (using imported desktop_interaction version)
        checked_url = url
        if not checked_url:
            logging.info("🔗 No URL provided, attempting to copy from browser...")
            checked_url = copy_current_url() # Uses desktop_interaction version

        if not checked_url:
            logging.warning("⚠️ Failed to get URL, assuming not a live video")
            return False, None

        # Check if URL contains live indicators
        live_indicators = ["/live", "live_room_mode"]
        is_live = any(indicator in checked_url.lower() for indicator in live_indicators)

        # Print result with colored background
        print("\n" + "=" * 60)
        if is_live:
            print("\033[41m\033[97m" + "  🔴 LIVE VIDEO DETECTED! SKIPPING PROCESSING  ".center(56) + "\033[0m")
            logging.info(f"🔗 Live URL detected: {checked_url}")
        else:
            print("\033[42m\033[30m" + "  ✅ REGULAR VIDEO DETECTED - PROCESSING  ".center(56) + "\033[0m")
            logging.info(f"🔗 Regular video URL: {checked_url}")
        print("=" * 60 + "\n")

        logging.info(f"🌙 Moon Dev says: {'Live video detected! Skipping processing.' if is_live else 'Regular video detected. Will process this one!'} 🔍")

        return is_live, checked_url # Return both the boolean and the URL used for the check

    except Exception as e:
        logging.error(f"❌ Error checking if video is live: {str(e)}\n{traceback.format_exc()}")
        # Default to False if detection fails
        return False, url # Return original URL if provided, else None

def save_live_video_url(url: str, video_number: int) -> bool:
    """Save the live video URL to a file."""
    # This function seems platform-independent
    try:
        if not url:
            logging.warning("⚠️ No URL provided to save for live video")
            return False

        # Ensure directory exists
        config.LIVE_VIDEOS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append to file
        entry = f"{timestamp} | Video #{video_number} | {url}\n"
        with open(config.LIVE_VIDEOS_FILE, "a", encoding='utf-8') as f:
            f.write(entry)

        logging.info(f"✅ Saved live video URL to {config.LIVE_VIDEOS_FILE}")
        return True

    except Exception as e:
        logging.error(f"❌ Error saving live video URL: {str(e)}\n{traceback.format_exc()}")
        return False

def safe_navigate_from_live(video_number: int) -> bool:
    """Safely navigate away from a live video to the next video using desktop_interaction."""
    try:
        logging.info("\n🛡️ Using safe navigation for live video...")

        # Move to the specific safe coordinates for live videos using desktop_interaction
        logging.info(f"🖱️ Moving to safe area for live videos ({config.LIVE_SAFE_CLICK_X}, {config.LIVE_SAFE_CLICK_Y})")
        if not move_mouse_smooth(config.LIVE_SAFE_CLICK_X, config.LIVE_SAFE_CLICK_Y, duration=config.MOVEMENT_SPEED):
            logging.warning("⚠️ Failed to move to primary safe area, trying fallback position")
            # Fallback position (far from edges)
            fallback_x = config.BROWSER_CLICK_X - 200 # Assuming BROWSER_CLICK_X/Y are defined in config
            fallback_y = config.BROWSER_CLICK_Y + 200
            if not move_mouse_smooth(fallback_x, fallback_y, duration=config.MOVEMENT_SPEED):
                logging.error("❌ Failed to move mouse to safe or fallback position!")
                return False

        # IMPORTANT: Perform a SINGLE gentle click using desktop_interaction
        logging.info("🖱️ Performing a SINGLE gentle click to activate browser")
        time.sleep(0.7)  # Longer pause before clicking

        # Use the click function from desktop_interaction (clicks at current mouse position)
        if not click(clicks=1):
            logging.error("❌ Failed to perform safe click!")
            # Maybe add a retry here if needed?
            return False

        time.sleep(1.0) # Longer pause after click
        logging.info("✅ Browser activated with single click")

        # Press down arrow to move to next video (using desktop_interaction version)
        logging.info("⬇️ Pressing down arrow to navigate to next video...")
        if not press_down_arrow():
            logging.error("❌ Failed to press down arrow! Retrying...")
            time.sleep(0.5)
            if not press_down_arrow():
                logging.error("❌ Failed to press down arrow on second attempt!")
                # As a last resort, try pressing Escape first
                logging.info("Trying Escape key before final down arrow attempt...")
                press_escape() # Uses desktop_interaction version
                time.sleep(0.5)
                if not press_down_arrow():
                     logging.error("❌ Failed all down arrow attempts after safe click!")
                     return False

        # Wait for next video to load (longer pause for live)
        wait_time = config.SCROLL_PAUSE + 1.0 + random.uniform(0.5, 1.5) # Add randomness
        logging.info(f"⏳ Waiting {wait_time:.2f} seconds for next video to load...")
        time.sleep(wait_time)

        logging.info(f"🌙 Moon Dev says: Safely navigated away from live video #{video_number}! 🚀")
        return True

    except Exception as e:
        logging.error(f"❌ Error navigating from live video: {str(e)}\n{traceback.format_exc()}")
        return False 