import time
import logging
import traceback
import base64
from datetime import datetime
from pathlib import Path
import pyautogui

# Import config and model factory
from . import config
from .models.model_factory import model_factory

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Refactor _capture_screen_region to use pyautogui
def _capture_screen_region(region: tuple[int, int, int, int], save_path: Path):
    """Helper function to capture a specific screen region using pyautogui and save it to a file."""
    try:
        # Pyautogui region is (left, top, width, height)
        left, top, width, height = region
        logging.info(f"📷 Attempting to capture screen region: left={left}, top={top}, width={width}, height={height} -> {save_path}")

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Capture the screenshot
        screenshot = pyautogui.screenshot(region=region)

        if screenshot is None:
            raise Exception("Failed to capture screenshot - pyautogui.screenshot returned None")

        # Check dimensions (Pillow Image object)
        img_width, img_height = screenshot.size
        if img_width == 0 or img_height == 0:
             raise ValueError(f"Invalid screenshot dimensions: {img_width}x{img_height}")
        logging.info(f"📏 Screenshot dimensions: {img_width}x{img_height}")

        # Save the image
        screenshot.save(str(save_path))

        if not save_path.exists():
            raise Exception(f"Screenshot file was not created: {save_path}")

        file_size = save_path.stat().st_size
        if file_size == 0:
            # Optional: Add a small delay and retry saving? Pyautogui saving might occasionally be async/take time.
            time.sleep(0.2)
            file_size = save_path.stat().st_size
            if file_size == 0:
                raise Exception(f"Screenshot file is empty: {save_path}")

        logging.info(f"✨ Screenshot saved successfully: {save_path} ({file_size} bytes)")
        return True

    except Exception as e:
        logging.error(f"❌ Error capturing/saving screen region with pyautogui: {e}\n{traceback.format_exc()}")
        return False

def capture_screenshot(video_number: int) -> str | None:
    """Capture screenshot of the main TikTok video and comments area."""
    for attempt in range(config.MAX_SCREENSHOT_RETRIES):
        try:
            logging.info(f"\n📸 Main screenshot attempt {attempt + 1}/{config.MAX_SCREENSHOT_RETRIES} for video #{video_number}...")

            # Ensure screenshot directory exists
            config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = config.SCREENSHOT_DIR / f"tiktok_video_{video_number:03d}_{timestamp}.png"

            logging.info(f"📝 Screenshot will be saved to: {screenshot_path}")

            # Define capture region from config (Needs adjustment for Windows!)
            # **IMPORTANT**: These coordinates are likely macOS-specific (negative Y often means from bottom).
            # They MUST be recalibrated on the target Windows machine.
            # Pyautogui expects (left, top, width, height). Assuming config provides these directly for now.
            x = config.SCREENSHOT_REGION_X
            y = config.SCREENSHOT_REGION_Y # This 'y' might need adjustment depending on how macOS coords were measured.
            width = config.SCREENSHOT_REGION_WIDTH
            height = config.SCREENSHOT_REGION_HEIGHT

            logging.warning(f"🎯 Using configured region: x={x}, y={y}, width={width}, height={height}. Ensure these are correct for Windows!")
            if not all(isinstance(val, (int, float)) for val in [x, y, width, height]):
                 raise ValueError("Invalid main screenshot coordinates in config (must be numbers)")

            # Ensure width and height are positive
            if width <= 0 or height <= 0:
                raise ValueError(f"Screenshot region width and height must be positive. Got: width={width}, height={height}")

            # Construct region tuple for pyautogui: (left, top, width, height)
            # Assuming x, y from config are top-left corner for now.
            region = (int(x), int(y), int(width), int(height))

            # Capture and save using the helper function
            if _capture_screen_region(region, screenshot_path):
                logging.info(f"🌙 Moon Dev says: Alpha secured from TikTok video #{video_number}! 💰")
                return str(screenshot_path)
            else:
                # Error logged in helper, raise exception to trigger retry
                raise Exception(f"_capture_screen_region failed for {screenshot_path}")

        except Exception as e:
            logging.error(f"❌ Error capturing main screenshot (attempt {attempt + 1}): {str(e)}")
            # Log full traceback only on the last attempt or if it's not the helper failure
            if attempt == config.MAX_SCREENSHOT_RETRIES - 1 or "_capture_screen_region failed" not in str(e):
                 logging.error(f"📋 Full error details:\n{traceback.format_exc()}")

            if attempt < config.MAX_SCREENSHOT_RETRIES - 1:
                logging.warning(f"😴 Waiting {config.SCREENSHOT_RETRY_DELAY} seconds before retry...")
                time.sleep(config.SCREENSHOT_RETRY_DELAY)
            else:
                logging.error("❌ All main screenshot attempts failed!")
                return None
    return None # Should not be reached, but added for safety

def detect_share_button() -> bool:
    """Take a small screenshot of the sound/share area and use AI to detect if it's a share button."""
    try:
        logging.info("\n🔍 Detecting if share button is present...")

        # Ensure share detection directory exists
        config.SHARE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = config.SHARE_DETECTION_DIR / f"share_detection_{timestamp}.png"

        # Define the region around the detection center point (Needs adjustment for Windows!)
        # **IMPORTANT**: Coordinates must be recalibrated. Assuming config provides center point (X,Y)
        # and detection size (Width, Height). Pyautogui needs top-left (left, top).
        center_x = config.SHARE_DETECT_CENTER_X
        center_y = config.SHARE_DETECT_CENTER_Y
        region_width = config.SHARE_DETECT_WIDTH
        region_height = config.SHARE_DETECT_HEIGHT

        # Calculate top-left coordinates for pyautogui region
        region_x = center_x - (region_width // 2)
        region_y = center_y - (region_height // 2) # Assuming Y increases downwards (Windows standard)

        logging.warning(f"🎯 Using detection region around ({center_x}, {center_y}) - Size: {region_width}x{region_height}. Ensure these are correct for Windows!")
        logging.info(f"📸 Capturing small area at ({region_x}, {region_y}) - Size: {region_width}x{region_height} -> {screenshot_path}")

        if region_width <= 0 or region_height <= 0:
            raise ValueError(f"Share detection region width and height must be positive. Got: width={region_width}, height={region_height}")


        # Construct region tuple for pyautogui: (left, top, width, height)
        region = (int(region_x), int(region_y), int(region_width), int(region_height))


        # Capture the small region
        if not _capture_screen_region(region, screenshot_path):
            logging.error("❌ Failed to capture screenshot for share button detection.")
            return False # Default to False if capture fails

        # Use AI to analyze the screenshot
        logging.info("🧠 Analyzing screenshot to detect share button...")

        # Initialize model
        model = model_factory.get_model(config.MODEL_TYPE, config.MODEL_NAME)
        if not model:
            raise Exception(f"Failed to initialize {config.MODEL_TYPE} model for share detection")

        # Encode image to base64
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare message with image
        messages = [
            {"role": "system", "content": "You are an expert at analyzing TikTok interface elements. Respond with ONLY 'true' or 'false'."},
            {"role": "user", "content": [
                {"type": "text", "text": config.SHARE_BUTTON_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

        # Get response from model
        response = model.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=10 # Reduced max_tokens for simple true/false
        )

        if not response or not response.choices or not hasattr(response.choices[0].message, 'content') or not response.choices[0].message.content:
            raise Exception("Empty or invalid response from AI model for share detection")

        # Extract result
        result_text = response.choices[0].message.content.strip().lower()
        logging.info(f"🔍 AI response for share detection: '{result_text}'")

        # Parse the result strictly
        has_share_button = result_text == "true"

        # Print result with colored background (using standard print for terminal colors)
        print("\n")
        if has_share_button:
            print("\033[42m\033[30m" + "  SHARE BUTTON DETECTED: TRUE   " + "\033[0m")
        else:
            print("\033[43m\033[30m" + "  NO SHARE BUTTON DETECTED: FALSE " + "\033[0m")
        print("\n")

        logging.info(f"🌙 Moon Dev says: {'Share button detected! Using alternate comment position.' if has_share_button else 'No share button found. Using standard comment position.'} 🔍")

        return has_share_button

    except Exception as e:
        logging.error(f"❌ Error detecting share button: {str(e)}\n{traceback.format_exc()}")
        # Default to False if detection fails
        return False 