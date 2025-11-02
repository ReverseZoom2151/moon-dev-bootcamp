import time
import logging
import traceback
import pyautogui
import pyperclip

# Import necessary constants from config
from . import config

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable pyautogui failsafe for cases where mouse needs to go to corner
pyautogui.FAILSAFE = False
# Consider adding pauses after actions to ensure reliability
pyautogui.PAUSE = 0.1 # Default pause after each pyautogui call

def move_mouse_smooth(x: int, y: int, duration: float | None = None):
    """Move mouse smoothly to target coordinates using pyautogui."""
    try:
        current_x, current_y = pyautogui.position()
        target_duration = duration if duration is not None else config.MOVEMENT_SPEED
        logging.info(f"\n📍 Moving mouse from ({current_x}, {current_y}) -> Target: ({x}, {y}) over {target_duration}s")
        pyautogui.moveTo(x, y, duration=target_duration)
        final_x, final_y = pyautogui.position()
        logging.info(f"📍 Final mouse position: ({final_x}, {final_y})")
        # Basic check, pyautogui might not be pixel perfect
        if abs(final_x - x) > 5 or abs(final_y - y) > 5:
             logging.warning(f"⚠️ Mouse position ({final_x}, {final_y}) slightly off target ({x}, {y})")
        return True
    except Exception as e:
        logging.error(f"❌ Error moving mouse: {e}\n{traceback.format_exc()}")
        return False

def click(x: int | None = None, y: int | None = None, button: str = 'left', clicks: int = 1, interval: float = 0.1):
    """Perform a mouse click at the current position or specified coordinates."""
    try:
        if x is not None and y is not None:
            logging.info(f"🖱️ Clicking {button} button {clicks}x at ({x}, {y})")
            pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
        else:
            current_x, current_y = pyautogui.position()
            logging.info(f"🖱️ Clicking {button} button {clicks}x at current position ({current_x}, {current_y})")
            pyautogui.click(button=button, clicks=clicks, interval=interval)
        logging.info(f"✅ Click successful ({clicks}x)")
        return True
    except Exception as e:
        logging.error(f"❌ Error clicking: {e}\n{traceback.format_exc()}")
        return False

def press_key(key_name: str):
    """Press a specific key."""
    try:
        logging.info(f"⌨️ Pressing key: {key_name}")
        pyautogui.press(key_name)
        logging.info(f"✅ Key '{key_name}' pressed successfully")
        return True
    except Exception as e:
        logging.error(f"❌ Error pressing key '{key_name}': {e}\n{traceback.format_exc()}")
        return False

def hotkey(*key_names):
    """Press a combination of keys simultaneously (e.g., ctrl, c)."""
    try:
        logging.info(f"⌨️ Pressing hotkey: {' + '.join(key_names)}")
        pyautogui.hotkey(*key_names)
        logging.info(f"✅ Hotkey {' + '.join(key_names)} pressed successfully")
        return True
    except Exception as e:
        logging.error(f"❌ Error pressing hotkey {' + '.join(key_names)}: {e}\n{traceback.format_exc()}")
        return False

def press_down_arrow():
    """Press down arrow key"""
    return press_key('down')

def press_escape():
    """Press Escape key"""
    return press_key('esc')

def copy_current_url() -> str | None:
    """Copy the current URL from the browser address bar using keyboard shortcuts."""
    # This relies on standard shortcuts (Ctrl+L, Ctrl+C) and clipboard access
    try:
        logging.info("\n🔗 Copying current URL from browser address bar...")

        # Clear clipboard first using pyperclip
        try:
            pyperclip.copy('')
            logging.info("Clipboard cleared.")
            time.sleep(0.2)
        except Exception as e:
            logging.warning(f"⚠️ Could not clear clipboard: {e}")

        # Move to and click the URL bar (using coordinates from config)
        logging.info(f"🖱️ Moving to URL address bar ({config.URL_BAR_CLICK_X}, {config.URL_BAR_CLICK_Y})")
        if not move_mouse_smooth(config.URL_BAR_CLICK_X, config.URL_BAR_CLICK_Y):
            logging.error("❌ Failed to move to URL address bar!")
            return None

        time.sleep(config.CLICK_PAUSE)
        if not click(): # Click at current location (the URL bar)
            logging.error("❌ Failed to click URL address bar!")
            return None

        time.sleep(0.5)  # Wait for URL bar to be active

        # Retry loop for copying
        for attempt in range(config.URL_COPY_RETRIES):
            logging.info(f"⌨️ URL copy attempt {attempt+1}/{config.URL_COPY_RETRIES}...")

            # Use Ctrl+L to select the URL bar content
            logging.info("⌨️ Using Ctrl+L to select URL bar...")
            if not hotkey('ctrl', 'l'):
                logging.warning("⚠️ Failed Ctrl+L hotkey")
                # You could add a fallback like trying to click again or Ctrl+A
                time.sleep(config.URL_COPY_RETRY_DELAY)
                continue

            time.sleep(0.3) # Pause after selection

            # Use Ctrl+C to copy
            logging.info("⌨️ Using Ctrl+C to copy URL...")
            if not hotkey('ctrl', 'c'):
                logging.error("❌ Failed Ctrl+C hotkey")
                time.sleep(config.URL_COPY_RETRY_DELAY)
                continue

            time.sleep(0.5) # Ensure clipboard has time to update

            # Get clipboard content using pyperclip
            try:
                clipboard_content = pyperclip.paste()
                logging.info(f"Clipboard content received (len: {len(clipboard_content)})")

                # Validate the URL
                if isinstance(clipboard_content, str) and clipboard_content.startswith("http") and "tiktok.com" in clipboard_content:
                    logging.info(f"✅ Successfully copied URL: {clipboard_content}")
                    logging.info(f"🌙 Moon Dev says: URL captured successfully! 🔗")
                    # Optionally move mouse away after success
                    # move_mouse_smooth(config.BROWSER_CLICK_X, config.BROWSER_CLICK_Y)
                    return clipboard_content
                else:
                    # Log carefully, clipboard might contain large non-URL data
                    log_content = clipboard_content[:100] + '...' if isinstance(clipboard_content, str) and len(clipboard_content) > 100 else clipboard_content
                    logging.warning(f"⚠️ Attempt {attempt+1}: Invalid URL or empty clipboard: '{log_content}'")
                    pyperclip.copy('') # Clear clipboard again before retry
                    time.sleep(config.URL_COPY_RETRY_DELAY)
            except Exception as e:
                 # pyperclip can sometimes raise errors on certain clipboard contents
                 logging.error(f"❌ Error pasting/validating clipboard content: {e}\n{traceback.format_exc()}")
                 pyperclip.copy('') # Attempt to clear clipboard
                 time.sleep(config.URL_COPY_RETRY_DELAY)

        logging.error(f"❌ Failed to copy a valid URL after {config.URL_COPY_RETRIES} attempts")
        return None

    except Exception as e:
        logging.error(f"❌ Error copying URL: {str(e)}\n{traceback.format_exc()}")
        return None 