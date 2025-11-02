import logging
import traceback
import base64
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import config and model factory
from . import config
from .models.model_factory import model_factory

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_screenshot_content(screenshot_path: str, video_number: int, video_url: str | None = None) -> str | None:
    """Analyze screenshot using an AI model to extract video content and comments, and save to CSV."""
    try:
        logging.info(f"\n🧠 Analyzing screenshot for video #{video_number} with {config.MODEL_NAME}...")
        logging.info(f"   Screenshot path: {screenshot_path}")
        logging.info(f"   Video URL: {video_url if video_url else 'N/A'}")

        # Check if file exists
        img_path = Path(screenshot_path)
        if not img_path.exists():
            logging.error(f"❌ Screenshot file not found: {screenshot_path}")
            raise FileNotFoundError(f"Screenshot file not found: {screenshot_path}")

        # Initialize AI model
        logging.info("🏭 Using Moon Dev's Model Factory singleton...")
        model = model_factory.get_model(config.MODEL_TYPE, config.MODEL_NAME)
        if not model:
            logging.error(f"❌ Failed to initialize {config.MODEL_TYPE} model '{config.MODEL_NAME}'. Check API keys and model availability.")
            raise Exception(f"Failed to initialize {config.MODEL_TYPE} model '{config.MODEL_NAME}'.")
        logging.info(f"✅ Successfully initialized {config.MODEL_TYPE} model: {config.MODEL_NAME}")

        # Encode image to base64
        logging.info("🔄 Encoding screenshot to base64...")
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare message with image for AI
        messages = [
            {"role": "system", "content": "You are an expert at analyzing TikTok videos and comments to extract trading insights."},
            {"role": "user", "content": [
                {"type": "text", "text": config.ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

        # Get response from AI model
        logging.info("🤖 Sending image to AI for analysis...")
        response = model.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=1000 # As per original script
        )

        if not response or not response.choices or not hasattr(response.choices[0].message, 'content') or not response.choices[0].message.content:
            logging.error("❌ Empty or invalid response from AI model during analysis.")
            raise Exception("Empty response from AI model")

        analysis_text = response.choices[0].message.content
        logging.info("✅ AI Analysis completed successfully.")

        # Clean up the analysis text for CSV
        logging.info("🧹 Cleaning up analysis text for CSV format...")
        cleaned_analysis = analysis_text.replace('\n', ' ') # Replace all newlines with a space
        cleaned_analysis = ' '.join(cleaned_analysis.split()) # Replace multiple spaces with a single space
        cleaned_analysis = cleaned_analysis.replace('"', '""') # Escape double quotes for CSV

        # Save analysis to CSV
        _save_analysis_to_csv(video_number, screenshot_path, cleaned_analysis, video_url)

        logging.info(f"🌙 Moon Dev says: Alpha extracted from video #{video_number}! 💸")
        return analysis_text # Return the original, uncleaned analysis text as before

    except FileNotFoundError:
        # Already logged, re-raise to be caught by the main loop if necessary
        raise
    except Exception as e:
        logging.error(f"❌ Error analyzing screenshot: {str(e)}")
        logging.error(f"📋 Full error details:\n{traceback.format_exc()}")
        return None

def _save_analysis_to_csv(video_number: int, screenshot_path: str, analysis: str, video_url: str | None):
    """Helper function to save the analysis data to a CSV file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_csv_path = Path(config.ANALYSIS_CSV)

        # Ensure the directory for the CSV exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df_columns = ["timestamp", "video_number", "screenshot_path", "analysis", "video_url"]

        # Create or load existing dataframe
        if output_csv_path.exists():
            try:
                df = pd.read_csv(output_csv_path)
                # Ensure loaded DataFrame has the correct columns
                if not all(col in df.columns for col in df_columns):
                    logging.warning(f"⚠️ CSV columns mismatch. Recreating CSV: {output_csv_path}")
                    df = pd.DataFrame(columns=df_columns)
            except Exception as e:
                logging.warning(f"❌ Error reading existing CSV ({output_csv_path}), creating new one: {e}")
                df = pd.DataFrame(columns=df_columns)
        else:
            df = pd.DataFrame(columns=df_columns)

        # Add new row
        new_row = {
            "timestamp": timestamp,
            "video_number": video_number,
            "screenshot_path": str(screenshot_path), # Ensure path is string
            "analysis": analysis, # Already cleaned
            "video_url": video_url if video_url else "N/A"
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Save to CSV with proper quoting
        df.to_csv(output_csv_path, index=False, quoting=1) # quoting=1 is QUOTE_ALL
        logging.info(f"📊 Analysis for video #{video_number} saved to CSV: {output_csv_path}")
        if video_url:
            logging.info(f"🔗 Video URL saved in CSV: {video_url}")

    except Exception as e:
        logging.error(f"❌ Error saving analysis to CSV: {e}")
        logging.error(f"📋 Full error details for CSV saving:\n{traceback.format_exc()}")
        # Optionally, try to save to a backup CSV as in the original script
        try:
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_csv_path.parent / f"{output_csv_path.stem}_backup_{backup_timestamp}.csv"
            df.to_csv(backup_path, index=False, quoting=1)
            logging.warning(f"📊 Backup analysis saved to: {backup_path}")
        except Exception as backup_e:
            logging.error(f"❌ Failed to save backup CSV as well: {backup_e}") 