# TikTok Agent - Windows Setup Guide

This guide will help you set up and run the TikTok Agent on Windows properly.

## Prerequisites

- Python 3.8+
- pip package manager
- Browser (Chrome, Edge, or Firefox) 
- Internet connection

## Setup Instructions

### 1. Install Dependencies

First, install all required packages with specific versions:

```bash
cd "Day_42_Projects"
pip install -r requirements.txt --upgrade
```

### 2. Configure Screen Coordinates (CRITICAL)

The TikTok agent relies on screen coordinates to interact with the browser. The default coordinates in `src/config.py` are placeholders and **must** be calibrated for your specific screen setup.

To calibrate:

1. Open a TikTok browser window, navigate to TikTok and position it where you want it to be during automation

2. Run the coordinate finder tool:
   ```bash
   cd "Day_42_Projects"
   python tiktok_agent.py --find-coordinates
   ```

3. Move your mouse to each of the following positions and press Ctrl+C in the terminal to capture the coordinates:
   - **COMMENT_BUTTON_X, COMMENT_BUTTON_Y**: Position of the comment button on a TikTok video
   - **COMMENT_BUTTON_Y_WITH_SHARE**: Position of comment button when share button is present (slightly lower)
   - **SHARE_DETECT_CENTER_X, SHARE_DETECT_CENTER_Y**: Position of share button
   - **BROWSER_CLICK_X, BROWSER_CLICK_Y**: Neutral area in browser window for clicking
   - **LIVE_SAFE_CLICK_X, LIVE_SAFE_CLICK_Y**: Safe area to click on live videos
   - **URL_BAR_CLICK_X, URL_BAR_CLICK_Y**: Position of browser's URL address bar
   - **SCREENSHOT_REGION_X, SCREENSHOT_REGION_Y**: Top-left corner of the area to screenshot

4. For each position, after pressing Ctrl+C, make note of the reported X and Y coordinates.

5. Edit `src/config.py` and update all coordinate values with the ones you captured.

6. For SCREENSHOT_REGION_WIDTH and SCREENSHOT_REGION_HEIGHT, measure approximately how wide and tall you want the screenshots to be (in pixels).

### 3. Test the Agent

After installing dependencies and configuring coordinates:

1. Open a web browser
2. Run the agent:
   ```bash
   cd "Day_42_Projects"
   python tiktok_agent.py
   ```

3. The agent will:
   - Open TikTok in your default browser
   - Begin navigating and capturing screenshots of videos
   - Store analysis in `Day_42_Projects/src/data/tiktok_agent/tiktok_analysis.csv`

### 4. Troubleshooting

- **Mouse Movement Issues**: If the mouse doesn't move to the right positions, recheck your coordinates in `src/config.py`
- **URL Capture Issues**: Ensure browser window has focus when the script is running
- **Model Errors**: Check the `.env` file to ensure API keys are correct
- **Empty Screenshots**: Verify the screenshot region coordinates in `config.py`

### 5. For Ollama Users

If you want to use Ollama models:
- Start the Ollama server in a separate terminal: `ollama serve`
- Then run the TikTok agent

## Key Files

- `tiktok_agent.py`: Main script
- `src/config.py`: Configuration including coordinates (edit this!)
- `src/data/tiktok_agent/`: Output directory for screenshots and analysis
- `.env`: Environment file for API keys

## Important Notes

- Ensure your browser window has focus when the script is running
- To stop the script at any time, press Ctrl+C in the terminal
- The agent will detect and skip live videos automatically 