#!/usr/bin/env python3
"""
Interactive script to chat with an OpenAI model, using conversation memory
stored in a local text file.
"""

import os
from openai import OpenAI, OpenAIError
from typing import Dict, Optional
import sys # For exiting

# --- Configuration ---
CONFIG = {
    # Use local files for memory if specific paths aren't needed
    "MEMORY_FILE": "memory.txt",
    "OPENAI_MODEL": "gpt-4o",
    "MEMORY_START_DELIMITER": "#### START MEMORY ####",
    "MEMORY_END_DELIMITER": "#### END MEMORY ####",
    "MAX_MEMORY_TOKENS": 4000 # Optional: Add a limit to prevent overly long memory
}

# --- OpenAI Client Initialization ---
try:
    import dontshare as d
    if not hasattr(d, 'openai_key') or not d.openai_key:
        raise ImportError("Variable 'openai_key' not found or empty in dontshare.py")
    OPENAI_API_KEY = d.openai_key
    print("OpenAI API key loaded.")
except ImportError as e:
    print(f"Error loading OpenAI API key: {e}")
    print("Please ensure dontshare.py exists and contains 'openai_key'.")
    sys.exit(1) # Use sys.exit

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Core Functions ---

def read_file_content(filepath: str) -> str:
    """Safely reads content from a file, returning empty string if not found/error."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read().strip()
        else:
             print(f"Note: Memory file '{filepath}' not found. Starting new conversation.")
             return "" # Return empty string, not None
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return "" # Return empty string on error

def write_file_content(filepath: str, content: str, mode: str = "w") -> bool:
    """Safely writes content to a file."""
    try:
        with open(filepath, mode, encoding="utf-8") as file:
            file.write(content)
        return True
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")
        return False

def get_prompt_from_console() -> str:
    """Gets the user's prompt from console input."""
    print("\nEnter your prompt (press Ctrl+D or Ctrl+Z then Enter to send):")
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

def build_full_prompt(memory: str, prompt: str, config: Dict) -> str:
    """Builds the full prompt including memory context."""
    # Optional: Trim memory if it gets too long (basic token estimate)
    # A more accurate method would use a tokenizer like tiktoken
    if len(memory.split()) > config.get("MAX_MEMORY_TOKENS", 4000):
         print("Warning: Memory is long, trimming...")
         # Simple trim - keep the end of the memory
         memory_lines = memory.splitlines()
         memory = "\n".join(memory_lines[-100:]) # Keep last ~100 lines as approximation

    return (
        f"{config['MEMORY_START_DELIMITER']}\n"
        f"{memory}\n"
        f"{config['MEMORY_END_DELIMITER']}\n\n"
        f"User Prompt:\n{prompt}"
    )

def call_openai_model(prompt: str, config: Dict) -> Optional[str]:
    """Calls the specified OpenAI model with the given prompt."""
    print(f"\nSending request to OpenAI model: {config['OPENAI_MODEL']}...")
    try:
        response = client.chat.completions.create(
            model=config["OPENAI_MODEL"],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        message_content = response.choices[0].message.content
        return message_content.strip() if message_content else None
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI call: {e}")
        return None

def update_memory(prompt: str, response: str, current_memory: str, config: Dict) -> str:
    """Appends the latest interaction to the memory and saves it."""
    memory_update = f"\n\nUser: {prompt}\nAI: {response}"
    new_memory = current_memory + memory_update

    if write_file_content(config["MEMORY_FILE"], new_memory.strip()): # Save trimmed memory
         print(f"Conversation updated in {config['MEMORY_FILE']}")
    else:
         print(f"Warning: Failed to update memory file {config['MEMORY_FILE']}")

    return new_memory # Return the updated memory string

# --- Main Execution ---

def main() -> None:
    """Main function to run the interactive chat loop."""
    print("--- Starting Interactive MoonGPT ---")
    print(f"Using model: {CONFIG['OPENAI_MODEL']}")
    print(f"Memory file: {CONFIG['MEMORY_FILE']}")
    print("(Type 'exit' or 'quit' to end the conversation)")

    # Load initial memory
    current_memory = read_file_content(CONFIG["MEMORY_FILE"])

    while True: # Loop for continuous conversation
        # Get prompt from user
        user_prompt = get_prompt_from_console()
        
        # Check for exit command
        if not user_prompt or user_prompt.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break # Exit the loop

        # Build the full prompt for the API call
        full_prompt_for_api = build_full_prompt(current_memory, user_prompt, CONFIG)

        # Call OpenAI model
        api_response = call_openai_model(full_prompt_for_api, CONFIG)

        # Process and display response, update memory
        if api_response:
            print("\n--- AI Response ---")
            print(api_response)
            print("-------------------\n")
            # Update the *in-memory* variable and save to file
            current_memory = update_memory(user_prompt, api_response, current_memory, CONFIG)
        else:
            print("\n--- No valid response received from OpenAI. Please try again. ---")
            # Optionally, decide if you want to retry or just prompt again
            # Memory is not updated if API fails

    print("\n--- Interaction Finished --- GGs") # Adjusted finished message

if __name__ == "__main__":
    main()
