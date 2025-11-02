import os
import logging
from typing import Dict, Optional
from openai import OpenAI, OpenAIError
from fastapi import HTTPException
from core.config import get_settings

# Configuration
CONFIG = {
    "MEMORY_FILE": "data/moongpt_memory.txt",
    "OPENAI_MODEL": "gpt-4o",
    "MEMORY_START_DELIMITER": "#### START MEMORY ####",
    "MEMORY_END_DELIMITER": "#### END MEMORY ####",
    "MAX_MEMORY_TOKENS": 4000
}

logger = logging.getLogger(__name__)

# OpenAI Client Initialization
settings = get_settings()
OPENAI_API_KEY = settings.OPENAI_API_KEY
if not OPENAI_API_KEY:
    logger.error("OpenAI API key not configured in settings")
    raise HTTPException(status_code=500, detail="OpenAI API key not configured")
logger.info("OpenAI API key loaded from configuration.")

client = OpenAI(api_key=OPENAI_API_KEY)

class MoonGPTService:
    def __init__(self):
        self.memory = self.read_file_content(CONFIG["MEMORY_FILE"])
        logger.info("MoonGPTService initialized")

    def read_file_content(self, filepath: str) -> str:
        """Safely reads content from a file, returning empty string if not found/error."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as file:
                    return file.read().strip()
            else:
                logger.info(f"Memory file '{filepath}' not found. Starting new conversation.")
                return ""
        except IOError as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return ""

    def write_file_content(self, filepath: str, content: str, mode: str = "w") -> bool:
        """Safely writes content to a file."""
        try:
            with open(filepath, mode, encoding="utf-8") as file:
                file.write(content)
            return True
        except IOError as e:
            logger.error(f"Error writing to file {filepath}: {e}")
            return False

    def build_full_prompt(self, memory: str, prompt: str, config: Dict) -> str:
        """Builds the full prompt including memory context."""
        if len(memory.split()) > config.get("MAX_MEMORY_TOKENS", 4000):
            logger.warning("Memory is long, trimming...")
            memory_lines = memory.splitlines()
            memory = "\n".join(memory_lines[-100:])
        return (
            f"{config['MEMORY_START_DELIMITER']}\n"
            f"{memory}\n"
            f"{config['MEMORY_END_DELIMITER']}\n\n"
            f"User Prompt:\n{prompt}"
        )

    def call_openai_model(self, prompt: str, config: Dict) -> Optional[str]:
        """Calls the specified OpenAI model with the given prompt."""
        logger.info(f"Sending request to OpenAI model: {config['OPENAI_MODEL']}")
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
            logger.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI call: {e}")
            return None

    def update_memory(self, prompt: str, response: str, current_memory: str, config: Dict) -> str:
        """Appends the latest interaction to the memory and saves it."""
        memory_update = f"\n\nUser: {prompt}\nAI: {response}"
        new_memory = current_memory + memory_update
        if self.write_file_content(config["MEMORY_FILE"], new_memory.strip()):
            logger.info(f"Conversation updated in {config['MEMORY_FILE']}")
        else:
            logger.warning(f"Failed to update memory file {config['MEMORY_FILE']}")
        return new_memory

    async def get_response(self, user_prompt: str) -> str:
        """Handles the user prompt and returns the AI response."""
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        full_prompt = self.build_full_prompt(self.memory, user_prompt, CONFIG)
        response = self.call_openai_model(full_prompt, CONFIG)
        if response:
            self.memory = self.update_memory(user_prompt, response, self.memory, CONFIG)
            return response
        else:
            raise HTTPException(status_code=500, detail="No valid response received from OpenAI")

moongpt_service = MoonGPTService()
