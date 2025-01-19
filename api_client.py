"""
API Client module for Multilingual HTML Translator
Original site: https://hitdarderut-haaretz.org
Translated versions: https://degeneration-of-nation.org

Handles interactions with Claude API, including message formatting,
error handling, and response processing
"""
import re
from utils import truncate_str
import time
import logging
from typing import Dict, Any, Optional
from anthropic import Anthropic, RateLimitError, BadRequestError
from config import API_CONFIG

logger = logging.getLogger('website_translator')

class TranslationAPIClient:
    def __init__(self, api_key: str):
        """Initialize API client with provided key"""
        self.client = Anthropic(api_key=api_key)
        
    def create_translation_message(self, 
                                 system_text: str,
                                 prompt: str,
                                 assistant_answer: str,
                                 max_tokens: int,
                                 temperature: float,
                                 use_cache: bool = True,
                                 new_model: bool = True) -> Dict[str, Any]:
        """
        Creates and sends translation request to Claude API
        Handles both new and old model formats
        """
        try:
            # Prepare conversation messages
            messages = [
                {"role": "user", 
                 "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", 
                 "content": [{"type": "text", "text": assistant_answer}]}
            ]
            
            # Prepare system message
            system_message = {"type": "text", "text": system_text}
            if use_cache:
                system_message["cache_control"] = {"type": "ephemeral"}
                
            # Send API request
            if logger.getEffectiveLevel() <= logging.INFO:
                logger.info(f"System text size: {len(system_text)} "
                          f"Temp={temperature} Max_tokens={max_tokens}")
                logger.info(f"Conversation: {truncate_log_message(str(messages))}")
            
            message = self.client.messages.create(
                model=API_CONFIG['model'] if new_model else API_CONFIG['old_model'],
                max_tokens=max_tokens,
                temperature=temperature,
                system=[system_message],
                messages=messages,
                stop_sequences=["</body>"],
                extra_headers=API_CONFIG['headers']
            )
            
            return message
            
        except (RateLimitError, BadRequestError) as e:
            logger.error(f"API Error: {str(e)}")
            raise RuntimeError("!Stop!") from e
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            raise

    def process_response(self, message: Dict[str, Any], 
                        previous_context: str = "") -> tuple[str, bool]:
        """
        Processes API response, handling errors and validation
        Returns processed response and success status
        """
        if not message.content or not message.content[0].text:
            return "", False
            
        answer = message.content[0].text.rstrip().lstrip('\n')
        
        # Remove trailing comments if any
        if (match := re.search(r'\s*(?:\[[^\]]+\][\s]*)+$', answer)):
            logger.error("\n\n[!COMMENT!]: ..." + 
                        answer[max(0, match.start() - 50):])
            answer = answer[:match.start()].rstrip()
            
        # Add context if needed
        if previous_context:
            answer = previous_context + answer
            
        return answer, True

def truncate_log_message(message: str, max_length: int = 200) -> str:
    """Truncates log messages to reasonable length"""
    if len(message) <= max_length:
        return message
    return message[:max_length] + "..."