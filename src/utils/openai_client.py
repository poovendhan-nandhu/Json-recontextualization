"""OpenAI client wrapper with retry logic."""
import json
from typing import Any, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.config import config


class OpenAIClient:
    """OpenAI client with deterministic settings and retry logic."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        self.temperature = config.OPENAI_TEMPERATURE
        self.seed = config.OPENAI_SEED
        self.timeout = config.OPENAI_TIMEOUT
        
        self.total_tokens_used = 0
        self.total_calls = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Generate JSON response from OpenAI with deterministic settings.
        
        Args:
            system_prompt: System message for context
            user_prompt: User message with the task
            max_tokens: Maximum tokens for response (auto-calculated if None)
        
        Returns:
            Parsed JSON response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                seed=self.seed,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                timeout=self.timeout
            )
            
            # Update stats
            self.total_tokens_used += response.usage.total_tokens
            self.total_calls += 1
            
            # Parse and return JSON
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse OpenAI response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")
    
    def generate_json_streaming(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None
    ):
        """
        Generate JSON response from OpenAI with streaming support.
        
        Yields chunks of the response as they arrive.
        
        Args:
            system_prompt: System message for context
            user_prompt: User message with the task
            max_tokens: Maximum tokens for response
        
        Yields:
            String chunks of the streaming response
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                seed=self.seed,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                timeout=self.timeout,
                stream=True  # Enable streaming
            )
            
            full_content = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk
                    yield content_chunk
            
            # Update stats (we don't get token counts from streaming, estimate)
            self.total_tokens_used += len(full_content) // 4  # Rough estimate
            self.total_calls += 1
            
            # Return the full parsed JSON at the end
            yield {"__complete__": True, "data": json.loads(full_content)}
            
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming API call failed: {e}")
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_calls": self.total_calls,
            "model": self.model
        }
    
    async def test_connection(self) -> bool:
        """Test OpenAI API connectivity."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False


# Singleton instance
openai_client = OpenAIClient()
