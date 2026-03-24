"""LLM clients for API providers."""
import requests
import json
from pathlib import Path
from typing import Optional


def load_api_key(key_name: str, keys_file: str = None) -> str:
    """Load API key from JSON file.
    
    Args:
        key_name: Name of the key (e.g., 'intern_s1', 'openai')
        keys_file: Path to keys file. If None, uses API/api_keys.json
        
    Returns:
        API key string
    """
    if keys_file is None:
        keys_file = Path(__file__).parent / "api_keys.json"
    
    with open(keys_file, 'r', encoding='utf-8') as f:
        keys = json.load(f)
    
    return keys.get(key_name, "")


class InternS1Client:
    """Intern S1 API client."""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://chat.intern-ai.org.cn/api/v1/",
        model: str = "intern-s1",
        thinking_mode: bool = False
    ):
        """Initialize Intern S1 client.
        
        Args:
            api_key: API key. If None, loads from API/api_keys.json
            base_url: API base URL
            model: Model name
            thinking_mode: Whether to enable thinking mode
        """
        if api_key is None:
            api_key = load_api_key("intern_s1")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.thinking_mode = thinking_mode
        self.url = f"{self.base_url}/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    
    def generate(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048
    ) -> str:
        """Generate text from the model.
        
        Args:
            prompt: Input prompt
            seed: Random seed
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking_mode": self.thinking_mode,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        if seed is not None:
            data["seed"] = seed
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=300 if not self.thinking_mode else 1200
            )
            
            if response.status_code != 200:
                raise Exception(
                    f"API call failed: {response.status_code}, {response.text}"
                )
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse response: {e}")


def create_llm_client(llm_config):
    """Create LLM client from configuration.
    
    Args:
        llm_config: LLM config object (SimpleNamespace) with attributes:
            - llm_type: "intern_s1"
            - base_url: API base URL
            - model: Model name
            - thinking_mode: Whether to enable thinking mode
        
    Returns:
        LLM client instance with generate() method
    """
    if llm_config.llm_type == "intern_s1":
        return InternS1Client(
            api_key=None,  # Will auto-load from API/api_keys.json
            base_url=llm_config.base_url,
            model=llm_config.model,
            thinking_mode=llm_config.thinking_mode
        )
    else:
        raise ValueError(f"Unknown llm_type: {llm_config.llm_type}")
