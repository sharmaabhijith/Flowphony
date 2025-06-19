import os
import json
import aiohttp
from typing import Any, Dict, Optional
from src.config.config import Config

class DeepInfraClient:
    def __init__(self, config: Config):
        self.api_key = os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPINFRA_API_KEY environment variable not set")
        
        self.base_url = "https://api.deepinfra.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "meta-llama/Llama-2-70b-chat-hf",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """Generate text using DeepInfra's API"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            if stop:
                payload["stop"] = stop
            
            async with session.post(
                f"{self.base_url}/openai/completions",
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepInfra API error: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["text"].strip()
    
    async def generate_structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        model: str = "meta-llama/Llama-2-70b-chat-hf",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate structured output following a schema"""
        schema_prompt = f"""
        Generate a response following this JSON schema:
        {json.dumps(output_schema, indent=2)}
        
        Prompt: {prompt}
        
        Response (in JSON format):
        """
        
        response = await self.generate_text(
            prompt=schema_prompt,
            model=model,
            temperature=temperature,
            stop=["\n\n", "Human:", "Assistant:"]
        )
        
        try:
            # Extract JSON from response
            json_str = response[response.find("{"):response.rfind("}")+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from response: {response}") 