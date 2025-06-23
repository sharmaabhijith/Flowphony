from typing import Dict, Any
from .helpers.base_agent import BaseAgent
from .helpers.config import Config

class HarmonyAgent(BaseAgent):
    """Harmony agent that adds harmonic elements to melodies"""
    
    def __init__(self, config: Config, agent_name: str = "harmony"):
        super().__init__(config, agent_name)
    
    def generate_response(self, message: str) -> str:
        """Generate a response using DeepInfra API"""
        # Prepare messages for API call
        messages = self.get_conversation_context()
        messages.append({"role": "user", "content": message})

        response = self.call_deepinfra_api(messages)
        return response
    
    def add_harmony_to_melody(self, melody: str) -> str:
        """Add harmonic elements to the given melody"""
        harmony_prompt = f"""
        Given the following melody in ABC notation:

        {melody}

        Please add appropriate harmonic elements to this melody:
        """
        return self.generate_response(harmony_prompt) 