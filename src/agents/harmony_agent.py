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
    
    def add_harmony_to_melody(self, melody: str, prompt_data: dict = None) -> str:
        """Add harmonic elements to the given melody"""
        # Extract relevant information from prompt_data if available
        key_info = ""
        chord_progression_info = ""
        genre_info = ""
        
        if prompt_data["key"]:
            key_info = f"Key: {prompt_data.get('key', 'Your choice')}\n"
        if prompt_data["chord_progression"]:
            chord_progression_info = f"Chord Progression: {prompt_data.get('chord_progression', 'Your choice')}\n"
        if prompt_data["genre"]:
            genre_info = f"Genre: {prompt_data.get('genre', 'Your choice')}\n"
        
        harmony_prompt = f"""
        Given the following melody in ABC notation:

        {melody}

        {key_info}{chord_progression_info}{genre_info}
        
        Please add appropriate harmonic elements to this melody, taking into account the specified key, chord progression, and genre style:
        """
        return self.generate_response(harmony_prompt) 