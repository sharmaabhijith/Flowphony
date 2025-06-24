from typing import Dict, Any
from .helpers.base_agent import BaseAgent
from .helpers.config import Config

class InstrumentAgent(BaseAgent):
    """Instrument agent that adds instrument elements to melodies"""
    
    def __init__(self, config: Config, agent_name: str = "instrument"):
        super().__init__(config, agent_name)
    
    def generate_response(self, message: str) -> str:
        """Generate a response using DeepInfra API"""
        # Prepare messages for API call
        messages = self.get_conversation_context()
        messages.append({"role": "user", "content": message})

        response = self.call_deepinfra_api(messages)
        return response
    
    def add_instrument_to_harmonic_melody(self, harmonic_melody: str, prompt_data: dict = None) -> str:
        """Add instrument elements to the given harmonic melody"""
        # Extract instrument information from prompt_data if available
        instruments_info = ""
        genre_info = ""
        
        if prompt_data:
            instruments_info = f"Specified Instruments: {prompt_data.get('instruments', 'Piano')}\n"
            genre_info = f"Genre: {prompt_data.get('genre', 'General')}\n"
        
        instrument_prompt = f"""
        Given the following harmonic melody in ABC notation:

        {harmonic_melody}

        {instruments_info}{genre_info}
        
        Please add appropriate instrument elements to this melody, using the specified instruments and considering the genre style:
        """
        return self.generate_response(instrument_prompt) 