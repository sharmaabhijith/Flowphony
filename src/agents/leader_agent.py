from typing import Dict, Any
from .helpers.base_agent import BaseAgent
from .helpers.config import Config
class LeaderAgent(BaseAgent):
    """Leader agent that coordinates the music production process"""
    
    def __init__(self, config: Config, agent_name: str = "leader"):
        super().__init__(config, agent_name)
    
    def generate_response(self, message: str) -> str:
        """Generate a response using DeepInfra API"""
        # Prepare messages for API call
        messages = self.get_conversation_context()
        messages.append({"role": "user", "content": message})
        
        # Call DeepInfra API
        response = self.call_deepinfra_api(messages)
        return response
    
    def create_melody_prompt(self, prompt_data: dict) -> str:
        """Create a specific prompt for melody generation based on client request"""
        # Extract all the data from the JSON
        prompt = prompt_data['prompt']
        title = prompt_data.get('title', 'Untitled')
        genre = prompt_data.get('genre', 'General')
        key = prompt_data.get('key', 'C major')
        chord_progression = prompt_data.get('chord_progression', 'Standard')
        instruments = prompt_data.get('instruments', 'Piano')
        tempo = prompt_data.get('tempo', 'Medium')
        rhythm = prompt_data.get('rhythm', 'Standard')
        emotion = prompt_data.get('emotion', 'Neutral')
        
        melody_prompt = f"""
            Based on the client's detailed request, please create a comprehensive prompt for the melody agent to generate a melody that matches the following specifications:

            TITLE: {title}
            GENRE: {genre}
            KEY: {key}
            CHORD PROGRESSION: {chord_progression}
            INSTRUMENTS: {instruments}
            TEMPO: {tempo}
            RHYTHM: {rhythm}
            EMOTION: {emotion}
            
            CLIENT REQUEST: "{prompt}"
            
            Please create a detailed melody generation prompt that incorporates all these musical elements and specifications to guide the melody agent in creating an appropriate musical piece.
        """
        return self.generate_response(melody_prompt)