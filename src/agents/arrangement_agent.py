from typing import Dict, Any
from .helpers.base_agent import BaseAgent
from .helpers.config import Config

class ArrangementAgent(BaseAgent):
    """Arrangement agent that formats and arranges musical pieces"""
    
    def __init__(self, config: Config, agent_name: str = "arrangement"):
        super().__init__(config, agent_name)
    
    def generate_response(self, message: str) -> str:
        """Generate a response using DeepInfra API"""
        # Prepare messages for API call
        messages = self.get_conversation_context()
        messages.append({"role": "user", "content": message})
        
        # Call DeepInfra API
        response = self.call_deepinfra_api(messages)
        return response
    
    def arrange_music(self, music_content: str) -> str:
        """Arrange and format the music into proper ABC notation"""
        arrangement_prompt = f"""
        Given the following musical content:

        {music_content}

        Please arrange and format this into proper ABC notation:
        1. Ensure all ABC notation rules are followed
        2. Remove any empty lines or formatting issues
        3. Organize the structure properly with headers and sections
        4. Make sure the notation is clean and readable
        5. Maintain all musical content while improving formatting

        Please provide the properly arranged ABC notation.
        """
        return self.generate_response(arrangement_prompt) 