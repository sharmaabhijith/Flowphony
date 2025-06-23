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
    
    def create_melody_prompt(self, client_request: str) -> str:
        """Create a specific prompt for melody generation based on client request"""
        melody_prompt = f"""
            Based on the client's request: \n"{client_request}"
            Please create a prompt for the melody agent to generate a melody that matches the following requirements:
        """
        return self.generate_response(melody_prompt)