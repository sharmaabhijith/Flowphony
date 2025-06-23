from typing import Dict, Any
from .helpers.base_agent import BaseAgent
from .helpers.config import Config

class ReviewerAgent(BaseAgent):
    """Reviewer agent that provides feedback and scoring for musical pieces"""
    
    def __init__(self, config: Config, agent_name: str = "reviewer"):
        super().__init__(config, agent_name)
    
    def generate_response(self, message: str) -> str:
        """Generate a response using DeepInfra API"""
        # Prepare messages for API call
        messages = self.get_conversation_context()
        messages.append({"role": "user", "content": message})
        
        # Call DeepInfra API
        response = self.call_deepinfra_api(messages)
        return response
    
    def review_music(self, music_content: str) -> str:
        """Review the music and provide feedback with scores"""
        review_prompt = f"""
        Please review the following musical piece in ABC notation:

        {music_content}

        Provide a comprehensive review with scores and feedback according to the specified criteria.
        """
        return self.generate_response(review_prompt)
    
    def extract_scores(self, review_response: str) -> Dict[str, int]:
        """Extract numerical scores from the review response"""
        try:
            # Try to parse JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', review_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                if 'scores' in data:
                    return data['scores']
            
            # If no valid JSON found, return default scores
            return {
                "melody": 50,
                "harmony": 50,
                "rhythm": 50,
                "timbre": 50,
                "form": 50
            }
        except Exception as e:
            print(f"Error extracting scores: {e}")
            return {
                "melody": 50,
                "harmony": 50,
                "rhythm": 50,
                "timbre": 50,
                "form": 50
            } 