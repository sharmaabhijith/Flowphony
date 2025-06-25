import re 
import json
from typing import Dict, Any
from .helpers.base_agent import BaseAgent
from .helpers.config import Config

class ReviewerAgent(BaseAgent):
    """Reviewer agent that provides feedback and scoring for musical pieces"""
    
    def __init__(self, config: Config, agent_name: str = "reviewer"):
        super().__init__(config, agent_name)
    
    def generate_response(self, message: str) -> str:
        """Generate a response using DeepInfra API"""
        messages = self.get_conversation_context()
        messages.append({"role": "user", "content": message})
        response = self.call_deepinfra_api(messages)
        return response
    
    def review_music(self, music_content: str) -> str:
        """Review the music and provide feedback with scores"""
        review_prompt = f"""
        Please review the following musical piece in ABC notation:

        {music_content}

        Provide a comprehensive review with scores and feedback according to the specified criteria.
        """
        review_response = self.generate_response(review_prompt)
        return self._extract_scores(review_response)
    
    def _extract_scores(self, review_response: str) -> Dict[str, int]:
        """Extract numerical scores from the review response"""
        try:
            json_match = re.search(r'\{.*\}', review_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data
            return {
                "melody": 0.1,
                "harmony": 0.1,
                "rhythm": 0.1,
                "timbre": 0.1,
                "form": 0.1
            }
        except Exception as e:
            print(f"Error extracting scores: {e}")
            return {
                "melody": 0.1,
                "harmony": 0.1,
                "rhythm": 0.1,
                "timbre": 0.1,
                "form": 0.1
            } 