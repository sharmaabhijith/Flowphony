import re
import logging
from ..agents.reviewer_agent import ReviewerAgent
from ..orchestrator import MusicOrchestrator
from ..agents.helpers.config import Config


class RewardFunction:
    """Reward function for melody generation using reviewer agent"""
    
    def __init__(self, config: Config, orchestrator: MusicOrchestrator):
        self.config = config
        self.temperature = config.temperature
        self.reviewer = orchestrator.reviewer
    
    def compute_reward(self, melody: str, prompt: str, full_music: str = None) -> float:
        """Compute reward for a generated melody"""
        quality_reward = self._compute_quality_reward(melody, prompt, full_music)
        validity_reward = self._compute_validity_reward(melody)
        total_reward = (
            quality_reward * 0.8 +
            validity_reward * 0.2
        )
        scaled_reward = total_reward ** (1 / self.temperature)
        return min(max(scaled_reward, 0.0), 1.0)
    
    def _compute_quality_reward(self, melody: str, prompt: str, full_music: str = None) -> float:
        """Compute quality reward using reviewer agent"""
        # Create review prompt
        if full_music:
            review_prompt = f"""
            Please review this complete music piece generated for the prompt: "{prompt}"
            
            Complete music in ABC notation:
            {full_music}
            
            Please provide a numerical score from 0 to 10 based on:
            1. Melodic Structure (flow, thematic development, variety)
            2. Harmony and musical coherence
            3. Rhythmic complexity and interest
            4. Overall musical quality
            5. How well it matches the original prompt
            
            Respond with only a number between 0 and 10.
            """
        else:
            review_prompt = f"""
            Please review this melody generated for the prompt: "{prompt}"
            
            Melody in ABC notation:
            {melody}
            
            Please provide a numerical score from 0 to 10 based on:
            1. Melodic Structure (flow, thematic development, variety)
            2. Harmony and musical coherence
            3. Rhythmic complexity and interest
            4. Overall musical quality
            
            Respond with only a number between 0 and 10.
            """
        
        # Get reviewer feedback
        review_response = self.reviewer.generate_response(review_prompt)
        
        # Extract numerical score
        score = self._extract_score_from_response(review_response)
        
        return score / 10.0
    
    def _compute_validity_reward(self, melody: str) -> float:
        """Compute reward for musical validity"""
        if not melody or len(melody.strip()) < 10:
            return 0.1
        
        score = 0.5  # Base score
        
        # Reward for proper ABC structure
        if "X:1" in melody and "T:" in melody and "K:" in melody:
            score += 0.2
        
        # Reward for reasonable length
        if 50 < len(melody) < 500:
            score += 0.1
        
        # Reward for musical content
        if any(note in melody for note in "ABCDEFGabcdefg"):
            score += 0.1
        
        # Reward for proper ending
        if ":|" in melody or "|]" in melody:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from reviewer response"""
        try:
            # Look for numbers in the response
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 10.0)
        except:
            pass
        
        # Fallback: estimate score based on response length and content
        if "excellent" in response.lower() or "great" in response.lower():
            return 8.0
        elif "good" in response.lower() or "nice" in response.lower():
            return 6.0
        elif "fair" in response.lower() or "average" in response.lower():
            return 4.0
        else:
            return 2.0