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
    
    def compute_reward(self, full_music: str) -> float:
        """Compute reward for a generated melody"""
        quality_reward = self._compute_quality_reward(full_music)
        validity_reward = self._compute_validity_reward(full_music)
        total_reward = (
            quality_reward * 0.8 +
            validity_reward * 0.2
        )
        return min(max(total_reward, 0), 1)
    
    def _compute_quality_reward(self, full_music: str ) -> float:
        """Compute quality reward using reviewer agent"""
        review_response = self.reviewer.review_music(full_music)
        return review_response["melody"] 
    
    def _compute_validity_reward(self, full_music: str) -> float:
        """Compute reward for musical validity"""
        if len(full_music.strip()) < 10:
            return 0.1
        
        score = 0
        if "X:1" in full_music and "T:" in full_music and "K:" in full_music:
            score += 0.4
        if 50 < len(full_music) < 500:
            score += 0.1
        if any(note in full_music for note in "ABCDEFGabcdefg"):
            score += 0.3
        if ":|" in full_music or "|]" in full_music:
            score += 0.2
        
        return min(score, 1.0)