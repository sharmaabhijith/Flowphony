from collections import deque
from typing import Dict, Any, List
import numpy as np


class ExperienceBuffer:
    """Buffer to store experiences for GFlowNet training"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience: Dict[str, Any]):
        """Add a new experience to the buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample experiences from the buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return np.random.choice(list(self.buffer), batch_size, replace=False).tolist()
    
    def __len__(self):
        return len(self.buffer)