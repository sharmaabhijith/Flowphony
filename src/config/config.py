from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import os
from pathlib import Path

def substitute_env_vars(value: str) -> str:
    """Substitute environment variables in a string value"""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, value)
    return value

@dataclass
class ModelConfig:
    """Configuration for fine-tuned models"""
    model_name: str
    model_path: str
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.api_key:
            self.api_key = substitute_env_vars(self.api_key)

@dataclass
class AgentConfig:
    """Configuration for agents"""
    name: str
    description: str
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60

@dataclass
class Config:
    """Main configuration class"""
    # Model configurations
    melody_model: ModelConfig
    harmony_model: ModelConfig
    
    # Agent configurations
    melody_agent: AgentConfig
    harmony_agent: AgentConfig
    
    # AutoGen configurations
    llm_config_list: List[Dict[str, Any]]
    
    # Output settings
    output_dir: str = "output"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Process llm_config_list to substitute environment variables
        for config in data['llm_config_list']:
            if 'api_key' in config:
                config['api_key'] = substitute_env_vars(config['api_key'])
        
        return cls(
            melody_model=ModelConfig(**data['melody_model']),
            harmony_model=ModelConfig(**data['harmony_model']),
            melody_agent=AgentConfig(**data['melody_agent']),
            harmony_agent=AgentConfig(**data['harmony_agent']),
            llm_config_list=data['llm_config_list'],
            output_dir=data.get('output_dir', 'output')
        )
    
    def save(self, config_path: str):
        """Save configuration to a JSON file"""
        config_dict = {
            'melody_model': self.melody_model.__dict__,
            'harmony_model': self.harmony_model.__dict__,
            'melody_agent': self.melody_agent.__dict__,
            'harmony_agent': self.harmony_agent.__dict__,
            'llm_config_list': self.llm_config_list,
            'output_dir': self.output_dir
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)