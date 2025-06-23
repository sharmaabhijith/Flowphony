from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import yaml
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
    device: str = "cuda"
    quantization: str = "4bit"
    batch_size: int = 1
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if hasattr(self, 'api_key') and self.api_key:
            self.api_key = substitute_env_vars(self.api_key)

@dataclass
class ApiModelConfig:
    """Configuration for API models"""
    model_name: str
    base_url: str
    api_key: str
    batch_size: int = 1
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    def __post_init__(self):
        if self.api_key:
            self.api_key = substitute_env_vars(self.api_key)

@dataclass
class Config:
    """Main configuration class"""
    melody_model: ModelConfig
    api_model: ApiModelConfig
    output_dir: str = "output"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Process api_model to substitute environment variables
        if 'api_model' in data:
            if 'api_key' in data['api_model']:
                data['api_model']['api_key'] = substitute_env_vars(data['api_model']['api_key'])
        
        return cls(
            melody_model=ModelConfig(**data['melody_model']),
            api_model=ApiModelConfig(**data['api_model']),
            output_dir=data.get('output_dir', 'output')
        )
    
    def save(self, config_path: str):
        """Save configuration to a YAML file"""
        config_dict = {
            'melody_model': self.melody_model.__dict__,
            'api_model': self.api_model.__dict__,
            'output_dir': self.output_dir
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_llm_config_list(self) -> List[Dict[str, Any]]:
        """Get LLM config list in the format expected by the orchestrator"""
        return [{
            'model_name': self.api_model.model_name,
            'base_url': self.api_model.base_url,
            'api_key': self.api_model.api_key
        }]