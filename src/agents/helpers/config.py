from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
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
class TrainingConfig:
    """Configuration for training"""
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 200
    max_iterations: int = 1000
    temperature: float = 1.0
    reward_scale: float = 1.0
    trajectory_balance_loss_weight: float = 1.0
    max_length: int = 512
    max_new_tokens: int = 128
    min_melody_length: int = 20
    max_melody_length: int = 200
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    experience_buffer_size: int = 1000
    min_experiences_for_training: int = 10
    update_frequency: int = 1
    reward_temperature: float = 0.1
    quality_weight: float = 0.8
    validity_weight: float = 0.2
    use_wandb: bool = True
    project_name: str = "melody-gflownet-lora"
    output_dir: str = "models/melody"
    checkpoint_dir: str = "checkpoints/melody"
    logs_dir: str = "logs/melody"

@dataclass
class PromptsConfig:
    """Configuration for prompt generation"""
    num_prompts: int = 500
    styles: List[str] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    tempos: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    num_eval_samples: int = 5
    eval_prompts_file: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["reward", "quality", "validity", "diversity"])

@dataclass
class SystemConfig:
    """System configuration"""
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

@dataclass
class Config:
    """Main configuration class"""
    melody_model: ModelConfig
    api_model: ApiModelConfig
    training: TrainingConfig
    prompts: PromptsConfig
    evaluation: EvaluationConfig
    system: SystemConfig
    output_dir: str = "output"
    prompts_file: Optional[str] = None
    resume_from: Optional[str] = None
    evaluate: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Process api_model to substitute environment variables
        if 'api_model' in data and 'api_key' in data['api_model']:
            data['api_model']['api_key'] = substitute_env_vars(data['api_model']['api_key'])
        
        return cls(
            melody_model=ModelConfig(**data['melody_model']),
            api_model=ApiModelConfig(**data['api_model']),
            training=TrainingConfig(**data.get('training', {})),
            prompts=PromptsConfig(**data.get('prompts', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            system=SystemConfig(**data.get('system', {})),
            output_dir=data.get('output_dir', 'output'),
            prompts_file=data.get('prompts_file'),
            resume_from=data.get('resume_from'),
            evaluate=data.get('evaluate', False)
        )
    
    def save(self, config_path: str):
        """Save configuration to a YAML file"""
        # A bit tricky to dump dataclasses, so we convert to dict
        config_dict = {
            'melody_model': self.melody_model.__dict__,
            'api_model': self.api_model.__dict__,
            'training': self.training.__dict__,
            'prompts': self.prompts.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__,
            'output_dir': self.output_dir,
            'prompts_file': self.prompts_file,
            'resume_from': self.resume_from,
            'evaluate': self.evaluate
        }
        # handle list for target_modules in training
        config_dict['training']['target_modules'] = list(config_dict['training']['target_modules'])

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_llm_config_list(self) -> List[Dict[str, Any]]:
        """Get LLM config list in the format expected by the orchestrator"""
        return [{
            'model_name': self.api_model.model_name,
            'base_url': self.api_model.base_url,
            'api_key': self.api_model.api_key
        }]