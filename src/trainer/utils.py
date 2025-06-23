from dataclasses import dataclass

@dataclass
class GFlowNetConfig:
    """Configuration for GFlowNet training"""
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 200
    
    # GFlowNet specific parameters
    temperature: float = 1.0
    reward_scale: float = 1.0
    trajectory_balance_loss_weight: float = 1.0
    flow_matching_loss_weight: float = 0.1
    
    # Model parameters
    max_length: int = 512
    max_new_tokens: int = 128
    min_melody_length: int = 20
    max_melody_length: int = 200
    
    # Online learning parameters
    experience_buffer_size: int = 1000
    min_experiences_for_training: int = 10
    update_frequency: int = 1
    
    # Reward parameters
    reward_temperature: float = 0.1
    quality_weight: float = 0.8
    validity_weight: float = 0.2
    
    # Logging
    use_wandb: bool = True
    project_name: str = "melody-gflownet"
    
    # Paths
    output_dir: str = "models/gflownet_melody"
    checkpoint_dir: str = "checkpoints/gflownet_melody"
    logs_dir: str = "logs/gflownet_melody"