#!/usr/bin/env python3
"""
GFlowNet Training Script for Melody Agent

This script performs fine-tuning of the melody agent using the full multi-agent pipeline.
The model learns from reviewer feedback in real-time, similar to PPO RL fine-tuning.

Key Features:
- No external dataset required
- Real-time learning from reviewer feedback
- Full multi-agent pipeline integration
- Experience buffer for stable training
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from trainer.gflownet_trainer import OnlineGFlowNetTrainer, OnlineGFlowNetConfig
from trainer.data_utils import generate_melody_prompts
from config.config import Config

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Online GFlowNet Training for Melody Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Maximum number of training iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for online training"
    )
    parser.add_argument(
        "--update_frequency",
        type=int,
        default=1,
        help="Update model every N generations"
    )
    parser.add_argument(
        "--min_experiences",
        type=int,
        default=5,
        help="Minimum experiences before starting training"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100,
        help="Experience buffer size"
    )
    
    # Model parameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation"
    )
    
    # Prompt generation
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=20,
        help="Number of prompts to generate for training"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Path to JSON file containing custom prompts (optional)"
    )
    
    # Configuration
    parser.add_argument(
        "--config_file",
        type=str,
        default="src/config/agent.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/gflownet_melody",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/gflownet_melody",
        help="Directory to save checkpoints"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="melody-gflownet",
        help="WandB project name"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log progress every N steps"
    )
    
    # Resume training
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    return parser.parse_args()

def load_or_generate_prompts(args) -> List[str]:
    """Load prompts from file or generate new ones"""
    if args.prompts_file and os.path.exists(args.prompts_file):
        logger.info(f"Loading prompts from {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if all(isinstance(item, dict) and 'prompt' in item for item in data):
                prompts = [item['prompt'] for item in data]
            elif all(isinstance(item, str) for item in data):
                prompts = data
            else:
                raise ValueError("Invalid prompts file format")
        elif isinstance(data, dict) and 'prompts' in data:
            prompts = data['prompts']
        else:
            raise ValueError("Invalid prompts file format")
        
        logger.info(f"Loaded {len(prompts)} prompts from file")
    else:
        logger.info(f"Generating {args.num_prompts} random prompts")
        prompts = generate_melody_prompts(args.num_prompts)
        
        # Save generated prompts
        os.makedirs(args.output_dir, exist_ok=True)
        prompts_file = os.path.join(args.output_dir, "generated_prompts.json")
        with open(prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        logger.info(f"Saved generated prompts to {prompts_file}")
    
    return prompts

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GFlowNet training for melody agent")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    try:
        config = Config.from_file(args.config_file)
        logger.info(f"Loaded configuration from {args.config_file}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Load or generate prompts
    try:
        prompts = load_or_generate_prompts(args)
        logger.info(f"Using {len(prompts)} prompts for training")
    except Exception as e:
        logger.error(f"Failed to load/generate prompts: {e}")
        return 1
    
    # Create online GFlowNet configuration
    online_config = OnlineGFlowNetConfig(
        learning_rate=args.learning_rate,
        update_frequency=args.update_frequency,
        min_experiences_for_training=args.min_experiences,
        experience_buffer_size=args.buffer_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps
    )
    
    # Initialize trainer
    try:
        trainer = OnlineGFlowNetTrainer(config, online_config)
        logger.info("Initialized GFlowNet trainer")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return 1
    
    # Resume from checkpoint if specified
    if args.resume_from:
        try:
            step = trainer.load_checkpoint(args.resume_from)
            logger.info(f"Resumed training from step {step}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 1
    
    # Start training
    try:
        logger.info("Starting training...")
        trainer.train_online(prompts, args.max_iterations)
        logger.info("Training completed successfully!")
        
        # Save final prompts for reference
        final_prompts_file = os.path.join(args.output_dir, "final_prompts.json")
        with open(final_prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        # Print training summary
        if trainer.total_rewards:
            final_avg_reward = np.mean(trainer.total_rewards[-10:])
            logger.info(f"Final average reward: {final_avg_reward:.3f}")
            logger.info(f"Total experiences collected: {len(trainer.total_rewards)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    import numpy as np
    exit_code = main()
    sys.exit(exit_code) 