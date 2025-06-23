#!/usr/bin/env python3
"""
GFlowNet Training Script for Melody Agent Fine-tuning

This script performs GFlowNet-based fine-tuning of the melody agent for diverse music generation.
The model learns from reviewer feedback and optimizes for both quality and diversity.

Usage:
    python src/train.py
"""

import json
import os
import sys
from pathlib import Path
from typing import List
import numpy as np
import random
sys.path.append(str(Path(__file__).parent.parent))
from trainer import Trainer
from agents.helpers.config import Config, PromptsConfig

def generate_training_prompts(prompts_config: PromptsConfig) -> List[str]:
    """Generate training prompts based on the provided configuration"""
    prompts = []
    for _ in range(prompts_config.num_prompts):
        components = []
        if prompts_config.styles:
            components.append(random.choice(prompts_config.styles))
        if prompts_config.keys:
            components.append(f"in {random.choice(prompts_config.keys)}")
        if prompts_config.instruments:
            components.append(f"for {random.choice(prompts_config.instruments)}")
        if prompts_config.tempos:
            components.append(f"at a {random.choice(prompts_config.tempos)} tempo")
        if prompts_config.moods:
            components.append(f"with a {random.choice(prompts_config.moods)} mood")
            
        if components:
            prompt = "Create a music " + " ".join(components)
            prompts.append(prompt.replace("  ", " ").strip())
            
    return prompts

def load_training_prompts(prompts_file: str) -> List[str]:
    """Load training prompts from a JSON file"""
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        # If it's a list of dictionaries with 'prompt' key
        if all(isinstance(item, dict) and 'prompt' in item for item in data):
            return [item['prompt'] for item in data]
        # If it's a list of strings
        elif all(isinstance(item, str) for item in data):
            return data
    elif isinstance(data, dict) and 'prompts' in data:
        return data['prompts']
    
    raise ValueError("Invalid prompts file format. Expected list of strings or list of dicts with 'prompt' key")

def main():
    """Main training function"""
    # Load configuration from config.yaml
    config = Config.from_file("src/config.yaml")
    # Load training prompts
    if config.prompts_file:
        prompts = load_training_prompts(config.prompts_file)
    else:
        prompts = generate_training_prompts(config.prompts)
    trainer = Trainer(config)
    trainer.train(prompts)
    # Save final prompts for reference
    final_prompts_file = os.path.join(config.training.output_dir, "final_prompts.json")
    with open(final_prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    # Evaluate if requested
    if config.evaluation.evaluate:
        if config.evaluation.eval_prompts_file:
            eval_prompts = load_training_prompts(config.evaluation.eval_prompts_file)
        else:
            eval_prompts = prompts[:5]  # Use first 5 training prompts for evaluation
        eval_results = trainer.evaluate_model(eval_prompts, config.evaluation.num_eval_samples)
        eval_file = os.path.join(config.training.output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
    

if __name__ == "__main__":
    main()