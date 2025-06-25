import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, TaskType

from ..agents.helpers.config import Config
from ..orchestrator import MusicOrchestrator
from .reward import RewardFunction
from .buffer import ExperienceBuffer


class Trainer:
    """GFlowNet trainer for melody agent fine-tuning with LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.orchestrator = MusicOrchestrator(config, fine_tune={"melody": True})
        self.orchestrator.melody.setup_lora()
        self.reward_function = RewardFunction(config, self.orchestrator)
        self.optimizer = torch.optim.AdamW(
            self.orchestrator.melody.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        # Create directories
        os.makedirs(config.training.output_dir, exist_ok=True)
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.logs_dir, exist_ok=True)
        # Training statistics
        self.global_step = 0
        self.total_rewards = []
        self.training_losses = []
        self.quality_rewards = []
        self.validity_rewards = []
    
    def _save_checkpoint(self, iteration: int):
        checkpoint = {
            'step': iteration,
            'model_state_dict': self.orchestrator.melody.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'total_rewards': self.total_rewards,
            'training_losses': self.training_losses,
            'quality_rewards': self.quality_rewards,
            'validity_rewards': self.validity_rewards
        }
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"checkpoint_step_{iteration}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

    
    def update_model(self, confidence: List[float], reward: float) -> float:
        """Update the model using a batch of experiences"""

        self.orchestrator.melody.model.train()
        self.optimizer.zero_grad()
        loss = (sum(confidence) - torch.log(torch.tensor(reward + 1e-8))) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.orchestrator.melody.model.parameters(),
            self.config.training.max_grad_norm
        )
        self.optimizer.step()
    
    def train(self, prompts: List[dict]):
        """Train the model using the provided prompts"""
        iteration = 0
        prompt_index = 0

        self.orchestrator.melody.model.train()
        progress_bar = tqdm(total=self.config.training.max_iterations, desc="Training")
        while iteration < len(prompts):
            prompt = prompts[iteration]
            response = self.orchestrator.sample_music(prompt)
            music = response["music"]
            confidence = response["confidence"]
            reward = self.reward_function.compute_reward(music)
            self.update_model(confidence, reward)
            if iteration % self.config.training.save_steps == 0:
                self._save_checkpoint(iteration)
            iteration += 1
            progress_bar.update(1)
        progress_bar.close()

        model_path = os.path.join(self.config.training.output_dir, "Final_Melody_Model")
        self.orchestrator.melody.model.save_pretrained(model_path)
        self.orchestrator.melody.tokenizer.save_pretrained(model_path)


        
    