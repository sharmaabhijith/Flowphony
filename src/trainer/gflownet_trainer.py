import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
import autogen
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb
from abc import ABC, abstractmethod
import time
from collections import deque

from ..agents.melody_agent import MelodyAgent
from ..orchestrator import MusicOrchestrator
from ..config.config import Config

@dataclass
class GFlowNetConfig:
    """Configuration for online GFlowNet training"""
    # Training parameters
    learning_rate: float = 1e-6  # Lower learning rate for online training
    batch_size: int = 1  # Online training uses single samples
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    
    # GFlowNet specific parameters
    temperature: float = 1.0
    reward_scale: float = 1.0
    trajectory_balance_loss_weight: float = 1.0
    
    # Model parameters
    max_length: int = 512
    max_new_tokens: int = 128
    
    # Online learning parameters
    experience_buffer_size: int = 100  # Store recent experiences
    min_experiences_for_training: int = 5  # Minimum experiences before training
    update_frequency: int = 1  # Update model every N generations
    
    # Logging
    use_wandb: bool = True
    project_name: str = "melody-gflownet"
    
    # Paths
    output_dir: str = "models/gflownet_melody"
    checkpoint_dir: str = "checkpoints/gflownet_melody"

class ExperienceBuffer:
    """Buffer to store recent experiences for online training"""
    
    def __init__(self, max_size: int = 100):
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

class RewardFunction:
    """Online reward function that uses the reviewer agent"""
    
    def __init__(self, config: Config, llm_config: Dict[str, Any]):
        self.config = config
        self.llm_config = llm_config
        
        # Load reviewer system message
        with open('src/config/sysmsg.json', 'r') as f:
            system_messages = json.load(f)
        
        self.reviewer = autogen.AssistantAgent(
            name="ReviewerAgent",
            system_message=system_messages["reviewer"],
            llm_config=llm_config
        )
        
        self.user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human client.",
            code_execution_config=False,
            llm_config=llm_config
        )
    
    def compute_reward(self, melody: str, prompt: str, full_music: str = None) -> float:
        """Compute reward using the reviewer agent"""
        try:
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
            chat_result = self.user_proxy.initiate_chat(
                self.reviewer,
                message=review_prompt,
                max_turns=1
            )
            
            if chat_result and chat_result.chat_history:
                response = chat_result.chat_history[-1]["content"]
                # Extract numerical score
                try:
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', response)
                    if numbers:
                        score = float(numbers[0])
                        # Normalize to 0-1 range
                        return min(max(score / 10.0, 0.0), 1.0)
                except:
                    pass
            
            # Fallback reward
            return self._fallback_reward(melody, full_music)
            
        except Exception as e:
            logging.warning(f"Error computing reviewer reward: {e}")
            return self._fallback_reward(melody, full_music)
    
    def _fallback_reward(self, melody: str, full_music: str = None) -> float:
        """Fallback reward function when reviewer fails"""
        if not melody or len(melody.strip()) < 10:
            return 0.1
        
        # Simple heuristics for melody quality
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
        
        # Additional reward for complete music
        if full_music and len(full_music) > len(melody):
            score += 0.1
        
        return min(score, 1.0)

class GFlowNetTrainer:
    """GFlowNet trainer for real-time melody agent fine-tuning"""
    
    def __init__(self, config: Config, online_config: GFlowNetConfig):
        self.config = config
        self.online_config = online_config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self._setup_model()
        
        # Initialize reward function
        self.reward_function = RewardFunction(config, self._get_llm_config())
        
        # Initialize experience buffer
        self.experience_buffer = ExperienceBuffer(online_config.experience_buffer_size)
        
        # Initialize orchestrator for full pipeline
        self.orchestrator = MusicOrchestrator(
            config=config,
            fine_tune=True,
            melody_model_path=None  # Will be updated dynamically
        )
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=online_config.learning_rate
        )
        
        # Setup wandb
        if online_config.use_wandb:
            wandb.init(
                project=online_config.project_name,
                config=online_config.__dict__
            )
        
        # Create output directories
        os.makedirs(online_config.output_dir, exist_ok=True)
        os.makedirs(online_config.checkpoint_dir, exist_ok=True)
        
        # Training statistics
        self.global_step = 0
        self.total_rewards = []
        self.training_losses = []
    
    def _setup_model(self):
        """Setup the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.melody_model.model_name
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.melody_model.model_name,
                device_map=self.config.melody_model.device,
                torch_dtype=torch.float16
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.device = self.config.melody_model.device
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for agents"""
        return {
            "config_list": [{
                "model": self.config.llm_config_list[0]["model_name"],
                "api_key": os.getenv("DEEPINFRA_API_KEY"),
                "base_url": self.config.llm_config_list[0]["base_url"]
            }],
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 60,
            "cache_seed": None
        }
    
    def generate_melody(self, prompt: str) -> str:
        """Generate a melody using the current model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.online_config.max_length,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.online_config.max_new_tokens,
                temperature=self.online_config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        melody = generated_text[len(prompt):].strip()
        return melody
    
    def compute_gflownet_loss(self, experience: Dict[str, Any]) -> torch.Tensor:
        """Compute GFlowNet loss for a single experience"""
        prompt = experience["prompt"]
        melody = experience["melody"]
        reward = experience["reward"]
        
        # Forward pass for the complete sequence
        full_text = prompt + " " + melody
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.online_config.max_length,
            truncation=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get the log probability of the generated melody
        melody_tokens = self.tokenizer(
            melody,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        
        if melody_tokens["input_ids"].size(1) > 0:
            # Compute log probability of the trajectory
            log_prob = 0.0
            for t in range(melody_tokens["input_ids"].size(1)):
                token_id = melody_tokens["input_ids"][0, t]
                log_prob += log_probs[0, -melody_tokens["input_ids"].size(1) + t, token_id]
            
            # Trajectory balance loss: log P(x) - log R(x)
            loss = (log_prob - torch.log(torch.tensor(reward + 1e-8, device=self.device))) ** 2
            return loss
        
        return torch.tensor(0.0, device=self.device)
    
    def update_model(self, experiences: List[Dict[str, Any]]) -> float:
        """Update the model using a batch of experiences"""
        if not experiences:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        
        for experience in experiences:
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_gflownet_loss(experience)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.online_config.max_grad_norm
            )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(experiences)
        self.training_losses.append(avg_loss)
        
        return avg_loss
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a single prompt through the full pipeline and collect experience"""
        self.logger.info(f"Processing prompt: {prompt}")
        
        # Generate melody using current model
        melody = self.generate_melody(prompt)
        
        # Run full pipeline to get complete music
        results_dir = "temp_output"
        os.makedirs(results_dir, exist_ok=True)
        
        # Temporarily update orchestrator's melody agent with current model
        self._update_orchestrator_model()
        
        # Generate complete music
        abc_filepath = self.orchestrator.run_music_generation(prompt, results_dir)
        
        # Read the generated music
        full_music = ""
        if abc_filepath and os.path.exists(abc_filepath):
            with open(abc_filepath, 'r') as f:
                full_music = f.read()
        
        # Compute reward using reviewer
        reward = self.reward_function.compute_reward(melody, prompt, full_music)
        
        # Create experience
        experience = {
            "prompt": prompt,
            "melody": melody,
            "full_music": full_music,
            "reward": reward,
            "step": self.global_step,
            "timestamp": time.time()
        }
        
        # Add to experience buffer
        self.experience_buffer.add(experience)
        
        # Update training statistics
        self.total_rewards.append(reward)
        
        # Log experience
        self.logger.info(f"Experience - Reward: {reward:.3f}, Melody length: {len(melody)}")
        
        return experience
    
    def _update_orchestrator_model(self):
        """Update the orchestrator's melody agent with the current model"""
        # Create a temporary melody agent with current model
        temp_melody_agent = MelodyAgent(
            name="TempMelodyAgent",
            model_path=None,  # Will use the current model in memory
            device=self.device,
            llm_config=self._get_llm_config(),
            config=self.config,
            agent_config=self.config.melody_agent
        )
        
        # Manually set the model and tokenizer
        temp_melody_agent.model = self.model
        temp_melody_agent.tokenizer = self.tokenizer
        temp_melody_agent.use_fine_tuned_model = True
        
        # Update the orchestrator's melody agent
        for i, agent in enumerate(self.orchestrator.agents):
            if agent.name == "MelodyAgent":
                self.orchestrator.agents[i] = temp_melody_agent
                break
    
    def train(self, prompts: List[str], max_iterations: int = 1000):
        """Train the model using the provided prompts"""
        self.logger.info(f"Starting training with {len(prompts)} prompts")
        self.logger.info(f"Max iterations: {max_iterations}")
        
        iteration = 0
        prompt_index = 0
        
        while iteration < max_iterations:
            # Get next prompt (cycle through prompts)
            prompt = prompts[prompt_index % len(prompts)]
            prompt_index += 1
            
            # Process the prompt
            experience = self.process_prompt(prompt)
            
            # Update model if we have enough experiences
            if len(self.experience_buffer) >= self.online_config.min_experiences_for_training:
                if iteration % self.online_config.update_frequency == 0:
                    # Sample experiences for training
                    batch_experiences = self.experience_buffer.sample(
                        min(len(self.experience_buffer), 5)  # Use up to 5 experiences
                    )
                    
                    # Update model
                    loss = self.update_model(batch_experiences)
                    
                    self.logger.info(f"Iteration {iteration}: Loss = {loss:.4f}, "
                                   f"Avg Reward = {np.mean(self.total_rewards[-10:]):.3f}")
                    
                    # Log to wandb
                    if self.online_config.use_wandb:
                        wandb.log({
                            'iteration': iteration,
                            'loss': loss,
                            'avg_reward': np.mean(self.total_rewards[-10:]),
                            'experience_buffer_size': len(self.experience_buffer),
                            'current_reward': experience['reward']
                        })
            
            # Save checkpoint periodically
            if iteration % self.online_config.save_steps == 0:
                self._save_checkpoint(iteration)
            
            # Log progress
            if iteration % self.online_config.logging_steps == 0:
                avg_reward = np.mean(self.total_rewards[-10:]) if self.total_rewards else 0.0
                self.logger.info(f"Progress - Iteration: {iteration}, "
                               f"Avg Reward: {avg_reward:.3f}, "
                               f"Buffer Size: {len(self.experience_buffer)}")
            
            iteration += 1
        
        # Save final model
        self._save_model("final_model")
        self.logger.info("Training completed!")
    
    def _save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.online_config.__dict__,
            'total_rewards': self.total_rewards,
            'training_losses': self.training_losses
        }
        
        checkpoint_path = os.path.join(
            self.online_config.checkpoint_dir,
            f"checkpoint_step_{step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_model(self, name: str):
        """Save the trained model"""
        model_path = os.path.join(self.online_config.output_dir, name)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        self.logger.info(f"Model saved: {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_rewards = checkpoint.get('total_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        self.global_step = checkpoint['step']
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['step'] 