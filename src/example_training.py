#!/usr/bin/env python3
"""
Example script demonstrating GFlowNet training for melody agent

This script shows how to:
1. Set up training without any external dataset
2. Train the melody agent using reviewer feedback
3. Monitor training progress
4. Use the trained model for music generation

The system works like PPO RL fine-tuning where the model learns
from the reviewer agent's feedback in real-time.
"""

import json
import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from trainer.gflownet_trainer import OnlineGFlowNetTrainer, OnlineGFlowNetConfig
from trainer.data_utils import generate_melody_prompts, save_prompts_to_file
from config.config import Config
from orchestrator import MusicOrchestrator

def setup_logging():
    """Setup logging for the example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_example.log'),
            logging.StreamHandler()
        ]
    )

def create_training_prompts():
    """Create a set of training prompts"""
    print("Generating training prompts...")
    
    # Generate diverse prompts
    prompts = generate_melody_prompts(num_prompts=15)
    
    # Add some specialized prompts
    jazz_prompts = generate_melody_prompts(num_prompts=5)
    classical_prompts = generate_melody_prompts(num_prompts=5)
    
    all_prompts = prompts + jazz_prompts + classical_prompts
    
    # Save prompts for reference
    os.makedirs("data", exist_ok=True)
    save_prompts_to_file(all_prompts, "data/online_training_prompts.json")
    
    print(f"Generated {len(all_prompts)} training prompts")
    return all_prompts

def run_training_example():
    """Run the training example"""
    print("=" * 60)
    print("GFlowNet Training Example")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = Config.from_file('src/config/agent.json')
        print("✓ Loaded configuration")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return
    
    # Create training prompts
    prompts = create_training_prompts()
    
    # Configure training
    online_config = OnlineGFlowNetConfig(
        learning_rate=1e-6,  # Conservative learning rate for training
        max_iterations=50,   # Start with fewer iterations for testing
        update_frequency=1,  # Update every generation
        min_experiences_for_training=3,  # Start training after 3 experiences
        experience_buffer_size=20,  # Smaller buffer for testing
        max_length=512,
        max_new_tokens=128,
        temperature=1.0,
        use_wandb=False,  # Disable wandb for this example
        output_dir="models/example",
        checkpoint_dir="checkpoints/example",
        save_steps=10,
        logging_steps=5
    )
    
    print(f"Training configuration:")
    print(f"  - Learning rate: {online_config.learning_rate}")
    print(f"  - Max iterations: {online_config.max_iterations}")
    print(f"  - Buffer size: {online_config.experience_buffer_size}")
    print(f"  - Min experiences: {online_config.min_experiences_for_training}")
    
    # Initialize trainer
    try:
        trainer = OnlineGFlowNetTrainer(config, online_config)
        print("✓ Initialized trainer")
    except Exception as e:
        print(f"✗ Failed to initialize trainer: {e}")
        return
    
    # Start training
    print("\nStarting training...")
    print("This will take some time as each iteration involves:")
    print("  1. Generating a melody with the current model")
    print("  2. Running the full multi-agent pipeline")
    print("  3. Getting feedback from the reviewer agent")
    print("  4. Updating the model based on the reward")
    
    try:
        trainer.train_online(prompts, max_iterations=online_config.max_iterations)
        print("✓ Training completed!")
        
        # Print training summary
        if trainer.total_rewards:
            final_avg_reward = sum(trainer.total_rewards[-5:]) / 5
            print(f"\nTraining Summary:")
            print(f"  - Total experiences: {len(trainer.total_rewards)}")
            print(f"  - Final average reward: {final_avg_reward:.3f}")
            print(f"  - Best reward: {max(trainer.total_rewards):.3f}")
            print(f"  - Model saved to: {online_config.output_dir}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    return trainer

def test_trained_model(trainer):
    """Test the trained model with new prompts"""
    print("\n" + "=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "Compose a jazz melody in G major with moderate tempo",
        "Create a folk tune in C major with happy mood",
        "Write a classical piece in D minor with slow tempo"
    ]
    
    print("Testing with new prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        # Generate melody with trained model
        melody = trainer.generate_melody(prompt)
        print(f"Generated melody: {melody[:100]}...")
        
        # Get reward from reviewer
        reward = trainer.reward_function.compute_reward(melody, prompt)
        print(f"Reviewer reward: {reward:.3f}")

def compare_with_baseline():
    """Compare the trained model with baseline AutoGen agents"""
    print("\n" + "=" * 60)
    print("Comparison with Baseline")
    print("=" * 60)
    
    # Load configuration
    config = Config.from_file('src/config/agent.json')
    
    # Test prompt
    test_prompt = "Compose a jazz melody in G major with moderate tempo"
    
    print(f"Test prompt: {test_prompt}")
    
    # Test with baseline (regular AutoGen agents)
    print("\n1. Testing with baseline AutoGen agents...")
    baseline_orchestrator = MusicOrchestrator(config, fine_tune=False)
    
    try:
        baseline_result = baseline_orchestrator.run_music_generation(
            test_prompt, "output/baseline_test"
        )
        print(f"✓ Baseline generation completed")
    except Exception as e:
        print(f"✗ Baseline generation failed: {e}")
        baseline_result = None
    
    # Test with trained model (if available)
    trained_model_path = "models/example/final_model"
    if os.path.exists(trained_model_path):
        print("\n2. Testing with trained model...")
        trained_orchestrator = MusicOrchestrator(
            config, 
            fine_tune=True, 
            melody_model_path=trained_model_path
        )
        
        try:
            trained_result = trained_orchestrator.run_music_generation(
                test_prompt, "output/trained_test"
            )
            print(f"✓ Trained model generation completed")
        except Exception as e:
            print(f"✗ Trained model generation failed: {e}")
            trained_result = None
    else:
        print(f"\n2. Trained model not found at {trained_model_path}")
        trained_result = None
    
    # Compare results
    print("\n3. Comparison Results:")
    if baseline_result and trained_result:
        print("  - Both models generated music successfully")
        print("  - Check the output files for detailed comparison")
    elif baseline_result:
        print("  - Only baseline model worked")
    elif trained_result:
        print("  - Only trained model worked")
    else:
        print("  - Both models failed")

def main():
    """Main function"""
    print("GFlowNet Training Example for Melody Agent")
    print("This example demonstrates fine-tuning without external datasets")
    
    # Check if user wants to run training
    response = input("\nDo you want to run training? (y/n): ").lower().strip()
    
    if response == 'y':
        # Run training
        trainer = run_training_example()
        
        if trainer:
            # Test the trained model
            test_trained_model(trainer)
            
            # Compare with baseline
            compare_with_baseline()
    else:
        print("Skipping training. You can run the example later with:")
        print("python src/example_training.py")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("\nTo run training manually:")
    print("python src/trainer/train_script.py --max_iterations 100")
    print("\nTo use the trained model:")
    print("python src/run.py --fine_tune --melody_model_path models/example/final_model")

if __name__ == "__main__":
    main() 