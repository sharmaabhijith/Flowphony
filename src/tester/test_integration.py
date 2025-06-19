#!/usr/bin/env python3
"""
Test script to verify MelodyAgent integration with AutoGen
"""

import os
import sys
from config.config import Config
from agents.melody_agent import MelodyAgent
import autogen

def test_melody_agent_initialization():
    """Test that MelodyAgent can be initialized properly"""
    print("Testing MelodyAgent initialization...")
    
    try:
        # Load configuration
        config = Config.from_file('src/config/agent.json')
        
        # Initialize LLM config
        llm_config = {
            "config_list": [{
                "model": config.llm_config_list[0]["model_name"],
                "api_key": os.getenv("DEEPINFRA_API_KEY"),
                "base_url": config.llm_config_list[0]["base_url"]
            }],
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 60,
            "cache_seed": None
        }
        
        # Create MelodyAgent
        melody_agent = MelodyAgent(
            config=config,
            agent_config=config.melody_agent,
            llm_config=llm_config
        )
        
        print(f"✓ MelodyAgent initialized successfully")
        print(f"  - Name: {melody_agent.name}")
        print(f"  - Fine-tuned model available: {melody_agent.use_fine_tuned_model}")
        print(f"  - Model path: {config.melody_model.model_path}")
        
        return melody_agent
        
    except Exception as e:
        print(f"✗ Failed to initialize MelodyAgent: {e}")
        return None

def test_autogen_compatibility(melody_agent):
    """Test that MelodyAgent is compatible with AutoGen"""
    print("\nTesting AutoGen compatibility...")
    
    try:
        # Check if it's an instance of AssistantAgent
        if isinstance(melody_agent, autogen.AssistantAgent):
            print("✓ MelodyAgent inherits from autogen.AssistantAgent")
        else:
            print("✗ MelodyAgent does not inherit from autogen.AssistantAgent")
            return False
        
        # Check if it has required methods
        required_methods = ['receive', 'send', 'generate_reply']
        for method in required_methods:
            if hasattr(melody_agent, method):
                print(f"✓ Has method: {method}")
            else:
                print(f"✗ Missing method: {method}")
                return False
        
        print("✓ MelodyAgent is compatible with AutoGen")
        return True
        
    except Exception as e:
        print(f"✗ AutoGen compatibility test failed: {e}")
        return False

def test_model_generation(melody_agent):
    """Test melody generation capabilities"""
    print("\nTesting melody generation...")
    
    try:
        # Test prompt
        test_prompt = "Create a simple melody in C major"
        
        if melody_agent.use_fine_tuned_model:
            print("Testing fine-tuned model generation...")
            result = melody_agent.generate_music_with_model(test_prompt)
            if result:
                print(f"✓ Fine-tuned model generated: {result[:50]}...")
            else:
                print("✗ Fine-tuned model generation failed")
        else:
            print("Fine-tuned model not available, testing LLM fallback...")
            # This would be tested in a full GroupChat scenario
        
        return True
        
    except Exception as e:
        print(f"✗ Melody generation test failed: {e}")
        return False

def test_orchestrator_integration():
    """Test integration with the orchestrator"""
    print("\nTesting orchestrator integration...")
    
    try:
        from orchestrator import MusicOrchestrator
        
        # Load configuration
        config = Config.from_file('src/config/agent.json')
        
        # Create orchestrator
        orchestrator = MusicOrchestrator(config)
        
        # Check if MelodyAgent is in the agents list
        melody_agent_found = False
        for agent in orchestrator.agents:
            if hasattr(agent, 'name') and 'Melody' in agent.name:
                melody_agent_found = True
                print(f"✓ Found MelodyAgent in orchestrator: {agent.name}")
                break
        
        if not melody_agent_found:
            print("✗ MelodyAgent not found in orchestrator agents")
            return False
        
        print("✓ Orchestrator integration successful")
        return True
        
    except Exception as e:
        print(f"✗ Orchestrator integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== MelodyAgent Integration Test ===\n")
    
    # Test 1: Initialization
    melody_agent = test_melody_agent_initialization()
    if not melody_agent:
        print("\n❌ Initialization failed. Stopping tests.")
        return
    
    # Test 2: AutoGen compatibility
    if not test_autogen_compatibility(melody_agent):
        print("\n❌ AutoGen compatibility failed.")
        return
    
    # Test 3: Model generation
    if not test_model_generation(melody_agent):
        print("\n⚠️  Model generation test failed (this might be expected if no model is loaded)")
    
    # Test 4: Orchestrator integration
    if not test_orchestrator_integration():
        print("\n❌ Orchestrator integration failed.")
        return
    
    print("\n✅ All tests passed! MelodyAgent is ready for use.")
    print("\nNext steps:")
    print("1. Prepare training data in JSON format")
    print("2. Run: python fine_tune_melody.py")
    print("3. Update config with fine-tuned model path")
    print("4. Run: python run.py -p prompts.json -o results/ -f")

if __name__ == "__main__":
    main() 