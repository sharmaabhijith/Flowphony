import json
import os
import sys
import autogen
import asyncio
import argparse
from orchestrator import MusicOrchestrator
from config.config import Config
from utils.converter import convert_abc_to_wav

def parse_arguments():
    parser = argparse.ArgumentParser(description="Welcome to ComposerX, a multi-agent based text-to-music generation system.")
    parser.add_argument("--prompts_file", "-p", type=str, required=True, help="Path to the JSON file containing multiple prompts.")
    parser.add_argument("--results_dir", "-o", type=str, required=True, help="Directory to store the results.")
    parser.add_argument("--fine_tune", "-f", action="store_true", help="Use fine-tuned models if available.")
    parser.add_argument("--melody_model_path", "-m", type=str, help="Path to fine-tuned melody model (Hugging Face format).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    results_dir = args.results_dir
    prompts_file = args.prompts_file
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load configuration and initialize orchestrator
    config = Config.from_file('src/config/agent.json')
    
    # Initialize orchestrator with fine-tuned model if specified
    if args.fine_tune and args.melody_model_path:
        orchestrator = MusicOrchestrator(
            config, 
            fine_tune=True, 
            melody_model_path=args.melody_model_path
        )
        print(f"Using fine-tuned melody model: {args.melody_model_path}")
    else:
        orchestrator = MusicOrchestrator(config, fine_tune=False)
        print("Using regular AutoGen agents")

    # Load prompts from JSON file
    with open(prompts_file, "r") as file:
        prompts = json.load(file)

    # Process each prompt
    for index, prompt in enumerate(prompts):
        print(f"Prompt {index + 1}: {prompt['prompt']}")
        print("================================================")
        abc_filepath = orchestrator.run_music_generation(prompt['prompt'], results_dir)
        # if abc_filepath:
        #     wav_filename = convert_abc_to_wav(abc_filepath, results_dir)
        #     if wav_filename:
        #         print(f"Generated WAV file: {wav_filename}")

if __name__ == "__main__":
    main()