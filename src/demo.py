import json
import os
import sys
import asyncio
import argparse
from orchestrator import MusicOrchestrator
from agents.helpers.config import Config
from utils.converter import convert_abc_to_wav
    

def main():
    parser = argparse.ArgumentParser(description="Welcome to ComposerX, a multi-agent based text-to-music generation system.")
    parser.add_argument("--prompts_file", "-p", type=str, required=True, help="Path to the JSON file containing multiple prompts.")
    parser.add_argument("--results_dir", "-o", type=str, required=True, help="Directory to store the results.")
    parser.add_argument("--fine_tune", "-f", type=dict, default={"melody": True, "harmony": False}, help="Use fine-tuned models if available.")
    args = parser.parse_args()

    results_dir = args.results_dir
    prompts_file = args.prompts_file
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    config = Config.from_file('agents/helpers/config.yaml')
    orchestrator = MusicOrchestrator(config, args.fine_tune)

    # Load prompts from JSON file
    with open(prompts_file, "r") as file:
        prompts = json.load(file)
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