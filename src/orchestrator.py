import json
import os
from typing import Dict, Any, List
from agents.helpers.config import Config
from agents.leader_agent import LeaderAgent
from agents.melody_agent import MelodyAgent
from agents.harmony_agent import HarmonyAgent
from agents.instrument_agent import InstrumentAgent
from agents.arrangement_agent import ArrangementAgent
from agents.reviewer_agent import ReviewerAgent

class MusicOrchestrator:
    def __init__(self, config: Config, fine_tune: dict):

        self.leader = LeaderAgent(config.api_model)
        self.harmony = HarmonyAgent(config.api_model)
        self.instrument = InstrumentAgent(config.api_model)
        self.arrangement = ArrangementAgent(config.api_model)
        self.reviewer = ReviewerAgent(config.api_model)

        melody_config = config.melody_model if fine_tune["melody"] else config.api_model
        self.melody = MelodyAgent(melody_config)
        
        self.agents = {
            "leader": self.leader,
            "melody": self.melody,
            "harmony": self.harmony,
            "arrangement": self.arrangement,
            "reviewer": self.reviewer
        }
    
    def extract_title(self, abc_notation: str) -> str:
        """Extract title from ABC notation"""
        for line in abc_notation.split('\n'):
            if line.startswith('T:'):
                return line[2:].strip().replace(' ', '_') + '.abc'
        return "Untitled.abc"
    
    def sample_music(self, prompt: str) -> dict:
        """Run sequential music generation process"""
        
        print(f"\n=== Starting Music Generation Process ===")
        print(f"Initial prompt: {prompt}")
        print(f"\n--- Step 1: Leader Analysis ---")
        leader_response = self.leader.create_melody_prompt(prompt)
        print(f"\n--- Step 2: Melody Composition ---")
        melody_response, confidence = self.melody.generate_melody(leader_response)
        print(f"\n--- Step 3: Harmony Addition ---")
        harmony_response = self.harmony.add_harmony_to_melody(melody_response)
        print(f"\n--- Step 4: Instrument Addition ---")
        instrument_response = self.instrument.add_instrument_to_harmonic_melody(harmony_response)
        print(f"\n--- Step 5: Music Arrangement ---")
        arrangement_response = self.arrangement.arrange_music(instrument_response)

        final_response = {
            "music": arrangement_response,
            "confidence": confidence
        }
        return final_response
    
    
    def run_music_generation(self, prompt: str, results_dir: str) -> str:
        """Run the music generation process for a single prompt"""
        chat_log_file = os.path.join(results_dir, 'chat_log.txt')
        
        # Log the prompt
        header = f"\n\n--- Prompt: {prompt} ---\n\n"
        response = self.sample_music(prompt)["music"]

        with open(chat_log_file, "a") as f:
            f.write(header)
            f.write(response)
        # Extract ABC notation from the response
        i = -1
        abc_notation = ""
        while "|" not in abc_notation:
            abc_notation = response.split("```")[i] if "```" in response else response
            i -= 1
        abc_notation = os.linesep.join([s for s in abc_notation.splitlines() if s])
        abc_filename = self.extract_title(abc_notation)
        abc_filepath = os.path.join(results_dir, abc_filename)
        with open(abc_filepath, 'w') as f:
            f.write(abc_notation)
        
        return abc_filepath