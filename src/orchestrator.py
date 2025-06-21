import autogen
import json
import os
from typing import Dict, Any, List
from config.config import Config
from agents.melody_agent import MelodyAgent

class MusicOrchestrator:
    def __init__(self, config: Config, fine_tune: bool = False):
        self.config = config
        # Configure LLM settings for all agents
        self.llm_config = {
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
        
        # Load system messages from JSON file
        with open('config/sysmsg.json', 'r') as f:
            self.system_messages = json.load(f)
        
        # Initialize AutoGen agents
        self.user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human client.",
            code_execution_config=False,
            llm_config=self.llm_config
        )
        
        Leader = autogen.AssistantAgent(
            name="Leader",
            system_message=self.system_messages["leader"],
            llm_config=self.llm_config
        )

        # Initialize Melody Agent with fine-tuned model if specified
        if fine_tune:
            # Use fine-tuned Hugging Face model for melody agent
            Melody = MelodyAgent(
                model_alias="MelodyAgent",
                model_name=self.config.melody_model.model_name,
                device=self.config.melody_model.device,
                llm_config=self.llm_config,
                config=self.config,
            )
        else:
            Melody = autogen.AssistantAgent(
                name="MelodyAgent",
                system_message=self.system_messages["melody"],
                llm_config=self.llm_config,
            )

        # Harmony = autogen.AssistantAgent(
        #     name="HarmonyAgent",
        #     system_message=self.system_messages["harmony"],
        #     llm_config=self.llm_config,
        # )
        
        # Instrument = autogen.AssistantAgent( 
        #     name="InstrumentAgent",
        #     system_message=self.system_messages["instrument"],
        #     llm_config=self.llm_config
        # )
        
        Arrangement = autogen.AssistantAgent(
            name="ArrangementAgent",
            system_message=self.system_messages["arrangement"],
            llm_config=self.llm_config
        )
        
        Reviewer = autogen.AssistantAgent(
            name="ReviewerAgent",
            system_message=self.system_messages["reviewer"],
            llm_config=self.llm_config
        )
        
        # Store agents for easy reuse
        self.agents = [self.user_proxy, Leader, Melody, Arrangement, Reviewer]
    
    def extract_title(self, abc_notation: str) -> str:
        """Extract title from ABC notation"""
        for line in abc_notation.split('\n'):
            if line.startswith('T:'):
                return line[2:].strip().replace(' ', '_') + '.abc'
        return "Untitled.abc"
    
    def process_request(self, prompt: str) -> str:
        """Process a music generation request"""
        # Create a fresh group chat for each prompt to avoid cross-talk and to allow multiple rounds
        groupchat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=8,
            speaker_selection_method="round_robin"
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
            human_input_mode=None
        )
        
        # Initialize chat with the prompt
        chat_result = self.user_proxy.initiate_chat(
            manager,
            message=prompt
        )
        
        # Get the final response from the last message in the chat
        if chat_result and chat_result.chat_history:
            return chat_result.chat_history[-1]["content"]
        return None
    
    def run_music_generation(self, prompt: str, results_dir: str) -> str:
        """Run the music generation process for a single prompt"""
        chat_log_file = os.path.join(results_dir, 'chat_log.txt')
        
        # Log the prompt
        header = f"\n\n--- Prompt: {prompt} ---\n\n"
        with open(chat_log_file, "a") as f:
            f.write(header)
        
        try:
            # Process the request using the orchestrator
            response = self.process_request(prompt)
            if not response:
                print("No response received from the agents")
                return None
            # Log the response
            with open(chat_log_file, "a") as f:
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
        except Exception as e:
            print(f"Error during generation: {e}")
            return None 