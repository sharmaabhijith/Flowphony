from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import requests
import json
import os
import yaml
from openai import OpenAI
from .config import Config

class BaseAgent(ABC):
    """Base class for all custom agents with send/receive communication"""
    
    def __init__(self, config: Config, agent_name: str = None):
        self.config = config
        self.model_name = agent_name
        with open('agents/helpers/SYSMSG.yaml', 'r') as f:
            sysmsg_data = yaml.safe_load(f)
            self.system_message = sysmsg_data[self.model_name.lower()]
        self.conversation_history: List[Dict[str, str]] = []
        self.client = OpenAI(api_key=os.getenv('DEEPINFRA_API_KEY'), base_url=self.config.base_url)

    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation context for API calls"""
        messages = [{"role": "system", "content": self.system_message}]
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return messages 
    
    def send(self, message: str, recipient: 'BaseAgent', silent: bool = False) -> None:
        """Send a message to another agent"""
        if not silent:
            print(f"[{self.model_name}] -> [{recipient.model_name}]: {message[:100]}...")
        self.conversation_history.append({
            "role": "assistant",
            "content": message,
            "sender": self.model_name,
            "recipient": recipient.model_name
        })
        recipient.receive(message, self)
    
    def receive(self, message: str, sender: 'BaseAgent') -> str:
        """Receive a message from another agent and generate a response"""
        self.conversation_history.append({
            "role": "user", 
            "content": message,
            "sender": sender.model_name,
            "recipient": self.model_name
        })
        response = self.generate_response(message)
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "sender": self.model_name,
            "recipient": sender.model_name
        })
        return response
    
    @abstractmethod
    def generate_response(self, message: str) -> str:
        """Generate a response to the given message - to be implemented by subclasses"""
        pass
    
    def call_deepinfra_api(self, messages: List[Dict[str, str]]) -> str:
        """Call DeepInfra API for inference using OpenAI client"""
        if not self.client:
            return "Error: DEEPINFRA_API_KEY not found in config or environment."
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_message}] + messages
        try:
            print(f"[{self.model_name}] Calling DeepInfra API with model: {self.config.model_name}")
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=getattr(self.config, 'temperature', 0.7),
                max_tokens=getattr(self.config, 'max_length', 2048),
                top_p=0.9
            )
            return response.choices[0].message.content  
        except Exception as e:
            print(f"[{self.model_name}] Error calling DeepInfra API: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"