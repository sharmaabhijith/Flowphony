from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .helpers.config import Config
from .helpers.base_agent import BaseAgent

class MelodyAgent(BaseAgent):
    def __init__(self, config: Config, agent_name: str = "melody"):
        super().__init__(config, agent_name)
        self.device = self.config.device

        try:
            print(f"Loading model with 4-bit quantization: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            print("✓ Tokenizer loaded successfully")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            print(f"✓ 4-bit quantized model loaded successfully from {self.config.model_name}")
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU memory used: {memory_used:.2f} GB")
        except Exception as e:
            print(f"❌ Error loading 4-bit quantized model: {e}")
            print(f"Error type: {type(e).__name__}")
            print("Falling back to LLM-based melody generation")
        
        self.valid_notes = list("ABCDEFGabcdefg")
        self.valid_rests = ['z']
        self.valid_durations = list("12345678")

    
    def _next_note(self, current_melody: str) -> Tuple[str, float]:
        """Generate the next note and get its probability using the fine-tuned model"""
        inputs = self.tokenizer(
            current_melody,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)
            tok_ids = top_indices.tolist()
            top_tokens = self.tokenizer.convert_ids_to_tokens(tok_ids)
            for token, prob in zip(top_tokens, top_probs.tolist()):
                valid_tokens = self.valid_notes + self.valid_rests
                if any(token.startswith(note) for note in valid_tokens):
                    return token.strip(), prob
        return "z4", 0.1
    
    def _generate_music(self, prompt: str) -> str:
        """Generate melody using the fine-tuned model"""
        current_melody = prompt
        max_notes = 64  # Limit for melody generation
        for _ in range(max_notes):
            next_note, confidence = self._next_note(current_melody)
            current_melody += next_note
            if (next_note in [":|", "|]"] or 
                len(current_melody) > 200 or  # Increased length limit
                current_melody.count("|") > 8):  # Limit number of measures
                break
        if not current_melody.endswith(":|"):
            current_melody += ":|"
        return current_melody.strip()
    
    def generate_response(self, message: str) -> str:
        """
        Generate a response to the given message.
        If it's a music request, use the fine-tuned model; otherwise use DeepInfra API.
        """
        melody = self._generate_music(message) or "z4"
        return f"```abc\n{melody}\n```"