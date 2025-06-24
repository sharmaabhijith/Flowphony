import torch
from typing import Tuple, List
from .helpers.config import Config
from .helpers.base_agent import BaseAgent
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class MelodyAgent(BaseAgent):
    def __init__(self, config: Config, agent_name: str = "melody", fine_tune: bool = True):
        super().__init__(config, agent_name)
        self.fine_tune = fine_tune
        
        if self.fine_tune:
            # Use fine-tuned model
            self.device = self.config.device
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
        else:
            # Use DeepInfra API
            print(f"Using DeepInfra API for melody generation with model: {self.config.model_name}")
            self.model = None
            self.tokenizer = None
        
        self.valid_notes = list("ABCDEFGabcdefg")
        self.valid_rests = ['z']
        self.valid_durations = list("12345678")
        self.confidence = []

    def setup_lora(self):
        """Setup LoRA for the melody agent's model"""
        if not self.fine_tune:
            print("LoRA setup not available when using DeepInfra API")
            return
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

    def _next_note(self, current_melody: str) -> Tuple[str, float]:
        """Generate the next note and get its probability using the fine-tuned model"""
        if not self.fine_tune:
            # Use DeepInfra API for next note generation
            prompt = f"""
            Given the current melody in ABC notation: {current_melody}
            
            Generate the next note in ABC notation. The note should be one of:
            - A single note (A, B, C, D, E, F, G, a, b, c, d, e, f, g)
            - A rest (z)
            - A note with duration (e.g., A4, B2, z4)
            - A bar line (|)
            - End repeat (:|)
            
            Return only the next note/duration/bar line, nothing else.
            """
            
            messages = self.get_conversation_context()
            messages.append({"role": "user", "content": prompt})
            
            response = self.call_deepinfra_api(messages)
            # Extract the note from the response
            note = response.strip().split()[0] if response.strip() else "z4"
            return note, 0.8  # Default confidence for API responses
        
        # Use fine-tuned model
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
    
    def generate_melody(self, prompt: str) -> Tuple[str, List[float]]:
        """Generate melody using the fine-tuned model or DeepInfra API"""
        if not self.fine_tune:
            # Use DeepInfra API for melody generation
            melody_prompt = f"""
            Create a complete melody in ABC notation based on the following prompt:
            
            {prompt}
            
            The melody should:
            1. Be in proper ABC notation format
            2. Include a title (T:)
            3. Include a key signature (K:)
            4. Include a meter (M:)
            5. Have a reasonable length (4-8 measures)
            6. End with a repeat bar (:|)
            
            Return the complete ABC notation wrapped in ```abc``` code blocks.
            """
            
            messages = self.get_conversation_context()
            messages.append({"role": "user", "content": melody_prompt})
            
            response = self.call_deepinfra_api(messages)
            # For API responses, we'll use a default confidence
            self.confidence = [0.8] * 10  # Default confidence for API responses
            return response,
        
        # Use fine-tuned model
        current_melody = prompt
        max_notes = 128  # Limit for melody generation
        for _ in range(max_notes):
            next_note, confidence = self._next_note(current_melody)
            current_melody += next_note
            self.confidence.append(confidence)
            if (next_note in [":|", "|]"] or 
                len(current_melody) > 200 or  # Increased length limit
                current_melody.count("|") > 8):  # Limit number of measures
                break
        if not current_melody.endswith(":|"):
            current_melody += ":|"

        return (f"```abc\n{current_melody.strip()}\n```", self.confidence)