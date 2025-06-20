from typing import Dict, Any, List, Tuple
import torch
import autogen
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config.config import Config

SYSTEM_MESSAGE = """

You are a highly skilled melody composer.
Your task is to generate a melody using ABC Notation. The melody will be composed one note at a time.

CRITICAL RULES:
1. You MUST output **only valid ABC Notation** enclosed in a markdown block (```).
2. DO NOT output any explanation, reasoning, or extra text.
3. STRICTLY terminate your output after each note or line of ABC notation.
4. DO NOT include phrases like "Let's choose" or "Here is".
---

FIRST OUTPUT: Start with the ABC Notation header only (no notes yet). Use this exact structure:
```
X:1
T:Title
C:Composer
M:Meter
L:Unit note length
K:Key
```

Example:
```
X:1
T:Balkan Brass Extravaganza
C:Your Name
M:7/8
L:1/8
K:G
```
After this header, wait for the next input. Each subsequent output must add exactly **one new note** in ABC notation format.
Start your melody with a barline `|:` and write only one note per response, like this:
```
|:G
```
When the system prompts "NEXT NOTE" with the current melody, you must append **exactly one new note**, like:
```
|:GA
```

ADDITIONAL ABC NOTATION SYMBOL RULES:

1. '|:' and ':|' are repeat barlines, where '|:' marks the beginning and ':|' marks the end of a repeated section. Use `|:` only at the start of your melody and `:|` at the end.
2. '|]' is a final barline that can be used instead of ':|' if the melody is not repeated. It indicates the **absolute end** of the piece.
3. '|' is a standard barline that separates measures. Use it periodically to divide the melody according to the time signature (e.g., every 4 beats in 4/4).
4. 'z' represents a **rest** (a silence) to insert a pause in the melody. It counts as one unit of the note length (L:1/8 by default). Example: `z`, `z2`, `z/2` for rest of different durations.
All symbols must be used accurately to preserve proper musical and structural meaning.

STRICTLY ensure that new note generated is in the same scale of the key.
Continue this until the melody is complete (~30 seconds long). End the melody with closing `:|` and full formatting.

Final output format:
```
X:1
T:...
C:...
M:...
L:...
K:...
|:...:|
```

DO NOT generate anything else. No headers, no footers, no explanations.
Your output must **only** contain valid ABC notation in a markdown code block. No helper text.

"""

class MelodyAgent(autogen.AssistantAgent):
    def __init__(
        self, 
        model_alias: str = "MelodyAgent",
        model_name: str = None,
        device: str = "cuda",
        llm_config: Dict[str, Any] | None = None,
        config: Config = None,
        ):
        super().__init__(
        name=model_alias,
        system_message=SYSTEM_MESSAGE,
        llm_config=llm_config,
        is_termination_msg=lambda m: isinstance(m, str) and m.strip().endswith("|]"),
        )  
        
        # Store configs
        self.config = config
        
        # Initialize fine-tuned model if path is provided
        if model_name:
            try:
                print(f"Loading model with 4-bit quantization: {model_name}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("✓ Tokenizer loaded successfully")
                
                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                # Load model with 4-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                self.device = device
                self.use_fine_tuned_model = True
                print(f"✓ 4-bit quantized model loaded successfully from {model_name}")
                
                # Print memory usage info
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"GPU memory used: {memory_used:.2f} GB")
                
            except Exception as e:
                print(f"❌ Error loading 4-bit quantized model: {e}")
                print(f"Error type: {type(e).__name__}")
                print("Falling back to LLM-based melody generation")
                self.use_fine_tuned_model = False
                self.model = None
                self.tokenizer = None
        else:
            self.use_fine_tuned_model = False
            self.model = None
            self.tokenizer = None
            self.device = device
        
        self.valid_notes = list("ABCDEFGabcdefg")
        self.valid_rests = ['z']
        self.valid_durations = list("12345678")

    
    def _next_note(self, current_melody: str) -> Tuple[str, float]:
        """Generate the next note and get its probability using the fine-tuned model"""
        if not self.use_fine_tuned_model:
            return "z4", 0.1
        if not self.config:
            return "z4", 0.1
        inputs = self.tokenizer(
            current_melody,
            return_tensors="pt",
            max_length=self.config.melody_model.max_length,
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
    
    def _is_music_request(self, text: str) -> bool:
        keywords = (
            "melody", "compose", "music", "tune", "abc notation",
            "note", "harmony", "song", "rhythm", "chord"
        )
        return any(k in text.lower() for k in keywords)
    
    def _generate_music(self, prompt: str) -> str:
        """Generate melody using the fine-tuned model"""
        if not self.use_fine_tuned_model:
            return None
        if not self.config:
            return None
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
    
    # ------------------------------------------------------------------
    # Generate a reply when the framework gives this agent the turn
    # ------------------------------------------------------------------
    def generate_reply(self, messages: List[dict]) -> str:
        """
        AutoGen passes the entire chat history as `messages`.
        We look at the most recent *external* message (i.e. not from ourselves).
        If it's a music request – or explicitly @mentions this agent – we
        generate/extend a melody.  Otherwise we fall back to the parent class.
        """
        # Find the latest message not written by this agent
        last_msg = next(
            (m for m in reversed(messages) if m.get("name") != self.name), None
        )
        prompt = str(last_msg.get("content", "")) if last_msg else ""

        wants_music = self._is_music_request(prompt) or f"@{self.name}" in prompt

        if wants_music:
            melody = self._generate_music(prompt) or "z4"
            # Wrap in a markdown ABC block as required by SYSTEM_MESSAGE
            return f"```abc\n{melody}\n```"

        # Not music-related → delegate to the normal LLM reply
        return super().generate_reply(messages)


    # ------------------------------------------------------------------
    # Intercept direct messages sent TO this agent
    # ------------------------------------------------------------------
    def receive(
        self,
        message: str | dict,
        sender: autogen.Agent,
        request_reply: bool | None = None,
        silent: bool = False,
    ):
        """
        If the incoming message is a music request (or @mentions us),
        respond immediately with a melody; otherwise use the default behaviour.
        """
        content = (
            message.get("content") if isinstance(message, dict) else message
        )
        content = str(content)

        if self._is_music_request(content) or f"@{self.name}" in content:
            melody = self._generate_music(content)
            if melody:
                payload = f"```abc\n{melody}\n```"
                if request_reply:
                    self.send(payload, recipient=sender, silent=silent)
                return payload

        # Fall back to base class handling
        return super().receive(message, sender, request_reply, silent)
