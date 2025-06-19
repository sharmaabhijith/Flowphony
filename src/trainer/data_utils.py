"""
Data utilities for online GFlowNet training

This module provides functions for generating melody prompts and managing training data
without requiring any external dataset.
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path

def load_training_prompts(prompts_file: str) -> List[str]:
    """Load training prompts from a JSON file"""
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # If it's a list of dictionaries with 'prompt' key
        if all(isinstance(item, dict) and 'prompt' in item for item in data):
            return [item['prompt'] for item in data]
        # If it's a list of strings
        elif all(isinstance(item, str) for item in data):
            return data
    elif isinstance(data, dict) and 'prompts' in data:
        return data['prompts']
    
    raise ValueError("Invalid prompts file format. Expected list of strings or list of dicts with 'prompt' key")

def generate_melody_prompts(num_prompts: int = 20) -> List[str]:
    """
    Generate diverse melody prompts for online training
    
    Args:
        num_prompts: Number of prompts to generate
        
    Returns:
        List of melody generation prompts
    """
    
    # Musical styles and genres
    styles = [
        "folk", "jazz", "classical", "blues", "country", "rock", "pop", 
        "electronic", "ambient", "lullaby", "march", "waltz", "ballad",
        "upbeat", "melancholic", "energetic", "peaceful", "dramatic"
    ]
    
    # Musical keys
    keys = [
        "C major", "G major", "D major", "A major", "E major", "B major",
        "F major", "Bb major", "Eb major", "Ab major", "Db major", "Gb major",
        "A minor", "E minor", "B minor", "F# minor", "C# minor", "G# minor",
        "D minor", "G minor", "C minor", "F minor", "Bb minor", "Eb minor"
    ]
    
    # Instruments
    instruments = [
        "piano", "guitar", "violin", "flute", "clarinet", "saxophone",
        "trumpet", "cello", "harp", "accordion", "mandolin", "banjo",
        "organ", "synthesizer", "acoustic guitar", "electric guitar"
    ]
    
    # Tempos
    tempos = [
        "slow", "moderate", "fast", "lively", "relaxed", "energetic",
        "gentle", "brisk", "leisurely", "upbeat", "calm", "dynamic"
    ]
    
    # Moods and emotions
    moods = [
        "happy", "sad", "peaceful", "excited", "romantic", "mysterious",
        "joyful", "contemplative", "energetic", "serene", "passionate",
        "nostalgic", "hopeful", "melancholic", "cheerful", "dramatic"
    ]
    
    # Prompt templates
    templates = [
        "Compose a {style} melody in {key} with {tempo} tempo, suitable for {instrument}",
        "Create a {style} tune in {key} with {mood} mood, featuring {instrument}",
        "Write a {style} piece in {key} with {tempo} tempo, {mood} feeling",
        "Generate a {style} melody in {key} for {instrument}, {mood} atmosphere",
        "Compose a {style} song in {key} with {tempo} pace, {mood} character",
        "Create a {style} composition in {key}, {mood} and {tempo}",
        "Write a {style} melody in {key} with {instrument}, {mood} tone",
        "Generate a {style} piece in {key}, {tempo} and {mood}",
        "Compose a {style} tune in {key} for {instrument}, {mood} style",
        "Create a {style} melody in {key} with {tempo} rhythm, {mood} mood"
    ]
    
    prompts = []
    
    for i in range(num_prompts):
        # Randomly select components
        style = random.choice(styles)
        key = random.choice(keys)
        instrument = random.choice(instruments)
        tempo = random.choice(tempos)
        mood = random.choice(moods)
        template = random.choice(templates)
        
        # Generate prompt
        prompt = template.format(
            style=style,
            key=key,
            instrument=instrument,
            tempo=tempo,
            mood=mood
        )
        
        prompts.append(prompt)
    
    return prompts

def generate_specialized_prompts(num_prompts: int = 10, style: str = None) -> List[str]:
    """
    Generate specialized prompts for a particular musical style
    
    Args:
        num_prompts: Number of prompts to generate
        style: Specific musical style (optional)
        
    Returns:
        List of specialized melody prompts
    """
    
    if style:
        # Style-specific templates
        style_templates = {
            "folk": [
                "Compose a traditional folk melody in {key} with {tempo} tempo",
                "Create a folk song in {key} with {mood} feeling",
                "Write a folk tune in {key} suitable for {instrument}"
            ],
            "jazz": [
                "Compose a jazz melody in {key} with swing rhythm",
                "Create a jazz tune in {key} with {mood} mood",
                "Write a jazz piece in {key} for {instrument}"
            ],
            "classical": [
                "Compose a classical melody in {key} with {tempo} tempo",
                "Create a classical piece in {key} with {mood} character",
                "Write a classical composition in {key} for {instrument}"
            ],
            "blues": [
                "Compose a blues melody in {key} with soulful feeling",
                "Create a blues tune in {key} with {mood} mood",
                "Write a blues piece in {key} for {instrument}"
            ]
        }
        
        templates = style_templates.get(style.lower(), [
            f"Compose a {style} melody in {{key}} with {{tempo}} tempo",
            f"Create a {style} tune in {{key}} with {{mood}} feeling"
        ])
    else:
        templates = [
            "Compose a melody in {key} with {tempo} tempo",
            "Create a tune in {key} with {mood} mood",
            "Write a piece in {key} for {instrument}"
        ]
    
    keys = ["C major", "G major", "D major", "A minor", "E minor", "F major"]
    tempos = ["slow", "moderate", "fast", "lively"]
    moods = ["happy", "sad", "peaceful", "energetic", "romantic"]
    instruments = ["piano", "guitar", "violin", "flute", "saxophone"]
    
    prompts = []
    
    for i in range(num_prompts):
        key = random.choice(keys)
        tempo = random.choice(tempos)
        mood = random.choice(moods)
        instrument = random.choice(instruments)
        template = random.choice(templates)
        
        prompt = template.format(
            key=key,
            tempo=tempo,
            mood=mood,
            instrument=instrument
        )
        
        prompts.append(prompt)
    
    return prompts

def create_prompt_variations(base_prompt: str, num_variations: int = 5) -> List[str]:
    """
    Create variations of a base prompt
    
    Args:
        base_prompt: The base prompt to vary
        num_variations: Number of variations to create
        
    Returns:
        List of prompt variations
    """
    
    # Simple variations by changing key words
    variations = []
    
    # Key variations
    keys = ["C major", "G major", "D major", "A minor", "E minor"]
    
    # Tempo variations
    tempos = ["slow", "moderate", "fast", "lively"]
    
    # Mood variations
    moods = ["happy", "sad", "peaceful", "energetic", "romantic"]
    
    for i in range(num_variations):
        variation = base_prompt
        
        # Replace key if present
        for key in keys:
            if key in variation:
                variation = variation.replace(key, random.choice(keys))
                break
        
        # Replace tempo if present
        for tempo in tempos:
            if tempo in variation:
                variation = variation.replace(tempo, random.choice(tempos))
                break
        
        # Replace mood if present
        for mood in moods:
            if mood in variation:
                variation = variation.replace(mood, random.choice(moods))
                break
        
        variations.append(variation)
    
    return variations

def save_prompts_to_file(prompts: List[str], filepath: str):
    """
    Save prompts to a JSON file
    
    Args:
        prompts: List of prompts to save
        filepath: Path to save the file
    """
    with open(filepath, 'w') as f:
        json.dump(prompts, f, indent=2)

def load_prompts_from_file(filepath: str) -> List[str]:
    """
    Load prompts from a JSON file
    
    Args:
        filepath: Path to the prompts file
        
    Returns:
        List of prompts
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'prompts' in data:
        return data['prompts']
    else:
        raise ValueError("Invalid prompts file format")

def validate_prompt(prompt: str) -> bool:
    """
    Validate if a prompt is suitable for melody generation
    
    Args:
        prompt: The prompt to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not prompt or len(prompt.strip()) < 10:
        return False
    
    # Check for basic melody-related keywords
    melody_keywords = [
        "melody", "tune", "song", "piece", "composition", "music",
        "compose", "create", "write", "generate", "play"
    ]
    
    has_melody_keyword = any(keyword in prompt.lower() for keyword in melody_keywords)
    
    # Check for musical elements
    musical_elements = [
        "major", "minor", "key", "tempo", "rhythm", "chord", "note",
        "piano", "guitar", "violin", "instrument", "style", "mood"
    ]
    
    has_musical_element = any(element in prompt.lower() for element in musical_elements)
    
    return has_melody_keyword or has_musical_element

def filter_valid_prompts(prompts: List[str]) -> List[str]:
    """
    Filter out invalid prompts
    
    Args:
        prompts: List of prompts to filter
        
    Returns:
        List of valid prompts
    """
    return [prompt for prompt in prompts if validate_prompt(prompt)]

def create_abc_prompt(prompt: str) -> str:
    """Convert a natural language prompt to an ABC notation prompt"""
    
    # Extract musical parameters from the prompt
    key_match = re.search(r'in\s+([A-G][#b]?\s+(?:major|minor))', prompt, re.IGNORECASE)
    tempo_match = re.search(r'(\w+)\s+tempo', prompt, re.IGNORECASE)
    rhythm_match = re.search(r'(\d+/\d+)', prompt)
    
    # Default values
    key = key_match.group(1) if key_match else "C major"
    tempo = tempo_match.group(1) if tempo_match else "moderate"
    meter = rhythm_match.group(1) if rhythm_match else "4/4"
    
    # Convert key to ABC notation
    key_mapping = {
        "C major": "C", "G major": "G", "D major": "D", "A major": "A", "E major": "E", "B major": "B",
        "F major": "F", "Bb major": "Bb", "Eb major": "Eb", "Ab major": "Ab", "Db major": "Db", "Gb major": "Gb",
        "A minor": "Am", "E minor": "Em", "B minor": "Bm", "F# minor": "F#m", "C# minor": "C#m", "G# minor": "G#m",
        "D minor": "Dm", "G minor": "Gm", "C minor": "Cm", "F minor": "Fm", "Bb minor": "Bbm", "Eb minor": "Ebm"
    }
    
    abc_key = key_mapping.get(key, "C")
    
    # Create ABC header
    abc_prompt = f"""X:1
T:Generated Melody
C:AI Composer
M:{meter}
L:1/8
K:{abc_key}
|:"""
    
    return abc_prompt

def split_train_val(prompts: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split prompts into training and validation sets"""
    random.seed(seed)
    shuffled_prompts = prompts.copy()
    random.shuffle(shuffled_prompts)
    
    split_idx = int(len(shuffled_prompts) * (1 - val_ratio))
    train_prompts = shuffled_prompts[:split_idx]
    val_prompts = shuffled_prompts[split_idx:]
    
    return train_prompts, val_prompts

def save_prompts(prompts: List[str], filepath: str):
    """Save prompts to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(prompts, f, indent=2)

def load_existing_abc_data(data_dir: str) -> List[str]:
    """Load existing ABC notation data for training"""
    abc_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.abc'):
                filepath = os.path.join(data_dir, file)
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        abc_files.append(content)
    
    return abc_files

def extract_abc_from_response(response: str) -> str:
    """Extract ABC notation from a response string"""
    # Look for ABC notation in markdown code blocks
    abc_pattern = r'```(?:abc)?\s*\n(.*?)\n```'
    match = re.search(abc_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Look for ABC notation without markdown
    abc_lines = []
    lines = response.split('\n')
    in_abc = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('X:') or line.startswith('T:') or line.startswith('K:'):
            in_abc = True
        
        if in_abc:
            abc_lines.append(line)
            
            # End if we see a closing barline
            if line.endswith(':|') or line.endswith('|]'):
                break
    
    if abc_lines:
        return '\n'.join(abc_lines)
    
    return ""

def validate_abc_notation(abc_text: str) -> bool:
    """Validate if the text contains valid ABC notation"""
    if not abc_text:
        return False
    
    # Check for basic ABC structure
    has_header = any(line.startswith('X:') for line in abc_text.split('\n'))
    has_key = any(line.startswith('K:') for line in abc_text.split('\n'))
    has_music = any('|' in line for line in abc_text.split('\n'))
    
    return has_header and has_key and has_music

def create_training_dataset(prompts: List[str], output_file: str):
    """Create a training dataset file with prompts and expected ABC structure"""
    dataset = []
    
    for i, prompt in enumerate(prompts):
        abc_prompt = create_abc_prompt(prompt)
        
        dataset.append({
            "id": i,
            "prompt": prompt,
            "abc_prompt": abc_prompt,
            "expected_structure": {
                "header": True,
                "key_signature": True,
                "time_signature": True,
                "melody_line": True,
                "closing_barline": True
            }
        })
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created training dataset with {len(dataset)} samples: {output_file}") 