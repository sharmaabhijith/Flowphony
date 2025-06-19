from pathlib import Path
import subprocess
import os
from typing import Optional

def convert_abc_to_wav(abc_file_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Convert an ABC notation file to WAV format using abc2midi and MuseScore.
    
    Args:
        abc_file_path: Path to the ABC notation file
        output_dir: Directory to save the output files (defaults to same directory as input)
    
    Returns:
        Path to the generated WAV file
    """
    if output_dir is None:
        output_dir = abc_file_path.parent
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert ABC to MIDI
    midi_file = output_dir / f"{abc_file_path.stem}.mid"
    try:
        subprocess.run(
            ["abc2midi", str(abc_file_path), "-o", str(midi_file)],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert ABC to MIDI: {e.stderr.decode()}")
    
    # Convert MIDI to WAV using MuseScore
    wav_file = output_dir / f"{abc_file_path.stem}.wav"
    try:
        # Try different possible MuseScore paths
        musescore_paths = [
            "/Applications/MuseScore 4.app/Contents/MacOS/mscore",  # macOS
            "C:/Program Files/MuseScore 4/bin/MuseScore4.exe",      # Windows
            "C:/Program Files (x86)/MuseScore 4/bin/MuseScore4.exe" # Windows (x86)
        ]
        
        musescore_found = False
        for path in musescore_paths:
            if os.path.exists(path):
                subprocess.run(
                    [path, "-f", "-o", str(wav_file), str(midi_file)],
                    check=True,
                    capture_output=True
                )
                musescore_found = True
                break
        
        if not musescore_found:
            raise FileNotFoundError("MuseScore executable not found")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert MIDI to WAV: {e.stderr.decode()}")
    
    return wav_file 