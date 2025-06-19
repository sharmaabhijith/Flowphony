# 🎶 Flowphony: Diverse Orchestration of Symbolic Music Symphony Using GFlowNet-Tuned Composing Agents

**Flowphony** is a multi-agent symbolic music generation system that composes diverse and coherent music using **GFlowNet** based finetuning for melody and harmony agents. Inspired by the collaborative architecture of ComposerX, Flowphony is designed for structured, high-quality symbolic composition in formats like **ABC notation** or **MIDI tokens**.

---

## 🧠 Core Idea

Flowphony focuses on the **structured sampling of diverse musical ideas** by fine-tuning **Melody** and **Harmony** agents with **Generative Flow Networks (GFlowNets)**. This enables it to:
- Generate **multiple diverse yet high-reward melodies** from a given theme
- Harmonize melodies in **varied emotional styles**
- Maintain structure and style consistency of music of interest
---

# Flowphony: GFlowNet-based Music Generation

A music generation system using GFlowNet for fine-tuning melody and harmony generation models.

## Overview

This project implements a GFlowNet-based approach to music generation, focusing on melody and harmony generation. The system uses GFlowNet to learn the underlying structure of musical compositions and generate new pieces that follow similar patterns.

## Features

- GFlowNet-based training for music generation
- Support for ABC notation
- Melody and harmony generation
- Model fine-tuning capabilities
- Comprehensive logging and monitoring
- Easy-to-use training pipeline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flowphony.git
cd flowphony
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional requirements:
- abc2midi (for ABC to MIDI conversion)
- MuseScore (for MIDI to WAV conversion)

## Usage

### Training

To train the GFlowNet model:

```bash
python src/train.py \
    --data_dir path/to/abc/files \
    --output_dir path/to/output \
    --vocab_size 256 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

### Generation

To generate music using the trained model:

```bash
python src/main.py \
    --prompt path/to/prompt.json \
    --output_dir path/to/output
```

## Project Structure

```
src/
├── agents/           # Agent implementations
├── models/           # Model implementations
│   └── gflownet.py   # GFlowNet implementation
├── utils/            # Utility functions
├── config/           # Configuration files
├── data/            # Data directory
├── main.py          # Main generation script
└── train.py         # Training script
```

## GFlowNet Training

The GFlowNet training process consists of several key components:

1. **Data Preparation**
   - ABC files are converted to tensor representations
   - State transitions are created for training
   - Rewards are computed based on musical properties

2. **Model Architecture**
   - Policy network for action selection
   - Flow network for state value estimation
   - Reward function for musical quality assessment

3. **Training Process**
   - Flow matching loss computation
   - Policy optimization
   - Checkpoint saving and loading

## Customization

### Reward Function

The reward function in `src/train.py` can be customized to consider various musical properties:

- Harmonic coherence
- Melodic flow
- Style consistency
- Musical validity

### Model Architecture

The GFlowNet architecture in `src/models/gflownet.py` can be modified to:

- Change network depth and width
- Add attention mechanisms
- Incorporate musical knowledge

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
