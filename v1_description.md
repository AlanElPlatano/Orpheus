# Simple AI Melody Generator - Minimal Version

## Project Goal
Build the simplest possible AI that can generate a melody over a chord progression using MIDI data.

## What We're Building
- **Input**: MIDI files with sample songs for AI generation
- **Output**: MIDI files generated from training on input files
- **Architecture**: Pytorch AI
- **Dataset**: Hand curated MIDI dataset

## Technical Stack
- **Python 3.10.8+**
- **PyTorch** (for the neural network)
- **pretty_midi** (for MIDI file handling)
- **numpy** (for data processing)
- **miditok** (for tokenization of MIDIs)
- **Gradio** (for GUI building)

## Data Representation
We'll represent music as a sequence of tokens using miditok's REMI strategy with bar tokens (REMI+)

## Phase 1: Data Exploration (Week 1)
**Goal**: Understand what's in MIDI files

### Step 1.1: Download and Look at Data
```bash
# Obtain a portion of the MIDI training dataset
```

### Step 1.2: Basic MIDI Parsing
Write a script that:
- Opens a MIDI file
- Reads basic info (duration, number of tracks, tempo)
- Extracts all notes and their timing
- Tokenizes all this info using Miditok with the REMI+ tokenizer

## Phase 2: Manual Data Preparation (Week 2)
**Goal**: Create a hand-curated training dataset

- Create files with a strict 2 track (melody-chords) structure

### Step 2.2: Basic Tokenization
Convert your hand-picked files into token sequences using Miditok's REMI+ tokenizer:


**Deliverable**: 5-10 files converted to token sequences, saved as JSON files

## Phase 3: Simplest Possible Model (Week 3)
**Goal**: Build an LSTM that can learn to continue a sequence

### Step 3.1: Build Vocabulary
Create mappings:
```python
token_to_id = {
    "CHORD_C": 0, "CHORD_Am": 1, "CHORD_F": 2, "CHORD_G": 3,
    "NOTE_ON_60": 4, "NOTE_OFF_60": 5, 
    "NOTE_ON_62": 6, "NOTE_OFF_62": 7,
    # ... etc for all notes C4-C6 (24 notes total)
    "WAIT_250": 50, "WAIT_500": 51, "WAIT_1000": 52
}
```

### Step 3.2: Basic LSTM Model
```python
class SimpleMelodyLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.output(lstm_out)
```

### Step 3.3: Training Loop
Train the model to predict the next token given previous tokens:
- Input: First N tokens of a sequence
- Target: Token N+1
- Loss: Cross-entropy loss
- Training: Just get it to run without errors first

**Deliverable**: A model that can be trained and generate token sequences (even if nonsensical)

## Phase 4: Generation and Testing (Week 4)
**Goal**: Generate melodies and convert back to MIDI

### Step 4.1: Basic Generation
Write a function that:
- Takes a chord progression as input: `["CHORD_C", "CHORD_Am", "CHORD_F", "CHORD_G"]`
- Uses the trained model to generate melody tokens
- Stops after reasonable length (50-100 tokens)

### Step 4.2: Convert Back to MIDI
Write a function that:
- Takes token sequence and converts back to MIDI events
- Creates a simple 2-track MIDI file (chord track + melody track)
- Saves as a .mid file you can play

### Step 4.3: Listen and Evaluate
- Generate 5-10 melodies
- Listen to them (they'll probably sound bad!)
- Note what's wrong (timing issues, wrong notes, etc.)

**Deliverable**: MIDI files you can actually play and listen to

## Success Criteria (Realistic Expectations)
After these 4 phases, you should have:
- ✅ A working data pipeline (MIDI → tokens → MIDI)
- ✅ A trainable neural network
- ✅ Generated melodies that have correct timing
- ❌ Melodies probably won't sound good yet
- ❌ Model might repeat patterns or generate nonsense
- ❌ Chord-melody relationship will be weak

## What You'll Learn
- How MIDI files work
- How to tokenize sequential data
- Basic LSTM training in PyTorch
- The challenges of music generation
- Where the problems are (data quality, model capacity, evaluation)

## Next Steps (After Minimal Version Works)
Once you have this basic version working, you can improve:
1. **Better data filtering** (find more suitable training files)
2. **Larger vocabulary** (more notes, rhythms, chords)
3. **Better model architecture** (attention, transformers)
4. **Conditioning** (explicit chord input)
5. **Evaluation metrics** (musical quality measures)

## File Structure
```
simple_melody_generator/
├── data/
│   ├── raw_midi/          # Original MIDI files
│   ├── simple_midi/       # Filtered simple files
│   └── tokens/            # Tokenized sequences
├── src/
│   ├── data_processing.py # MIDI parsing and tokenization
│   ├── model.py          # LSTM model definition
│   ├── train.py          # Training loop
│   └── generate.py       # Generation and MIDI export
├── notebooks/
│   └── exploration.ipynb # Data exploration
└── README.md
```

## Time Estimate
- **Total time**: 4 weeks (working part-time)
- **Key milestone**: Week 4 - hearing your first AI-generated melody
- **Reality check**: It will probably sound bad, but it will be YOUR bad AI melody!

## Most Important Principle
**Make it work first, make it good later.** Get something end-to-end working, even if it's terrible. You'll learn more from a working bad system than from a perfect system that doesn't exist.
