# Simple AI Melody Generator - Minimal Version

## Project Goal
Build the simplest possible AI that can generate a melody over a chord progression using MIDI data.

## What We're Building
- **Input**: A simple chord progression (like C-Am-F-G)
- **Output**: A single melody line that sounds good over those chords
- **Architecture**: Basic LSTM model
- **Dataset**: Lakh MIDI (filtered to simple files only)

## Technical Stack
- **Python 3.8+**
- **PyTorch** (for the neural network)
- **pretty_midi** (for MIDI file handling)
- **numpy** (for data processing)

## Data Representation (Keep It Simple)
We'll represent music as a sequence of events:
- `NOTE_ON_60` (start playing middle C)
- `NOTE_OFF_60` (stop playing middle C)  
- `WAIT_120` (wait 120 milliseconds)
- `CHORD_C` (chord change to C major)

## Phase 1: Data Exploration (Week 1)
**Goal**: Understand what's in MIDI files

### Step 1.1: Download and Look at Data
```bash
# Download a small subset of Lakh MIDI dataset
# Start with just 100 files to keep it manageable
```

### Step 1.2: Basic MIDI Parsing
Write a script that:
- Opens a MIDI file
- Prints basic info (duration, number of tracks, tempo)
- Extracts all notes and their timing
- Saves this info to a text file

### Step 1.3: Find Simple Files
Filter the dataset to find files that have:
- Only 1-2 instruments (no drums)
- Duration between 30-120 seconds
- Not too many notes (avoid classical pieces with rapid passages)

**Deliverable**: A folder with ~50 "simple" MIDI files

## Phase 2: Manual Data Preparation (Week 2)
**Goal**: Create a tiny, hand-curated training dataset

### Step 2.1: Create Training Examples by Hand
Pick 5-10 of your simplest MIDI files and manually:
- Identify which track has chords (multiple simultaneous notes)
- Identify which track has melody (single notes)
- Write down the chord progression by listening
- Note the key signature

### Step 2.2: Basic Tokenization
Convert your hand-picked files into token sequences:
```python
# Example output for "Twinkle Twinkle Little Star" in C major:
tokens = [
    "CHORD_C", "NOTE_ON_60", "WAIT_500", "NOTE_OFF_60", 
    "NOTE_ON_60", "WAIT_500", "NOTE_OFF_60",
    "NOTE_ON_67", "WAIT_500", "NOTE_OFF_67",
    "NOTE_ON_67", "WAIT_1000", "NOTE_OFF_67",
    "CHORD_F", "NOTE_ON_65", "WAIT_500", "NOTE_OFF_65"
    # ... etc
]
```

**Deliverable**: 5-10 files converted to token sequences, saved as text files

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
