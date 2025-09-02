# AI Melody Generation Over Chord Progressions

## Project Overview
Building an AI system that generates melodies over chord progressions using machine learning. The AI will be trained on a combination of custom MIDI files and existing datasets to produce new musical compositions, starting with basic melody generation and later expanding to mood-based generation.

## Dataset Specifications

### Primary Dataset: Lakh MIDI Dataset
* **Source**: Lakh MIDI Dataset (large-scale collection of MIDI files)
* **Size**: Thousands of training examples
* **Benefit**: Diverse musical styles and key signatures for better generalization
* **Challenge**: Contains corrupt files that need filtering

### Secondary Dataset: Custom Guitar Pro Files
* **Source Format**: Guitar Pro 8 (.gp) files → Manual export to MIDI format
* **Content**: Simple 2-track compositions (chord progression + melody)
* **Chord Progressions**: Extremely simple - single sustained chords without rhythmic variance, arpeggios, or expression
* **Processing**: Manual export with cleanup to ensure better compatibility with AI training
* **Mood Labels**: Each file tagged with emotions (for future phases)

## Technical Architecture Recommendations

### Framework & Libraries
* **Primary Framework**: PyTorch (better for research/experimental work)
* **Music Processing**:
   * `pretty_midi` or `mido` for MIDI parsing
   * `music21` for music theory utilities
   * **MIDI Validation**: Custom corruption detection and filtering system
* **Model Architecture**: Start with LSTM baseline, progress to Transformer

### Data Representation
* **Format**: Event-based tokenization from MIDI
* **Tokens**: `NOTE_ON_C4`, `NOTE_OFF_C4`, `TIME_SHIFT_120`, `CHORD_F#m`, etc.
* **Mood Conditioning**: Reserved for later phases

### Model Architecture Options
1. **LSTM/GRU with attention** (recommended for baseline)
   * Simpler implementation
   * Good for prototyping
   * Lower computational requirements

2. **Transformer-based** (recommended for advanced version)
   * Music Transformer architecture
   * Excellent long-term dependencies
   * Built-in conditional generation support

### Time Signature Control System
* **Generation-Time Conditioning**: Time signature specified as model input parameter
* **Token-Based Implementation**: Include time signature tokens in sequence vocabulary
* **Constrained Generation**: Enforce timing rules during generation process (not post-processing)
* **Supported Signatures**: 4/4, 3/4, 2/4, 6/8 (expandable)
* **Validation Integration**: Generated files validated for complete time signature consistency

## Implementation Roadmap

### Phase 0: Proof of Concept & Baseline
* **Rule-based melody generator** using basic music theory
* Simple harmonic relationships with sustained chords
* Establish baseline performance metrics
* Validate problem understanding and approach

### Phase 1: Data Pipeline & Validation
* **MIDI file parsing and corruption detection**
  * Implement robust MIDI validation system
  * Filter out corrupt files from Lakh dataset
  * Log and analyze corruption patterns
* **Tokenization system development**
* **Data preprocessing pipeline**
* **Basic data augmentation strategies**

### Phase 2: Basic Model Development
* **Simple LSTM baseline implementation**
* **Training infrastructure setup**
* **Focus**: Single-note melodies over sustained chords
* **Evaluation**: Time signature correctness validation
* **Model evaluation metrics implementation**

### Phase 3: Advanced Architecture
* **Transformer model implementation**
* **Attention mechanisms**
* **Advanced sampling strategies**
* **Progressive complexity increase**

### Phase 4: Mood Conditioning (Future Phase)
* **Mood tag integration** (deferred from earlier phases)
* **Conditional generation with mood embeddings**
* **Advanced mood-melody relationship modeling**

### Phase 5: Production Interface
* **Quantity Selector UI** with reasonable limits (e.g., 1-20 files max)
* **Batch generation system** with invalid file filtering
* **Generation attempt limits** (max 50 attempts to prevent infinite loops)
* **Progress indicators** for batch generation process
* MIDI generation and export for multiple valid files
* Post-processing for musical coherence
* User-friendly error handling for failed generation attempts
* *Future*: Mood selection interface

## Target Functionality

### Initial Version
**Input**: Basic parameters (tempo, key, time signature) + quantity selector
**Output**: Multiple valid 2-track MIDI files (sustained chord + single-note melody)
**Evaluation**: Time signature correctness validation with automatic filtering
**Generation Logic**: Batch generation with invalid file removal and time signature enforcement

### Future Version
**Input**: Mood tag selection + time signature + quantity selector
**Output**: Multiple valid 2-track MIDI files (chord progression + melody)
**Interaction**: Single-click generation with quantity and time signature control

## Key Technical Challenges
* **MIDI corruption filtering** - Robust detection and removal of invalid files
* **Batch generation with validation** - Efficient filtering of invalid outputs
* **Generation attempt management** - Preventing infinite loops with poorly performing models
* **Time signature consistency** - Ensuring generated melodies respect rhythmic constraints throughout entire file
* **Time signature conditioning** - Implementing generation-time control rather than post-processing filtering
* Maintaining harmonic consistency between chords and melody
* Balancing creativity with musical coherence
* *Future*: Effective mood conditioning

## Evaluation Metrics
### Primary Metric: Time Signature Validation
* **4/4 Time**: Verify each bar contains exactly 4 beats worth of notes
* **Other Signatures**: Adapt validation logic accordingly
* **Success Criteria**: Generated melody respects specified time signature constraints
* **Batch Processing**: Automatic filtering of invalid files during generation

### Generation Quality Control
* **Attempt Limits**: Maximum 50 generation attempts to prevent infinite loops
* **UI Constraints**: User quantity selector limited to reasonable range (1-20 files)
* **Success Rate Monitoring**: Track ratio of valid to invalid generations
* **Fallback Handling**: Clear user messaging when generation targets cannot be met

### Secondary Metrics (Future)
* Harmonic consonance with chord progressions
* Human evaluation scores
* Comparison with rule-based baseline

## Development Environment
* Local GPU setup for training
* Python-based development stack
* PyTorch ecosystem

## Next Steps
1. **Download and analyze Lakh MIDI dataset**
2. **Implement MIDI corruption detection system**
3. **Set up MIDI parsing pipeline with validation**
4. **Create rule-based baseline (Phase 0)**
5. **Build tokenization system**
6. **Implement time signature validation metrics**
7. **Build simple LSTM baseline**
8. **Scale to transformer architecture**
