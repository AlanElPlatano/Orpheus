# MIDI Parser Specification v2.0 – MidiTok Integration

> Goal: Convert a folder of MIDI files into a token-based format suitable for training generative models using MidiTok, with high-fidelity round-trip conversion.

---

## TL;DR
- **Primary Library:** **MidiTok** for tokenization/detokenization with **miditoolkit** for MIDI I/O
- **Tokenization Strategy:** **REMI** (default) for transformer compatibility
- **Output Format:** JSON with integer token sequences and comprehensive metadata
- **Key Feature:** Professional-grade tokenization with proven transformer compatibility

---

## 1. Technology Stack & Rationale

### Core Libraries
- **MidiTok**: Primary tokenization engine (supports REMI, TSD, Structured, CPWord, Octuple)
- **miditoolkit**: MIDI file I/O and manipulation
- **mido**: Low-level MIDI backend (via miditoolkit)

### Why MidiTok?
- **Battle-tested**: Used in major music AI research projects
- **Multiple Strategies**: Supports all major tokenization formats
- **Round-trip Fidelity**: Designed specifically for lossless MIDI conversion
- **Transformer Optimized**: Tokens are optimized for modern neural architectures

---

## 2. Tokenization Strategy Selection

### Primary Choice: **REMI Tokenization**
```python
from miditok import REMI
tokenizer = REMI(
    pitch_range=(21, 109),      # A0 to C8
    beat_resolution=4,          # 16th note resolution
    num_velocities=16,          # Velocity quantization
    additional_tokens={
        'Chord': True,          # Include chord tokens
        'Rest': True,           # Include rest tokens
        'Tempo': True,          # Include tempo changes
        'TimeSignature': True,  # Include time signature changes
        'Program': True,        # Include program changes
    }
)
```

### Alternative Strategies (Configurable)
- **TSD**: Time-Shift Duration encoding
- **Structured**: Explicit musical structure
- **CPWord**: Compound Word encoding
- **Octuple**: 8-dimensional feature encoding

### Decision Framework
```python
TOKENIZATION_STRATEGIES = {
    "transformer_training": "REMI",      # Best for transformers
    "simplicity": "TSD",                 # Simpler vocabulary
    "explicit_structure": "Structured",  # Clear musical hierarchy
    "compression": "CPWord",             # Shorter sequences
}
```

---

## 3. Output Format Specification

### JSON Schema
```json
{
  "version": "2.0",
  "source_file": "original.mid",
  "tokenization": "REMI",
  "tokenizer_config": {
    "pitch_range": [21, 109],
    "beat_resolution": 4,
    "num_velocities": 16,
    "additional_tokens": {
      "Chord": true,
      "Rest": true,
      "Tempo": true,
      "TimeSignature": true,
      "Program": true
    }
  },
  "metadata": {
    "ppq": 480,
    "tempo_changes": [
      {"tick": 0, "bpm": 120.0},
      {"tick": 1920, "bpm": 140.0}
    ],
    "time_signatures": [
      {"tick": 0, "numerator": 4, "denominator": 4}
    ],
    "key_signature": "C major",
    "duration_seconds": 180.5
  },
  "tracks": [
    {
      "name": "Piano Right Hand",
      "program": 0,
      "is_drum": false,
      "type": "melody",
      "tokens": [120, 45, 678, 234, 890, 123, ...],
      "token_count": 1450,
      "note_count": 320
    }
  ],
  "global_events": [56, 78, 92],  // Tempo, time signature changes
  "sequence_length": 2048
}
```

### File Naming Convention
```
{KEY}-{AVG_TEMPO}bpm-{TOKENIZATION}-{sanitized_title}.json
```
Example: `Cm-120bpm-REMI-sonata_no_1.json`

---

## 4. Processing Pipeline

### Step 1: MIDI Loading & Validation
```python
def load_and_validate_midi(file_path):
    # Load MIDI with miditoolkit
    # Validate file integrity
    # Extract metadata (PPQ, tempo map, time signatures)
    return midi_object, metadata
```

### Step 2: Track Analysis & Typing
```python
def analyze_tracks(midi_obj):
    # Apply chord/melody detection heuristics
    # Classify tracks as: 'melody', 'chord', 'bass', 'drum', 'accompaniment'
    # Filter out empty/invalid tracks
    return track_metadata
```

### Step 3: Tokenization with MidiTok
```python
def tokenize_midi(midi_obj, tokenizer):
    # Let MidiTok handle complex tokenization logic
    # Preserve all musical features automatically
    tokens = tokenizer(midi_obj)
    return tokens, tokenizer.vocab
```

### Step 4: JSON Serialization
```python
def create_output_json(midi_path, tokens, metadata, track_info):
    # Combine all data into standardized JSON format
    # Compress if needed (gzip optional)
    return json_data
```

---

## 5. Round-trip Fidelity Assurance

### Validation Metrics
```python
VALIDATION_TOLERANCES = {
    "note_start_tick": 1,           # Max 1 tick difference
    "note_duration": 2,             # Max 2 ticks difference  
    "velocity_bin": 1,              # Max 1 velocity bin difference
    "missing_notes_ratio": 0.01,    # Max 1% notes missing
    "extra_notes_ratio": 0.01,      # Max 1% extra notes
}
```

### Automated Testing Suite
```python
def round_trip_test(original_midi, tokenizer):
    # Tokenize original MIDI
    tokens = tokenizer(original_midi)
    
    # Detokenize back to MIDI
    reconstructed_midi = tokenizer(tokens)
    
    # Compare using miditoolkit
    metrics = compare_midi_files(original_midi, reconstructed_midi)
    
    return metrics, metrics_within_tolerance(metrics)
```

---

## 6. Track Classification Heuristics

### Melody Detection
- **Name-based**: Contains "melody", "lead", "solo", "voice"
- **Pattern-based**: Monophonic or light polyphony (< 3 simultaneous notes)
- **Range-based**: Moderate pitch range (1-2 octaves)
- **Density-based**: Moderate note density (not too sparse/dense)

### Chord Detection  
- **Name-based**: Contains "chord", "accomp", "rhythm", "pad"
- **Pattern-based**: High polyphony (≥ 3 simultaneous notes common)
- **Sustain-based**: Long note durations, sustained patterns

### Bass Detection
- **Name-based**: Contains "bass", "low", "bajo"
- **Range-based**: Low pitch range (C1-C3 typical)
- **Rhythm-based**: Rhythmic, foundational patterns

---

## 7. Configuration Management

### Default Configuration
```python
DEFAULT_CONFIG = {
    "tokenization": "REMI",
    "pitch_range": (21, 109),
    "beat_resolution": 4,
    "num_velocities": 16,
    "max_seq_length": 2048,
    "track_classification": {
        "chord_threshold": 3,
        "min_notes_per_track": 10,
        "max_empty_ratio": 0.8
    },
    "output": {
        "compress_json": True,
        "include_vocabulary": True,
        "pretty_print": False
    }
}
```

### Environment-based Overrides
```python
# config.yaml (optional)
tokenization: "CPWord"
beat_resolution: 8
output:
  compress_json: false
```

---

## 8. Project Structure

```
midi_parser/
├── __init__.py
├── config/
│   ├── defaults.py
│   └── strategies.yaml
├── core/
│   ├── midi_loader.py
│   ├── tokenizer_manager.py
│   ├── track_analyzer.py
│   └── json_serializer.py
├── validation/
│   ├── round_trip.py
│   ├── metrics.py
│   └── quality_check.py
├── cli/
│   ├── batch_process.py
│   └── single_file.py
└── tests/
    ├── test_tokenization.py
    ├── test_fidelity.py
    └── fixtures/
```

---

## 9. CLI Interface

### Batch Processing
```bash
python -m midi_parser batch \
    --input-dir ./raw_midis \
    --output-dir ./tokenized \
    --tokenization REMI \
    --config ./config.yaml \
    --validate
```

### Single File Processing
```bash
python -m midi_parser single \
    --input ./song.mid \
    --output ./song_tokens.json \
    --tokenization CPWord
```

### Validation Mode
```bash
python -m midi_parser validate \
    --input-dir ./tokenized \
    --threshold 0.98
```

---

## 10. Quality Assurance Checklist

### Pre-processing Validation
- [ ] MIDI file integrity check
- [ ] PPQ consistency across files
- [ ] Tempo map extraction accuracy
- [ ] Time signature change handling

### Tokenization Validation  
- [ ] Round-trip fidelity > 98%
- [ ] Vocabulary coverage of all musical events
- [ ] Track classification accuracy
- [ ] Sequence length distribution analysis

### Output Validation
- [ ] JSON schema compliance
- [ ] Metadata completeness
- [ ] Token sequence integrity
- [ ] File size optimization

---

## 11. Performance Considerations

### Memory Management
- Stream large MIDI files without full loading
- Batch processing with memory limits
- Token sequence chunking for long pieces

### Processing Speed
- Parallel track processing where possible
- Cached tokenizer instances
- Optimized JSON serialization

### Storage Optimization
- Gzip compression for JSON files
- Binary token storage option
- Metadata-only mode for analysis

---

## 12. Migration & Compatibility

### Version 1.0 → 2.0 Migration
```python
def migrate_v1_to_v2(v1_json):
    # Convert custom tokens to MidiTok integer tokens
    # Preserve metadata structure
    # Update schema version
    return v2_json
```

### Backward Compatibility
- Read both v1 and v2 JSON formats
- Convert between tokenization strategies
- Preserve all original metadata

---

## Appendix A: MidiTok Tokenization Strategies Comparison

| Strategy | Vocab Size | Seq Length | Transformer Suitability | Musical Expressiveness |
|----------|------------|------------|------------------------|------------------------|
| **REMI** | Medium | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **TSD** | Small | Long | ⭐⭐⭐ | ⭐⭐⭐ |
| **Structured** | Large | Short | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CPWord** | Medium | Short | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Octuple** | Large | Short | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Appendix B: Example Token Sequences

### REMI Token Example
```
[Bar_Start, Position_0, Tempo_120, TimeSignature_4_4, 
 NoteOn_C4_v80, Duration_4, NoteOn_E4_v80, Duration_4, 
 NoteOn_G4_v80, Duration_4, Bar_Start, Position_0, ...]
```

### Corresponding Integer Tokens
```
[120, 45, 256, 312, 678, 48, 712, 48, 745, 48, 120, 45, ...]
```

---

## Closing Notes

This specification leverages MidiTok's professional tokenization capabilities while maintaining the project's original goals of high fidelity and transformer compatibility. The modular design allows for easy experimentation with different tokenization strategies while ensuring consistent output format and comprehensive metadata preservation.

**Key Advantages:**
- No need to design token vocabulary from scratch
- Proven compatibility with transformer architectures
- Comprehensive handling of complex MIDI features
- Active library maintenance and community support