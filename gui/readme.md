# 🎵 Orpheus MIDI Parser - GUI

Web-based interface for processing MIDI files into tokenized JSON for AI music generation.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the Orpheus directory
pip install gradio
```

### 2. Launch the GUI

```bash
# From the Orpheus directory
python gui/gradio_app.py
```

The interface will open in your browser at `http://localhost:7860`

### 3. Process Your First File

1. Choose **Simple Mode** (for melody + chord) or **Advanced Mode** (for any MIDI)
2. Upload a MIDI file
3. Set output directory (default: `./processed`)
4. Click **🚀 Process File**
5. Download your tokenized JSON!

## 📋 Features

### Simple Mode (Recommended for Music Generation)
- Processes MIDI with exactly 2 tracks: Melody + Chords
- Validates monophonic melody
- Optimized for AI training
- Strict structure checking

### Advanced Mode (General Purpose)
- Processes any MIDI file
- Multiple tracks supported
- Flexible validation
- Comprehensive metadata

### Batch Processing
- Process multiple files at once
- Progress tracking per file
- Detailed success/failure reports
- Automatic error recovery

## 🎼 Preparing MIDI Files

### For Simple Mode:

1. **Melody Track:**
   - Must be monophonic (one note at a time)
   - No overlapping notes
   - Clear melodic line

2. **Chord Track:**
   - Sustained chords
   - No rhythmic variation
   - Each chord holds until next chord

3. **Track Naming Tips:**
   - Name clearly: "Melody", "Chords", "Lead", etc.
   - Supports multiple languages
   - Auto-detection based on content

### Example Structure:
```
✅ GOOD:
- Track 1: "Melody" - Monophonic piano melody
- Track 2: "Chords" - Sustained pad chords

❌ BAD:
- Track 1: "Melody" - Has chord accompaniment mixed in
- Track 2: "Drums" - Wrong track type
- Track 3: "Bass" - Too many tracks
```

## 📊 Output Format

Processed files are saved as JSON (optionally compressed):

```json
{
  "version": "2.0",
  "source_file": "my_song.mid",
  "tokenization": "REMI",
  "metadata": {
    "ppq": 480,
    "tempo_changes": [...],
    "duration_seconds": 180.5,
    "track_count": 2,
    "note_count": 1234
  },
  "tracks": [
    {
      "name": "Melody",
      "type": "melody",
      "tokens": [45, 67, 89, ...],
      "token_count": 2048,
      "note_count": 567
    },
    {
      "name": "Chords",
      "type": "chord",
      "tokens": [12, 34, 56, ...],
      "token_count": 1024,
      "note_count": 345
    }
  ],
  "sequence_length": 3072
}
```

**File naming:** `{KEY}-{TEMPO}bpm-{TOKENIZATION}-{title}.json`

Example: `Cmajor-120bpm-remi-my_song.json`

## 🎯 Use Cases

### 1. AI Music Generation Training
Process your MIDI dataset with Simple Mode to create consistent training data:

```python
# Batch process all your MIDI files
input_dir = "source_midis/batch/"
output_dir = "processed/"

# Use Gradio batch tab or Python script
from midi_parser.interface import MidiParserGUI
parser = MidiParserGUI(config=simple_config)
results = parser.process_batch(midi_files, output_dir)
```

### 2. MIDI Analysis
Use Advanced Mode to analyze any MIDI file:
- Extract musical features
- View track statistics
- Analyze structure and complexity

### 3. Dataset Preparation
Clean and standardize MIDI files:
- Filter out invalid tracks
- Normalize structure
- Validate musical content

## 🛠️ Configuration

### Simple Mode Settings:
- **Tokenization:** REMI (best for structured music)
- **Beat Resolution:** 8 (good rhythm precision)
- **Velocity Bins:** 16 (sufficient dynamics)
- **Max Sequence Length:** 2048 tokens

### Advanced Mode Settings:
- **Tokenization:** REMI (configurable)
- **Beat Resolution:** 8
- **Velocity Bins:** 32 (more dynamic range)
- **Max Sequence Length:** 4096 tokens

## 🚨 Troubleshooting

### "Expected 2 tracks, found X"
**Solution:** Use preprocessor to reduce tracks or switch to Advanced Mode

```bash
# Use preprocessor to clean MIDI first
python preprocessor/process_midi_file.py input.mid output.mid
```

### "Melody track is not monophonic"
**Solution:** Remove overlapping notes in your MIDI editor

### "Processing takes too long"
**Solution:** 
- Try smaller files first
- Close other applications
- Check file complexity (very dense files take longer)

### "Memory warning"
**Solution:**
- Process smaller files
- Close other applications
- Use batch processing for multiple small files instead of one large file

### "No valid tracks found"
**Solution:**
- Check MIDI file isn't empty
- Verify tracks have notes
- Try Advanced Mode (less strict)

## 💡 Tips for Best Results

1. **Clean your MIDI files first**
   - Use the preprocessor module
   - Remove empty tracks
   - Normalize velocities

2. **Use consistent naming**
   - Name tracks clearly
   - Use recognized keywords (Melody, Chord, etc.)

3. **Batch process similar files together**
   - Same genre/style
   - Similar length
   - Same structure

4. **Monitor output quality**
   - Check token counts
   - Verify track types
   - Review statistics

5. **Start with Simple Mode**
   - Enforces good structure
   - Better for AI training
   - Easier to validate

## 🎨 Customization

### Custom Configuration:
```python
from midi_parser.config.defaults import MidiParserConfig, TokenizerConfig
from midi_parser.interface import MidiParserGUI

# Create custom config
my_config = MidiParserConfig(
    tokenization="REMI",
    tokenizer=TokenizerConfig(
        pitch_range=(21, 108),
        beat_resolution=16,  # Higher resolution
        num_velocities=32,
        additional_tokens={
            "Chord": True,
            "Rest": True,
            "Tempo": True
        }
    )
)

# Use in GUI
parser = MidiParserGUI(config=my_config)
```

### Extending the GUI:
The Gradio interface is built with Blocks, making it easy to customize:
- Add new tabs
- Modify layouts
- Add visualizations
- Integrate with other tools

## 📈 Performance

**Typical Processing Times:**
- Small file (<1MB, <5 min): 1-3 seconds
- Medium file (1-5MB, 5-10 min): 3-10 seconds
- Large file (5-10MB, 10-20 min): 10-30 seconds
- Very large file (>10MB): May require chunking

**Memory Usage:**
- Small files: <100MB RAM
- Medium files: 100-500MB RAM
- Large files: 500MB-2GB RAM
- Files >50MB: Automatic memory warning

## 🔗 Integration

### With Validation Module:
```python
# Process and validate in one step
from midi_parser.interface import MidiParserGUI
from midi_parser.validation.quality_control_main import validate_tokenization_pipeline

parser = MidiParserGUI()
result = parser.process_file(input_path, output_dir)

if result.success:
    # Validate the output
    validation_result = validate_tokenization_pipeline(
        result.output_path,
        original_midi=input_path
    )
    print(f"Quality score: {validation_result.quality_score}")
```

### With Preprocessor:
```python
# Preprocess then parse
from preprocessor.preprocessor import preprocess_midi
from midi_parser.interface import MidiParserGUI

# Clean MIDI first
cleaned_path = preprocess_midi(input_path)

# Then parse
parser = MidiParserGUI()
result = parser.process_file(cleaned_path, output_dir)
```

## 🤝 Contributing

Found a bug or have a feature request? Contributions welcome!

## 📄 License

Part of the Orpheus Project - AI Music Generation

---

**Happy Music Making! 🎵**