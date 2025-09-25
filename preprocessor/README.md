## Preprocessor

All of the scripts in this module are meant to clean and normalize raw MIDI files to prepare them for AI training. The module loads a MIDI, computes tempo (only when needed), removes problematic notes and tracks, optionally quantizes timing to a grid, and returns both the processed `PrettyMIDI` object and summary statistics.

### What this module does
- **Cleanup (optional)**: Removes empty/zero-length notes, ultra‑short notes (< 1/64th note), and trims overlapping notes of the same pitch. Skips drum tracks for these operations.
- **Quantization (optional)**: Snaps note start/end times to a user‑specified grid (e.g., 16 = 1/16 notes), preserving a minimum duration.
- **Bass track removal (optional)**: Removes tracks named 'bass' (with multi‑language heuristics) or with >35% notes below a threshold pitch (default C2, MIDI 36). Drum tracks are never considered.
- **Empty track removal (optional)**: Drops tracks with zero notes.
- **Statistics**: Reports counts (notes, tracks), tempo, pitch range, density, short notes remaining, and how many notes/tracks were removed.

**All preprocessing steps now optional** - you can enable or disable any combination of steps based on your needs. The BPM calculation is only performed when required by enabled steps.

### Drum Track Handling

**Drum tracks are preserved and protected** throughout the preprocessing pipeline. The module automatically detects drum tracks (via `instrument.is_drum`) and excludes them from note cleanup, quantization, and bass track removal operations. This preserves the original timing and characteristics of drum patterns, which often rely on precise micro-timing and short percussive hits that would be inappropriate to modify. Drum tracks are only affected by the empty track removal step - empty drum tracks are removed just like empty melodic tracks.

## Pipeline flow

There are two main ways to run the preprocessor: the plain function or the class wrapper.

### 1. Functional API  
`process_midi_file(...)` does all the work in order, here is a detailed description:

1. **Load MIDI** → turns the file into a `PrettyMIDI` object.  
2. **Determine tempo needs** → only calculates tempo if cleanup_notes=True or quantize_grid is provided.  
3. **Find tempo (if needed)** → grabs the first tempo change if it exists, otherwise estimates it, and if that fails defaults to 120 BPM.  
4. **Initial stats** → collect basic info about the file before changes are made.  
5. **Cleanup (optional)** → if `cleanup_notes=True`, removes zero-length notes, very short notes (< 1/64th note), and trims overlapping notes of the same pitch. Drum tracks are ignored.  
6. **Quantize (optional)** → if a grid is given (like 16 = 1/16th notes), it snaps note timing to that grid and keeps a minimum 1/64-note duration.  
7. **Remove bass (optional)** → if `remove_bass=True`, drops tracks that look like bass (by name or by having 35% of notes below a cutoff, default C2).  
8. **Remove empty tracks (optional)** → if `remove_empty=True`, deletes any track with no notes left.  
9. **Final stats** → recomputes info and notes what steps were applied.  
10. **Return** → `(success, processed_midi, stats)`.

### 2. Class API  
`MIDIPreprocessor` caches the last computed tempo for the current `PrettyMIDI` object and reuses it across helper-method calls. The full `process_midi_file(...)` pipeline computes tempo only when needed (cache is cleared per new file). Use the class when you want to invoke individual steps without having to re-supply tempo, or to control verbosity via the constructor. 

- **Stateful design**  
  - Caches the last computed tempo for reuse with the same MIDI object.  
  - Keeps verbosity settings and options across runs.  

- **Convenience methods**  
  You can call each step individually if you don't want the full pipeline:  
  - `preprocess_notes(midi, tempo_bpm)`  
  - `quantize_midi_timing(midi, grid, tempo_bpm)`  
  - `remove_bass_tracks(midi, threshold_note)`  
  - `remove_empty_tracks(midi)`  
  - `get_preprocessing_stats(midi, tempo_bpm)`  

- **Full pipeline**  
  - `process_midi_file(...)` runs the same sequence as the functional API.  
  - Accepts the same arguments (`quantize_grid`, `remove_empty`, `remove_bass`, `bass_threshold`, `cleanup_notes`).  

- **When to use**  
  - Functional API → quick one-off processing.  
  - Class API → repeated runs, batch jobs, or when you want more control over intermediate steps.  


### Functions and behavior

- **`bpm_reader.get_tempo_from_midi(midi)`**  
  Returns a single BPM value.  
  - Uses the first tempo change if present.  
  - Falls back to estimated tempo.  
  - Defaults to 120 BPM if all else fails.  
  - **Only called when needed** (when cleanup_notes=True or quantization is enabled).

- **`cleanup.preprocess_notes(midi, tempo_bpm, verbose=False)`**  
  Cleans up notes (ignores drums):  
  - Removes zero-length notes.  
  - Removes ultra-short notes (< 1/64 note).  
  - Trims overlapping notes of the same pitch.  
  - **Optional step** - controlled by `cleanup_notes` parameter.

- **`quantizer.quantize_midi_timing(midi, quantize_grid, tempo_bpm, verbose=False)`**  
  Adjusts note timing to a grid.  
  - If `quantize_grid` is None, does nothing.  
  - Snaps start/end to the given grid (e.g., 16 = 1/16 notes).  
  - Keeps at least 1/64 note length.  
  - Falls back to 120 BPM if tempo looks invalid.  

- **`remove_bass_tracks.remove_bass_tracks(midi, threshold_note=36, verbose=False)`**  
  Removes bass-like tracks (ignores drums):  
  - Drops tracks with bass-related names (e.g., "bass", "bajo", "contrabajo").  
  - Or if more than 35% of notes are below `threshold_note` (default C2).  
  - **Optional step** - controlled by `remove_bass` parameter.

- **`remove_empty_tracks.remove_empty_tracks(midi, verbose=False)`**  
  Removes tracks with no notes.  
  - **Optional step** - controlled by `remove_empty` parameter.

- **`stats.get_preprocessing_stats(midi, tempo_bpm=None, verbose=False)`**  
  Returns a dictionary of summary stats, including:  
  - Total notes, tracks (drums vs melodic).  
  - Duration (seconds).  
  - Tempo (BPM) - only calculated if not provided.  
  - Notes per second.  
  - Pitch range and span.  
  - Remaining short notes.  
  - Time signatures.


### Using the pipeline elsewhere
You can either use the functional API or the class wrapper. All preprocessing steps are now optional.

Functional API:
```python
from preprocessor.process_midi_file import process_midi_file

success, processed_midi, stats = process_midi_file(
    file_path="/path/to/file.mid",
    quantize_grid=16,      # None to skip; 16 -> 1/16 notes, 32 -> 1/32, etc.
    remove_empty=True,     # Remove empty tracks
    remove_bass=True,      # Remove bass tracks
    bass_threshold=36,     # MIDI note number; 36 = C2
    cleanup_notes=True,    # Clean up short/empty notes and overlaps
    verbose=False,
)

if success:
    processed_midi.write("/path/to/out_processed.mid")
```

Object‑oriented API:
```python
from preprocessor.preprocessor import MIDIPreprocessor

pre = MIDIPreprocessor(verbose=True)
ok, midi, stats = pre.process_midi_file(
    "./source.mid",
    quantize_grid=None,    # Skip quantization
    remove_empty=True,     # Remove empty tracks
    remove_bass=False,     # Keep bass tracks
    bass_threshold=36,
    cleanup_notes=False,   # Skip note cleanup
)
```

Batch example (see `tests.py` for more detailed implementation):
```python
from pathlib import Path
from preprocessor import MIDIPreprocessor

src = Path("./source_midis")
out = Path("./processed"); out.mkdir(parents=True, exist_ok=True)

pre = MIDIPreprocessor(verbose=True)
for midi_path in src.glob("*.mid"):
    ok, midi, stats = pre.process_midi_file(
        str(midi_path), 
        remove_bass=True, 
        bass_threshold=36,
        cleanup_notes=True,     # Enable note cleanup
        remove_empty=True       # Remove empty tracks
    )
    if ok:
        midi.write(str(out / f"{midi_path.stem}_processed{midi_path.suffix}"))
```

### Inputs, outputs, and arguments

- **Input**  
  - `file_path`: Path to a MIDI file (string).  

- **Key arguments**  
  - `quantize_grid`: `None` (skip) or integer grid (16 = 1/16, 32 = 1/32, etc.).  
  - `remove_empty`: `True` to drop empty tracks (default `True`).  
  - `remove_bass`: `True` to remove bass-like tracks (default `False`).  
  - `bass_threshold`: Pitch cutoff for bass detection (default `36 = C2`).  
  - `cleanup_notes`: `True` to perform note cleanup - remove short/empty notes and trim overlaps (default `True`).
  - `verbose`: Print step-by-step details (`False` by default).
    - Functional API: pass verbose per call.
    - Class API: set verbose in the constructor; the class methods use it automatically.

- **Output**  
  - A tuple `(success, processed_midi, stats)`  
    - `success`: `True` if everything worked, `False` otherwise.  
    - `processed_midi`: The cleaned `PrettyMIDI` object (or `None` on failure).  
    - `stats`: Dictionary with summary info (`notes`, `tracks`, `tempo`, `pitch range`, etc). Includes `stats['preprocessing_applied']` to show which steps were run, including the new `note_cleanup` field.  

### GUI Integration Ready

The modular design with optional steps makes this module perfect for GUI integration:

- Each preprocessing step can be represented as a checkbox
- BPM calculation is automatically handled - only computed when needed
- If no preprocessing steps are selected, the pipeline simply loads and returns the MIDI with basic stats
- The `preprocessing_applied` field in the stats shows exactly which steps were performed

Example GUI mapping:
- ☑ **Clean up notes** (`cleanup_notes=True`) - Remove short notes and overlaps
- ☑ **Quantize timing** (`quantize_grid=16`) - Snap to 1/16 note grid
- ☐ **Remove bass tracks** (`remove_bass=False`) - Remove low-pitched tracks
- ☑ **Remove empty tracks** (`remove_empty=True`) - Remove tracks with no notes

### Dependencies
- `pretty_midi`
- `miditoolkit`

Install:
```bash
pip install pretty_midi miditoolkit
```

### Practical notes and assumptions
- The pipeline treats tempo as a single representative BPM; for files with multiple tempo changes, it effectively uses the first detected tempo or an estimate.
- **Tempo is only calculated when needed** - if no cleanup or quantization is requested, BPM calculation is skipped.
- Drum tracks are excluded from cleanup, quantization, and bass heuristics.
- A minimum 1/64‑note duration is enforced during cleanup and after quantization to avoid zero/near‑zero lengths.
- On tempo extraction/estimation failures or implausible BPM, 120.0 BPM is used as a safe default.
- All preprocessing steps can be independently enabled or disabled.

### Repository layout context
- Place raw MIDIs in `source_midis/` and outputs will commonly go to `processed/` (see `tests.py`). You can adopt the same structure in another project, or wire your own I/O while reusing the functions/classes above.
