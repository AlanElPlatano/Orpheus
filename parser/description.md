# MIDI Parser — Design & Specification

> Goal: convert a folder of MIDI files into a deterministic, token-based text format suitable for training generative models, and be able to convert model output tokens back into valid MIDI files with high fidelity.

---

## TL;DR
- **Library :** primary: **miditoolkit** (works natively in MIDI ticks and builds on `mido`). Use `mido` as the low-level read/write backend when you need ultimate control.
- **Best parsed-file format:** **JSON** (one file per MIDI) with a compact token array and explicit metadata; for large corpora prefer **NDJSON (.jsonl)** or gzipped JSON.  
- **Round-trip (reversible) strategy:** preserve native timing (ticks), full tempo map, time signatures, PPQ (ticks-per-quarter), program/patch info, channel info, and deterministic event ordering; run automated round-trip tests (MIDI → tokens → MIDI → compare). 
---

## 1. Why the library choice matters
Working with symbolic MIDI data can be done at different abstraction layers:
- low-level message & tick fidelity (raw delta ticks, channels, program changes),
- note-level abstractions (notes as events with start/end in seconds or ticks),
- high-level musical analysis (key detection, chord labeling, beat/beat positions).

Your parser must be able to both read all relevant MIDI meta events and write MIDI back with the same timing semantics. That pushes us toward libraries that:
- operate in MIDI ticks (not only seconds),
- expose tempo maps, time signatures, program / channel info,
- let you read and write files faithfully.

### Recommendation (short)
- **Primary library: `miditoolkit`** — built around ticks and exposes note objects, tempo map, time signatures and instrument/program info. It uses `mido` under the hood but provides a more convenient note-level API while preserving tick-accurate timing.  
- **Fallback / low-level layer: `mido`** — if you need to work at the message/delta-time level or you want guaranteed control over every message.  
- **Use `pretty_midi` only for analysis helpers** (tempo estimation, chroma, pitch class features) — it is easy to use but works more in seconds and sometimes hides tick-level details.

(See the full pros/cons section in the appendix.)

---

## 2. Chosen parsed-file format
**Format:** one JSON file per MIDI, with a schema that stores metadata and token arrays per track.

**Filename convention:**
```
{KEY}-{TEMPO}bpm-{sanitized_title}.tok.json
```
Example: `F#m-136bpm-very_cool_title.tok.json`

**Schema (example)**
```json
{
  "version": "1.0",
  "source_file": "original_filename.mid",
  "ppq": 480,
  "time_signatures": [{"numerator":4,"denominator":4,"tick":0}],
  "tempo_map": [{"bpm":136.0,"tick":0}],
  "key_signature": "F# minor",
  "tracks": [
    {
      "name": "Piano RH",
      "program": 0,
      "is_drum": false,
      "type": "melody",
      "tokens": ["BAR_0","POS_0","NOTE_ON_60","WAIT_48","NOTE_OFF_60"]
    }
  ],
  "global_tokens": ["TEMPO_136"]
}
```

**Token design notes:**
- Use integer tick units to avoid rounding errors. `WAIT_{ticks}` is the recommended form.  
- `NOTE_OFF_{pitch}` or alternatively encode durations by using a `DUR_{ticks}` token after `NOTE_ON`. Pick one style and document it.  
- For chords use `CHORD_{p1}_{p2}_{p3}`, all over the project we'll ignore velocities and default them inside the parser in the JSON -> MIDI to default all notes at 80 velocity

---

## 3. Token grammar & deterministic rules
Pick a precise token grammar. Here's a compact, unambiguous EBNF-like sketch:

```
DOCUMENT   ::= METADATA TRACK+
METADATA   ::= {version, ppq, tempo_map, time_signatures, key_signature, source_file}
TRACK      ::= {name, program, is_drum, type, tokens}
TOKENS     ::= TOKEN+
TOKEN      ::= BAR_{bar_index} | POS_{pos_index} | WAIT_{ticks} | TEMPO_{bpm} | NOTE_ON_{pitch} | NOTE_OFF_{pitch} | CHORD_{p1}_{p2}_... | PROG_{program}
```

**Event ordering rules (deterministic)**
At a given tick:  
1. Tempo/time-signature/key-signature changes first (apply before notes).  
2. NOTE_OFF messages next.  
3. NOTE_ON messages last.  
Within the same group sort by track index then by pitch ascending. This ordering is important to guarantee deterministic round-trip conversions.

**Quantization rules**
- Work in *ticks* using the original file's PPQ (ticks-per-quarter). Avoid converting to seconds as the canonical internal representation.  
- Quantize to a grid (e.g., 1/24 or 1/96 of a quarter) for a fixed resolution across files

---

## 4. Chord detection & track typing
**Chord detection (when to emit a `CHORD` token):**
- Group notes that start on the exact same tick and have at least `chord_threshold` notes (default 3).  
- Sort chord pitches ascending and list velocities in the same order.

**Track type detection:**
1. Try heuristics on the track name (case-insensitive): if it contains words like `melody`, `lead`, `melodia` → `melody`. If it contains `chord`, `guitar`, `piano_accomp`, `acordes` → `chord`.  
2. If track name is inconclusive, analyze event structure: if the track has >= 1 event with simultaneous notes count ≥ `chord_threshold`, mark as `chord`; otherwise `melody`.

Record the detected type in the parsed file so downstream consumers can filter or treat tracks differently.

---

## 5. Ensuring two-way fidelity (MIDI → tokens → MIDI)
To guarantee the parser works both ways, you must:

1. **Preserve all needed metadata** in the parsed file: `ppq`, `tempo_map` (list of tempo events with ticks), `time_signatures`, `instrument programs`, `channels`, `is_drum` flags.  
2. **Use ticks as canonical time units** — store all timing as integer ticks.  
3. **Keep the exact tempo map** (not just a single tempo).  
4. **Record program changes and control change events** these are irrelevant, we can skip them.
5. **Define and document deterministic ordering** for simultaneous events (see above).
6. **Round-trip validation**: implement tests that compare the original MIDI and the round-tripped MIDI (tolerance for allowed differences, e.g., removed tiny timing noise if quantized, ignore program changes). Store a small suite of metrics after conversion: total notes, number of unmatched note_on/off events, percent of notes with changed start tick, mean start offset, etc.

**Round-trip test outline:**
- For each MIDI: parse → tokens → write-MIDI → compare (by expanding both files into event lists in ticks and comparing).  
- Report mismatches. If > `X%` of events differ, flag file for manual inspection.

---

## 6. Token vocabulary choices (design tradeoffs)
You must decide what information your tokens carry. The most common choices are:

1. **Note-only tokens (pitch + durations via WAIT / NOTE_OFF):** small vocabulary, simpler, but loses velocity and expressive info.  
2. **Note + Velocity tokens (`NOTE_ON_{pitch}_{vel}`):** keeps dynamics; vocabulary grows with velocity resolution (e.g., 32 levels of velocity is common).  
3. **REMI-like or Compound Word approaches:** encode bar/position/velocity/grouped tokens which reduce sequence length and have shown empirical success for Transformers. If you plan to use existing tokenizers/models, consider standard formats like REMI or Compound Word as a starting point.  

If your primary goal is to train generative models that produce realistic music, consider adopting an established tokenization (or be compatible with one) — that will let you reuse existing tooling and pre-trained models.

---

## 7. Project structure (recommended modules)
Split the project into small, testable modules:
```
parser/
├── __init__.py
├── io.py                # read/write MIDI and JSON files (miditoolkit + mido)
├── tokenizer.py         # convert events -> tokens
├── detokenizer.py       # tokens -> events
├── quantizer.py         # quantize timings and handle tick math
├── heuristics.py        # track type detection, chord detection rules
├── validators.py        # round-trip tests and event diffs
├── config.py            # default params (ppq target, chord_threshold, velocity_bins)
├── cli.py               # CLI for batch processing (process /processed folder)
└── tests/               # unit tests & fixtures
```

---

## 8. Other important factors & gotchas
- **Time signatures & changing meters:** some tokenizations assume 4/4; decide whether you support arbitrary meters and time signature changes.  
- **Pedal (sustain) & control changes:** pedal can extend note durations; treat pedal specially if you want musical fidelity.  
- **Velocity quantization:** use a limited set of velocity buckets (e.g., 8 or 16) to reduce vocabulary. Record the mapping in metadata.  
- **Drum tracks:** usually on channel 9 — either skip them or parse them separately with a different token vocabulary.  
- **Polyphonic ambiguity & overlaps:** overlapping notes of the same pitch — decide how to represent (allow multiple note-on before note-off vs merge into one with longer duration).  
- **Micro-timing vs quantization:** choose whether to keep micro-timing or aggressively quantize to grid — store your choice.  
- **File encoding & compression:** gzip your JSON for large datasets.  
- **Licensing/copyright:** if you intend to share a dataset, verify permissions.

---

## 9. Suggested default configuration (starter)
- `ppq_target`: preserve original PPQ (no resampling).  
- `quantize_grid`: None by default; optional grid as `ppq/24` (i.e., 24 ticks per 1/4 note subdivision) if you need a fixed grid.  
- `velocity_bins`: 16 (map 0–127 into 16 bins).  
- `chord_threshold`: 3 notes.  
- `max_simultaneous_notes_for_chord`: 8.  
- `preserve_tempo_map`: true.  
- `drop_meta`: only drop lyrics and sysex by default; keep tempo/time/program/cc.

---

## 10. Validation checklist before first run
- [ ] Confirm `miditoolkit` + `mido` versions and compatibility.  
- [ ] Define token schema and document it (vocabulary list).  
- [ ] Implement round-trip unit tests on a diverse set of MIDIs (different PPQs, tempos, time signatures).  
- [ ] Implement deterministic event ordering and document it.
- [ ] Create a small reference dataset and check file sizes (JSON vs gzipped JSON).  

---

## Appendix A — Pros & cons of popular Python MIDI libraries
**miditoolkit** — pros: native-tick handling, note objects, tempo/time signature parsing, built for symbolic tasks; cons: less widespread than `mido` but built on it.  

**mido** — pros: low-level, battle-tested, explicit message & tick handling; cons: you need to implement higher-level note grouping yourself.  

**pretty_midi** — pros: high-level music-related analysis utilities (get_tempo_changes, chroma, estimate_tempo); cons: often works in seconds and hides tick detail (be careful if you need exact tick-level round-trips).

---

## Appendix B — Example minimal token sequence (illustrative)
If a piece starts with a single C4 quarter note at tick 0, PPQ=480 and tempo 120bpm, tokens could look like:

```
{
  "ppq": 480,
  "tempo_map": [{"bpm":120, "tick":0}],
  "tracks": [
    {"name":"lead","type":"melody","tokens":[
      "TEMPO_120",
      "NOTE_ON_60_100",
      "WAIT_480",
      "NOTE_OFF_60"
    ]}
  ]
}
```

## Closing notes
This design aims for clarity, determinism and round-trip fidelity while staying flexible enough to support common tokenization strategies (REMI/Compound Word) if you later move into training Transformer-based models. Start with a small, well-documented token schema and round-trip tests — then evolve toward more compact tokenizations if you need performance in model training.