"""
Track analysis and classification module.

This module provides functionality for analyzing MIDI tracks, classifying them
by type (melody, chord, bass, drums), and filtering based on musical characteristics.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from miditoolkit import MidiFile, Instrument, Note

from midi_parser.config.defaults import (
    TrackClassificationConfig,
    MidiParserConfig,
    DEFAULT_CONFIG,
    ERROR_HANDLING_STRATEGIES
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrackStatistics:
    """Statistical analysis of a MIDI track."""
    total_notes: int = 0
    unique_pitches: int = 0
    pitch_range: Tuple[int, int] = (0, 0)
    avg_pitch: float = 0.0
    avg_velocity: float = 0.0
    avg_duration: float = 0.0
    total_duration: float = 0.0
    density: float = 0.0  # Notes per second
    polyphony_ratio: float = 0.0  # Ratio of time with multiple notes
    max_polyphony: int = 0  # Maximum simultaneous notes
    avg_polyphony: float = 0.0  # Average simultaneous notes
    empty_ratio: float = 0.0  # Ratio of empty space
    note_onset_variance: float = 0.0  # Rhythmic regularity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for JSON serialization."""
        return {
            "total_notes": int(self.total_notes),
            "unique_pitches": int(self.unique_pitches),
            "pitch_range": [int(self.pitch_range[0]), int(self.pitch_range[1])],
            "avg_pitch": round(float(self.avg_pitch), 2),
            "avg_velocity": round(float(self.avg_velocity), 2),
            "avg_duration": round(float(self.avg_duration), 3),
            "total_duration": round(float(self.total_duration), 2),
            "density": round(float(self.density), 3),
            "polyphony_ratio": round(float(self.polyphony_ratio), 3),
            "max_polyphony": int(self.max_polyphony),
            "avg_polyphony": round(float(self.avg_polyphony), 2),
            "empty_ratio": round(float(self.empty_ratio), 3),
            "note_onset_variance": round(float(self.note_onset_variance), 3)
        }


@dataclass
class TrackInfo:
    """Comprehensive information about a MIDI track."""
    index: int
    name: Optional[str] = None
    program: int = 0
    is_drum: bool = False
    type: str = "unknown"  # melody, chord, bass, drums, unknown
    subtype: Optional[str] = None  # lead, rhythm, pad, etc.
    statistics: TrackStatistics = field(default_factory=TrackStatistics)
    confidence: float = 0.0  # Classification confidence (0-1)
    language_hints: List[str] = field(default_factory=list)  # Detected language keywords
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert track info to dictionary for JSON serialization."""
        return {
            "index": int(self.index),
            "name": self.name,
            "program": int(self.program),
            "is_drum": bool(self.is_drum),
            "type": self.type,
            "subtype": self.subtype,
            "statistics": self.statistics.to_dict(),
            "confidence": round(float(self.confidence), 3),
            "language_hints": self.language_hints
        }


# ============================================================================
# Track Name Patterns (Multi-language Support)
# ============================================================================

# Comprehensive track name patterns in multiple languages
TRACK_NAME_PATTERNS = {
    "melody": {
        "keywords": [
            # English
            "melody", "lead", "solo", "voice", "vocal", "main", "theme",
            # Spanish
            "melodía", "melodia", "principal", "voz", "tema",
            # Portuguese
            "melodia", "principal", "voz", "tema",
            # French
            "mélodie", "melodie", "principal", "voix", "thème",
            # Italian
            "melodia", "principale", "voce", "tema",
            # German
            "melodie", "haupt", "stimme", "thema"
        ],
        "regex": re.compile(r"(melod|lead|solo|voice|vocal|main|theme|principal|voz|voix|stimme)", re.IGNORECASE)
    },
    "chord": {
        "keywords": [
            # English
            "chord", "chords", "harmony", "harmon", "accomp", "comp", "rhythm", "pad",
            # Spanish
            "acorde", "acordes", "armonía", "armonia", "acompañamiento", "ritmo",
            # Portuguese
            "acorde", "acordes", "harmonia", "acompanhamento", "ritmo",
            # French
            "accord", "accords", "harmonie", "accompagnement", "rythme",
            # Italian
            "accordo", "accordi", "armonia", "accompagnamento", "ritmo",
            # German
            "akkord", "akkorde", "harmonie", "begleitung", "rhythmus"
        ],
        "regex": re.compile(r"(chord|accord|harmon|armon|accomp|comp|rhythm|ritmo|rythme|pad|begleit)", re.IGNORECASE)
    },
    "bass": {
        "keywords": [
            # English
            "bass", "base", "low", "bottom",
            # Spanish
            "bajo", "grave", "base", "tololoche",
            # Portuguese
            "baixo", "grave", "base",
            # French
            "basse", "grave",
            # Italian
            "basso", "grave",
            # German
            "bass", "tief"
        ],
        "regex": re.compile(r"(bass|basse|basso|bajo|baixo|base|low|grave|tief|bottom)", re.IGNORECASE)
    },
    "drums": {
        "keywords": [
            # English
            "drum", "drums", "percussion", "perc", "kit",
            # Spanish
            "batería", "bateria", "percusión", "percusion", "tambor",
            # Portuguese
            "bateria", "percussão", "percussao", "tambor",
            # French
            "batterie", "percussion", "tambour",
            # Italian
            "batteria", "percussione", "tamburo",
            # German
            "schlagzeug", "perkussion", "trommel"
        ],
        "regex": re.compile(r"(drum|bater|percussion|perc|tambor|tambour|schlagzeug|trommel|kit)", re.IGNORECASE)
    }
}


# ============================================================================
# Track Analysis Functions
# ============================================================================

def calculate_track_statistics(instrument: Instrument, ppq: int = 480) -> TrackStatistics:
    """
    Calculate comprehensive statistics for a MIDI track.
    
    Args:
        instrument: Instrument object from miditoolkit
        ppq: Pulses per quarter note for timing calculations
        
    Returns:
        TrackStatistics object with calculated metrics
    """
    stats = TrackStatistics()
    
    if not instrument.notes:
        return stats
    
    notes = instrument.notes
    stats.total_notes = len(notes)
    
    # Pitch analysis - CONVERT TO NATIVE PYTHON TYPES
    pitches = [n.pitch for n in notes]
    stats.unique_pitches = len(set(pitches))
    stats.pitch_range = (int(min(pitches)), int(max(pitches)))
    stats.avg_pitch = float(np.mean(pitches))
    
    # Velocity analysis - CONVERT TO NATIVE PYTHON TYPES
    velocities = [n.velocity for n in notes]
    stats.avg_velocity = float(np.mean(velocities))
    
    # Duration analysis - CONVERT TO NATIVE PYTHON TYPES
    durations = [n.end - n.start for n in notes]
    stats.avg_duration = float(np.mean(durations)) / ppq  # Convert to beats
    stats.total_duration = float(sum(durations)) / ppq
    
    # Time range of track
    track_start = min(n.start for n in notes)
    track_end = max(n.end for n in notes)
    track_length = (track_end - track_start) / ppq
    
    if track_length > 0:
        stats.density = float(stats.total_notes / track_length)
    
    # Polyphony analysis
    polyphony_counts = calculate_polyphony(notes)
    if polyphony_counts:
        stats.max_polyphony = int(max(polyphony_counts.keys()))
        total_time_units = sum(polyphony_counts.values())
        
        # Calculate average polyphony - CONVERT TO NATIVE PYTHON TYPES
        weighted_sum = sum(k * v for k, v in polyphony_counts.items())
        stats.avg_polyphony = float(weighted_sum / total_time_units if total_time_units > 0 else 0)
        
        # Calculate polyphony ratio (time with >1 note)
        poly_time = sum(v for k, v in polyphony_counts.items() if k > 1)
        stats.polyphony_ratio = float(poly_time / total_time_units if total_time_units > 0 else 0)
    
    # Empty space analysis
    if track_length > 0:
        covered_time = calculate_covered_time(notes) / ppq
        stats.empty_ratio = float(1.0 - (covered_time / track_length))
    
    # Rhythmic regularity (onset variance) - CONVERT TO NATIVE PYTHON TYPES
    if len(notes) > 1:
        onsets = sorted([n.start for n in notes])
        intervals = np.diff(onsets)
        if len(intervals) > 0:
            stats.note_onset_variance = float(np.std(intervals)) / ppq
    
    return stats


def calculate_polyphony(notes: List[Note]) -> Dict[int, int]:
    """
    Calculate polyphony distribution over time.
    
    Args:
        notes: List of Note objects
        
    Returns:
        Dictionary mapping polyphony level to duration at that level
    """
    if not notes:
        return {}
    
    # Create events for note starts and ends
    events = []
    for note in notes:
        events.append((note.start, 1))   # Note starts
        events.append((note.end, -1))    # Note ends
    
    # Sort by time, with note ends before note starts at the same time
    events.sort(key=lambda x: (x[0], -x[1]))
    
    # Calculate polyphony over time
    polyphony_counts = Counter()
    current_poly = 0
    last_time = events[0][0]
    
    for time, delta in events:
        if time > last_time and current_poly > 0:
            duration = time - last_time
            polyphony_counts[current_poly] += duration
        
        current_poly += delta
        last_time = time
    
    return dict(polyphony_counts)


def calculate_covered_time(notes: List[Note]) -> float:
    """
    Calculate total time covered by notes (handling overlaps).
    
    Args:
        notes: List of Note objects
        
    Returns:
        Total time in ticks covered by notes
    """
    if not notes:
        return 0.0
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda x: x.start)
    
    # Merge overlapping intervals
    merged_intervals = []
    current_start = sorted_notes[0].start
    current_end = sorted_notes[0].end
    
    for note in sorted_notes[1:]:
        if note.start <= current_end:
            # Overlapping, extend current interval
            current_end = max(current_end, note.end)
        else:
            # Non-overlapping, save current and start new
            merged_intervals.append((current_start, current_end))
            current_start = note.start
            current_end = note.end
    
    # Add the last interval
    merged_intervals.append((current_start, current_end))
    
    # Calculate total covered time
    total_time = sum(end - start for start, end in merged_intervals)
    return float(total_time)


# ============================================================================
# Track Classification Functions
# ============================================================================

def classify_track_by_name(track_name: Optional[str]) -> Tuple[Optional[str], List[str], float]:
    """
    Classify track based on its name using multi-language patterns.
    
    Args:
        track_name: Name of the track
        
    Returns:
        Tuple of (track_type, language_hints, confidence)
    """
    if not track_name:
        return None, [], 0.0
    
    track_name_lower = track_name.lower()
    detected_languages = []
    
    # Check each pattern type
    for track_type, patterns in TRACK_NAME_PATTERNS.items():
        # Check regex pattern first (faster)
        if patterns["regex"].search(track_name):
            # Find which specific keywords matched
            for keyword in patterns["keywords"]:
                if keyword.lower() in track_name_lower:
                    detected_languages.append(keyword)
            
            # Higher confidence if multiple keywords match
            confidence = min(1.0, 0.6 + 0.2 * len(detected_languages))
            return track_type, detected_languages[:3], confidence  # Limit to top 3 hints
    
    return None, [], 0.0


def classify_track_by_statistics(
    stats: TrackStatistics,
    config: TrackClassificationConfig
) -> Tuple[str, str, float]:
    """
    Classify track based on statistical analysis.
    
    Args:
        stats: Track statistics
        config: Classification configuration
        
    Returns:
        Tuple of (track_type, subtype, confidence)
    """
    # Start with moderate confidence that can be adjusted
    confidence = 0.5
    track_type = "unknown"
    subtype = None
    
    # Check for melody characteristics
    melody_score = 0.0
    
    # Monophonic or light polyphony
    if stats.avg_polyphony <= config.melody_max_polyphony:
        melody_score += 0.3
    
    # Moderate pitch range (1-2 octaves typical)
    pitch_range_semitones = stats.pitch_range[1] - stats.pitch_range[0]
    if 12 <= pitch_range_semitones <= 24:
        melody_score += 0.2
    
    # Higher pitch average (above middle C)
    if stats.avg_pitch > 60:
        melody_score += 0.2
    
    # Moderate note density
    if 1.0 <= stats.density <= 8.0:
        melody_score += 0.15
    
    # Some empty space (breathing room)
    if 0.2 <= stats.empty_ratio <= 0.6:
        melody_score += 0.15
    
    # Check for bass characteristics
    bass_score = 0.0
    
    # Low pitch range
    if stats.avg_pitch < config.bass_pitch_threshold:
        bass_score += 0.4
    
    # Mostly monophonic
    if stats.avg_polyphony < 1.5:
        bass_score += 0.2
    
    # Regular rhythmic patterns (low onset variance)
    if stats.note_onset_variance < 0.5:
        bass_score += 0.2
    
    # Moderate to low density
    if stats.density < 6.0:
        bass_score += 0.2
    
    # Check for chord characteristics
    chord_score = 0.0
    
    # High polyphony
    if stats.avg_polyphony >= config.chord_threshold:
        chord_score += 0.4
    elif stats.max_polyphony >= config.chord_threshold:
        chord_score += 0.2
    
    # Wide pitch range
    if pitch_range_semitones > 24:
        chord_score += 0.2
    
    # Long note durations
    if stats.avg_duration > 1.0:  # More than 1 beat average
        chord_score += 0.2
    
    # Less empty space
    if stats.empty_ratio < 0.3:
        chord_score += 0.2
    
    # Determine primary classification
    scores = {
        "melody": melody_score,
        "bass": bass_score,
        "chord": chord_score
    }
    
    # Get the highest scoring type
    max_score = max(scores.values())
    
    if max_score < 0.3:
        # Too low confidence, remain unknown
        return "unknown", None, 0.3
    
    # Find type with highest score
    for t, s in scores.items():
        if s == max_score:
            track_type = t
            confidence = min(0.9, max_score)
            break
    
    # Determine subtypes
    if track_type == "melody":
        if stats.avg_polyphony < 1.1:
            subtype = "monophonic"
        elif stats.avg_pitch > 72:
            subtype = "lead"
        else:
            subtype = "main"
    
    elif track_type == "chord":
        if stats.avg_duration > 2.0:
            subtype = "pad"
        elif stats.note_onset_variance < 0.3:
            subtype = "rhythm"
        else:
            subtype = "harmony"
    
    elif track_type == "bass":
        if stats.note_onset_variance < 0.2:
            subtype = "rhythmic"
        else:
            subtype = "walking"
    
    return track_type, subtype, confidence


def classify_track(
    instrument: Instrument,
    index: int,
    config: TrackClassificationConfig,
    ppq: int = 480
) -> TrackInfo:
    """
    Perform comprehensive track classification.
    
    Args:
        instrument: Instrument object to classify
        index: Track index in the MIDI file
        config: Classification configuration
        ppq: Pulses per quarter note
        
    Returns:
        TrackInfo object with classification results
    """
    # Initialize track info
    info = TrackInfo(
        index=index,
        name=instrument.name,
        program=instrument.program,
        is_drum=instrument.is_drum
    )

    # Handle drum tracks explicitly
    if instrument.is_drum or index == config.drum_channel:
        info.type = "drums"
        info.confidence = 1.0
        info.statistics = calculate_track_statistics(instrument, ppq)
        return info
    
    # Calculate statistics
    info.statistics = calculate_track_statistics(instrument, ppq)
    
    # Try name-based classification first
    name_type, lang_hints, name_confidence = classify_track_by_name(instrument.name)
    
    # Get statistics-based classification
    stat_type, subtype, stat_confidence = classify_track_by_statistics(
        info.statistics, config
    )
    
    # Combine classifications
    if name_type and name_confidence > 0.7:
        # Strong name match, use it
        info.type = name_type
        info.confidence = name_confidence
        info.language_hints = lang_hints
        info.subtype = subtype  # Still use statistical subtype
    elif name_type and stat_type == name_type:
        # Agreement between name and statistics
        info.type = name_type
        info.confidence = min(0.95, name_confidence + stat_confidence) / 2 + 0.2
        info.language_hints = lang_hints
        info.subtype = subtype
    elif stat_confidence > 0.6:
        # Use statistics-based classification
        info.type = stat_type
        info.confidence = stat_confidence
        info.subtype = subtype
    elif name_type:
        # Fall back to name if statistics inconclusive
        info.type = name_type
        info.confidence = name_confidence * 0.8  # Reduce confidence
        info.language_hints = lang_hints
    else:
        # Keep as unknown
        info.type = "unknown"
        info.confidence = 0.3
    
    return info


# ============================================================================
# Track Filtering Functions
# ============================================================================

def should_keep_track(
    track_info: TrackInfo,
    config: TrackClassificationConfig
) -> bool:
    """
    Determine if a track should be kept based on configuration criteria.
    
    Args:
        track_info: Track information and statistics
        config: Classification configuration
        
    Returns:
        True if track should be kept, False otherwise
    """
    stats = track_info.statistics
    
    # Always keep drum tracks
    if track_info.is_drum or track_info.type == "drums":
        return True
    
    # Check minimum notes threshold
    if stats.total_notes < config.min_notes_per_track:
        logger.debug(f"Filtering track {track_info.index}: "
                    f"too few notes ({stats.total_notes} < {config.min_notes_per_track})")
        return False
    
    # Check empty ratio threshold
    if stats.empty_ratio > config.max_empty_ratio:
        logger.debug(f"Filtering track {track_info.index}: "
                    f"too sparse ({stats.empty_ratio:.2f} > {config.max_empty_ratio})")
        return False
    
    # Check if track has reasonable content
    if stats.unique_pitches < 3:
        logger.debug(f"Filtering track {track_info.index}: "
                    f"too few unique pitches ({stats.unique_pitches})")
        return False
    
    # Keep tracks with sufficient confidence in classification
    if track_info.type != "unknown" or track_info.confidence > 0.4:
        return True
    
    # Default to keeping if unsure (can be made stricter)
    return True


def filter_duplicate_tracks(tracks: List[TrackInfo]) -> List[TrackInfo]:
    """
    Filter out duplicate or redundant tracks.
    
    Args:
        tracks: List of track information
        
    Returns:
        Filtered list with duplicates removed
    """
    if len(tracks) <= 1:
        return tracks
    
    filtered = []
    seen_signatures = set()
    
    for track in tracks:
        # Create a signature for duplicate detection
        signature = (
            track.program,
            track.statistics.total_notes,
            track.statistics.pitch_range,
            round(track.statistics.avg_pitch),
            round(track.statistics.avg_duration, 1)
        )
        
        if signature not in seen_signatures:
            filtered.append(track)
            seen_signatures.add(signature)
        else:
            logger.debug(f"Filtering duplicate track: {track.name or f'Track {track.index}'}")
    
    return filtered


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_tracks(
    midi: MidiFile,
    config: Optional[MidiParserConfig] = None
) -> List[TrackInfo]:
    """
    Analyze and classify all tracks in a MIDI file.
    
    This is the main entry point for track analysis as specified in
    the processing pipeline (Section 4, Step 2).
    
    Args:
        midi: MidiFile object to analyze
        config: Optional configuration (uses default if not provided)
        
    Returns:
        List of TrackInfo objects for valid tracks
        
    Example:
        >>> track_infos = analyze_tracks(midi_file)
        >>> for info in track_infos:
        >>>     print(f"Track {info.index}: {info.type} ({info.confidence:.2f})")
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not midi.instruments:
        logger.warning("No instruments found in MIDI file")
        return []
    
    logger.info(f"Analyzing {len(midi.instruments)} tracks")
    
    # Analyze each track
    track_infos = []
    for i, instrument in enumerate(midi.instruments):
        track_info = classify_track(
            instrument,
            i,
            config.track_classification,
            midi.ticks_per_beat
        )
        
        # Apply filtering
        if should_keep_track(track_info, config.track_classification):
            track_infos.append(track_info)
        else:
            strategy = ERROR_HANDLING_STRATEGIES.get("empty_tracks", "remove")
            if strategy != "remove":
                # Keep but mark as filtered
                track_info.type = "filtered"
                track_infos.append(track_info)
    
    # Filter duplicates
    track_infos = filter_duplicate_tracks(track_infos)
    
    # Log analysis results
    type_counts = Counter(t.type for t in track_infos)
    logger.info(f"Track analysis complete: {dict(type_counts)}")
    
    # Sort tracks by type priority for consistent ordering
    type_priority = {"melody": 0, "chord": 1, "bass": 2, "drums": 3, "unknown": 4, "filtered": 5}
    track_infos.sort(key=lambda x: (type_priority.get(x.type, 5), x.index))
    
    return track_infos


def get_track_by_type(
    track_infos: List[TrackInfo],
    track_type: str,
    prefer_high_confidence: bool = True
) -> Optional[TrackInfo]:
    """
    Get the best track of a specific type.
    
    Args:
        track_infos: List of analyzed tracks
        track_type: Type to search for (melody, chord, bass, drums)
        prefer_high_confidence: Whether to prefer higher confidence tracks
        
    Returns:
        Best matching TrackInfo or None if not found
    """
    matching_tracks = [t for t in track_infos if t.type == track_type]
    
    if not matching_tracks:
        return None
    
    if prefer_high_confidence:
        # Sort by confidence descending
        matching_tracks.sort(key=lambda x: x.confidence, reverse=True)
    
    return matching_tracks[0]


def get_tracks_by_types(
    track_infos: List[TrackInfo],
    track_types: List[str]
) -> List[TrackInfo]:
    """
    Get all tracks matching any of the specified types.
    
    Args:
        track_infos: List of analyzed tracks
        track_types: List of types to include
        
    Returns:
        List of matching TrackInfo objects
    """
    return [t for t in track_infos if t.type in track_types]


# Export main functions
__all__ = [
    'analyze_tracks',
    'classify_track',
    'calculate_track_statistics',
    'get_track_by_type',
    'get_tracks_by_types',
    'TrackInfo',
    'TrackStatistics',
]