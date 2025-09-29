"""
Musical feature preservation analysis for MIDI validation.

This module analyzes the preservation of musical features beyond basic round-trip
testing, including pitch distribution, rhythm patterns, harmonic content, and
tempo stability.
"""

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from miditoolkit import MidiFile, Instrument, TempoChange

from parser.core.track_analyzer import TrackInfo
from parser.validation.validation_metrics import RoundTripMetrics

logger = logging.getLogger(__name__)


@dataclass
class PitchDistributionAnalysis:
    """Analysis of pitch distribution preservation."""
    original_distribution: Dict[int, int] = field(default_factory=dict)
    reconstructed_distribution: Dict[int, int] = field(default_factory=dict)
    pitch_range: Tuple[int, int] = (0, 127)
    most_common_pitches_original: List[Tuple[int, int]] = field(default_factory=list)
    most_common_pitches_reconstructed: List[Tuple[int, int]] = field(default_factory=list)
    distribution_similarity: float = 0.0
    pitch_class_distribution_similarity: float = 0.0
    octave_distribution_similarity: float = 0.0
    range_preserved: bool = True
    tessitura_shift: int = 0  # Shift in average pitch


@dataclass
class RhythmPatternAnalysis:
    """Analysis of rhythm pattern preservation."""
    original_onset_intervals: List[int] = field(default_factory=list)
    reconstructed_onset_intervals: List[int] = field(default_factory=list)
    original_duration_distribution: Dict[int, int] = field(default_factory=dict)
    reconstructed_duration_distribution: Dict[int, int] = field(default_factory=dict)
    rhythm_similarity: float = 0.0
    groove_preserved: bool = True
    syncopation_preserved: bool = True
    tempo_stability: float = 1.0
    timing_drift: float = 0.0  # Cumulative timing drift


@dataclass
class HarmonicContentAnalysis:
    """Analysis of harmonic content preservation."""
    original_chord_progressions: List[Set[int]] = field(default_factory=list)
    reconstructed_chord_progressions: List[Set[int]] = field(default_factory=list)
    chord_similarity: float = 0.0
    voicing_preserved: bool = True
    harmonic_rhythm_preserved: bool = True
    interval_distribution_original: Dict[int, int] = field(default_factory=dict)
    interval_distribution_reconstructed: Dict[int, int] = field(default_factory=dict)
    interval_similarity: float = 0.0
    polyphony_preservation: float = 1.0


@dataclass
class MusicalFeatureMetrics:
    """Comprehensive musical feature preservation metrics."""
    pitch_analysis: PitchDistributionAnalysis = field(default_factory=PitchDistributionAnalysis)
    rhythm_analysis: RhythmPatternAnalysis = field(default_factory=RhythmPatternAnalysis)
    harmonic_analysis: HarmonicContentAnalysis = field(default_factory=HarmonicContentAnalysis)
    overall_musical_fidelity: float = 0.0
    feature_preservation_scores: Dict[str, float] = field(default_factory=dict)
    preservation_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "overall_musical_fidelity": round(self.overall_musical_fidelity, 4),
            "feature_scores": {k: round(v, 4) for k, v in self.feature_preservation_scores.items()},
            "pitch": {
                "distribution_similarity": round(self.pitch_analysis.distribution_similarity, 4),
                "pitch_class_similarity": round(self.pitch_analysis.pitch_class_distribution_similarity, 4),
                "octave_similarity": round(self.pitch_analysis.octave_distribution_similarity, 4),
                "range_preserved": self.pitch_analysis.range_preserved,
                "tessitura_shift": self.pitch_analysis.tessitura_shift
            },
            "rhythm": {
                "similarity": round(self.rhythm_analysis.rhythm_similarity, 4),
                "groove_preserved": self.rhythm_analysis.groove_preserved,
                "tempo_stability": round(self.rhythm_analysis.tempo_stability, 4),
                "timing_drift": round(self.rhythm_analysis.timing_drift, 4)
            },
            "harmony": {
                "chord_similarity": round(self.harmonic_analysis.chord_similarity, 4),
                "voicing_preserved": self.harmonic_analysis.voicing_preserved,
                "interval_similarity": round(self.harmonic_analysis.interval_similarity, 4),
                "polyphony_preservation": round(self.harmonic_analysis.polyphony_preservation, 4)
            },
            "warnings": self.preservation_warnings
        }


class MusicalFeatureAnalyzer:
    """
    Analyzes preservation of musical features in round-trip conversion.
    
    This class provides deep analysis of musical characteristics beyond
    basic note matching, ensuring musical integrity is maintained.
    """
    
    def __init__(self, beat_resolution: int = 4):
        """
        Initialize the musical feature analyzer.
        
        Args:
            beat_resolution: Beat resolution for rhythm analysis
        """
        self.beat_resolution = beat_resolution
        
    def analyze_musical_preservation(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        track_infos: Optional[List[TrackInfo]] = None,
        round_trip_metrics: Optional[RoundTripMetrics] = None
    ) -> MusicalFeatureMetrics:
        """
        Perform comprehensive musical feature preservation analysis.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            track_infos: Optional track analysis information
            round_trip_metrics: Optional round-trip validation metrics
            
        Returns:
            MusicalFeatureMetrics with comprehensive analysis
        """
        logger.info("Starting musical feature preservation analysis")
        
        metrics = MusicalFeatureMetrics()
        
        # Analyze pitch distribution preservation
        metrics.pitch_analysis = self._analyze_pitch_distribution(
            original.instruments, reconstructed.instruments
        )
        
        # Analyze rhythm pattern preservation
        metrics.rhythm_analysis = self._analyze_rhythm_patterns(
            original, reconstructed
        )
        
        # Analyze harmonic content preservation
        metrics.harmonic_analysis = self._analyze_harmonic_content(
            original.instruments, reconstructed.instruments, track_infos
        )
        
        # Calculate overall musical fidelity score
        metrics.overall_musical_fidelity = self._calculate_overall_fidelity(metrics)
        
        # Generate feature preservation scores
        metrics.feature_preservation_scores = self._calculate_feature_scores(metrics)
        
        # Generate preservation warnings
        metrics.preservation_warnings = self._generate_preservation_warnings(metrics)
        
        logger.info(f"Musical feature analysis complete. Fidelity: {metrics.overall_musical_fidelity:.2%}")
        
        return metrics
    
    def _analyze_pitch_distribution(
        self,
        original_tracks: List[Instrument],
        reconstructed_tracks: List[Instrument]
    ) -> PitchDistributionAnalysis:
        """
        Analyze pitch distribution preservation across tracks.
        
        Args:
            original_tracks: Original instrument tracks
            reconstructed_tracks: Reconstructed instrument tracks
            
        Returns:
            PitchDistributionAnalysis with results
        """
        analysis = PitchDistributionAnalysis()
        
        # Collect all pitches
        orig_pitches = []
        recon_pitches = []
        
        for track in original_tracks:
            orig_pitches.extend([note.pitch for note in track.notes])
        
        for track in reconstructed_tracks:
            recon_pitches.extend([note.pitch for note in track.notes])
        
        if not orig_pitches or not recon_pitches:
            return analysis
        
        # Calculate distributions
        analysis.original_distribution = dict(Counter(orig_pitches))
        analysis.reconstructed_distribution = dict(Counter(recon_pitches))
        
        # Most common pitches
        analysis.most_common_pitches_original = Counter(orig_pitches).most_common(10)
        analysis.most_common_pitches_reconstructed = Counter(recon_pitches).most_common(10)
        
        # Calculate pitch range
        orig_range = (min(orig_pitches), max(orig_pitches))
        recon_range = (min(recon_pitches), max(recon_pitches))
        analysis.range_preserved = (
            abs(orig_range[0] - recon_range[0]) <= 2 and
            abs(orig_range[1] - recon_range[1]) <= 2
        )
        
        # Calculate tessitura shift
        orig_mean = statistics.mean(orig_pitches)
        recon_mean = statistics.mean(recon_pitches)
        analysis.tessitura_shift = int(recon_mean - orig_mean)
        
        # Calculate distribution similarity
        analysis.distribution_similarity = self._calculate_distribution_similarity(
            analysis.original_distribution,
            analysis.reconstructed_distribution
        )
        
        # Calculate pitch class distribution similarity (ignoring octave)
        orig_pc = Counter(p % 12 for p in orig_pitches)
        recon_pc = Counter(p % 12 for p in recon_pitches)
        analysis.pitch_class_distribution_similarity = self._calculate_distribution_similarity(
            dict(orig_pc), dict(recon_pc)
        )
        
        # Calculate octave distribution similarity
        orig_octaves = Counter(p // 12 for p in orig_pitches)
        recon_octaves = Counter(p // 12 for p in recon_pitches)
        analysis.octave_distribution_similarity = self._calculate_distribution_similarity(
            dict(orig_octaves), dict(recon_octaves)
        )
        
        return analysis
    
    def _analyze_rhythm_patterns(
        self,
        original: MidiFile,
        reconstructed: MidiFile
    ) -> RhythmPatternAnalysis:
        """
        Analyze rhythm pattern preservation.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            
        Returns:
            RhythmPatternAnalysis with results
        """
        analysis = RhythmPatternAnalysis()
        
        # Collect onset times and durations
        orig_onsets = []
        orig_durations = []
        recon_onsets = []
        recon_durations = []
        
        for track in original.instruments:
            for note in track.notes:
                orig_onsets.append(note.start)
                orig_durations.append(note.end - note.start)
        
        for track in reconstructed.instruments:
            for note in track.notes:
                recon_onsets.append(note.start)
                recon_durations.append(note.end - note.start)
        
        if not orig_onsets or not recon_onsets:
            return analysis
        
        # Sort onsets for interval calculation
        orig_onsets.sort()
        recon_onsets.sort()
        
        # Calculate onset intervals (IOIs - Inter-Onset Intervals)
        analysis.original_onset_intervals = [
            orig_onsets[i+1] - orig_onsets[i] 
            for i in range(len(orig_onsets)-1)
        ]
        analysis.reconstructed_onset_intervals = [
            recon_onsets[i+1] - recon_onsets[i]
            for i in range(len(recon_onsets)-1)
        ]
        
        # Duration distributions
        analysis.original_duration_distribution = dict(Counter(orig_durations))
        analysis.reconstructed_duration_distribution = dict(Counter(recon_durations))
        
        # Calculate rhythm similarity based on IOI patterns
        if analysis.original_onset_intervals and analysis.reconstructed_onset_intervals:
            analysis.rhythm_similarity = self._calculate_rhythm_similarity(
                analysis.original_onset_intervals,
                analysis.reconstructed_onset_intervals
            )
        
        # Analyze groove preservation (rhythmic feel)
        analysis.groove_preserved = self._check_groove_preservation(
            analysis.original_onset_intervals,
            analysis.reconstructed_onset_intervals
        )
        
        # Analyze syncopation preservation
        analysis.syncopation_preserved = self._check_syncopation_preservation(
            orig_onsets, recon_onsets, original.ticks_per_beat
        )
        
        # Analyze tempo stability
        analysis.tempo_stability = self._analyze_tempo_stability(
            original.tempo_changes, reconstructed.tempo_changes
        )
        
        # Calculate timing drift
        analysis.timing_drift = self._calculate_timing_drift(
            orig_onsets, recon_onsets
        )
        
        return analysis
    
    def _analyze_harmonic_content(
        self,
        original_tracks: List[Instrument],
        reconstructed_tracks: List[Instrument],
        track_infos: Optional[List[TrackInfo]] = None
    ) -> HarmonicContentAnalysis:
        """
        Analyze harmonic content preservation.
        
        Args:
            original_tracks: Original instrument tracks
            reconstructed_tracks: Reconstructed instrument tracks
            track_infos: Optional track analysis information
            
        Returns:
            HarmonicContentAnalysis with results
        """
        analysis = HarmonicContentAnalysis()
        
        # Extract chord progressions (simplified - simultaneous notes)
        analysis.original_chord_progressions = self._extract_chord_progressions(original_tracks)
        analysis.reconstructed_chord_progressions = self._extract_chord_progressions(reconstructed_tracks)
        
        # Calculate chord similarity
        if analysis.original_chord_progressions and analysis.reconstructed_chord_progressions:
            analysis.chord_similarity = self._calculate_chord_similarity(
                analysis.original_chord_progressions,
                analysis.reconstructed_chord_progressions
            )
        
        # Analyze interval distributions
        analysis.interval_distribution_original = self._calculate_interval_distribution(original_tracks)
        analysis.interval_distribution_reconstructed = self._calculate_interval_distribution(reconstructed_tracks)
        
        # Calculate interval similarity
        analysis.interval_similarity = self._calculate_distribution_similarity(
            analysis.interval_distribution_original,
            analysis.interval_distribution_reconstructed
        )
        
        # Check voicing preservation
        analysis.voicing_preserved = self._check_voicing_preservation(
            original_tracks, reconstructed_tracks
        )
        
        # Check harmonic rhythm preservation
        analysis.harmonic_rhythm_preserved = self._check_harmonic_rhythm_preservation(
            analysis.original_chord_progressions,
            analysis.reconstructed_chord_progressions
        )
        
        # Calculate polyphony preservation
        if track_infos:
            orig_polyphony = [ti.statistics.avg_polyphony for ti in track_infos]
            # Simplified - calculate reconstructed polyphony
            recon_polyphony = [self._calculate_track_polyphony(track) for track in reconstructed_tracks]
            
            if orig_polyphony and recon_polyphony:
                avg_orig = statistics.mean(orig_polyphony)
                avg_recon = statistics.mean(recon_polyphony)
                if avg_orig > 0:
                    analysis.polyphony_preservation = min(1.0, avg_recon / avg_orig)
        
        return analysis
    
    def _calculate_distribution_similarity(
        self,
        dist1: Dict[int, int],
        dist2: Dict[int, int]
    ) -> float:
        """
        Calculate similarity between two distributions using cosine similarity.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Similarity score (0-1)
        """
        if not dist1 or not dist2:
            return 0.0
        
        # Get all keys
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        # Create vectors
        v1 = [dist1.get(k, 0) for k in all_keys]
        v2 = [dist2.get(k, 0) for k in all_keys]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        similarity = dot_product / (mag1 * mag2)
        return max(0.0, min(1.0, similarity))
    
    def _calculate_rhythm_similarity(
        self,
        ioi1: List[int],
        ioi2: List[int]
    ) -> float:
        """
        Calculate rhythm similarity based on inter-onset intervals.
        
        Args:
            ioi1: First IOI sequence
            ioi2: Second IOI sequence
            
        Returns:
            Similarity score (0-1)
        """
        if not ioi1 or not ioi2:
            return 0.0
        
        # Normalize IOIs to proportions
        total1 = sum(ioi1)
        total2 = sum(ioi2)
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        norm1 = [i / total1 for i in ioi1]
        norm2 = [i / total2 for i in ioi2]
        
        # Use dynamic time warping or simple correlation
        # Simplified: use distribution similarity
        dist1 = dict(Counter(ioi1))
        dist2 = dict(Counter(ioi2))
        
        return self._calculate_distribution_similarity(dist1, dist2)
    
    def _check_groove_preservation(
        self,
        ioi1: List[int],
        ioi2: List[int]
    ) -> bool:
        """
        Check if rhythmic groove is preserved.
        
        Args:
            ioi1: Original IOI sequence
            ioi2: Reconstructed IOI sequence
            
        Returns:
            Whether groove is preserved
        """
        if not ioi1 or not ioi2:
            return False
        
        # Check for preservation of rhythmic patterns
        # Simplified: check if most common intervals are preserved
        common1 = Counter(ioi1).most_common(5)
        common2 = Counter(ioi2).most_common(5)
        
        common1_intervals = {interval for interval, _ in common1}
        common2_intervals = {interval for interval, _ in common2}
        
        overlap = len(common1_intervals & common2_intervals)
        return overlap >= 3  # At least 3 common intervals preserved
    
    def _check_syncopation_preservation(
        self,
        onsets1: List[int],
        onsets2: List[int],
        ticks_per_beat: int
    ) -> bool:
        """
        Check if syncopation patterns are preserved.
        
        Args:
            onsets1: Original onset times
            onsets2: Reconstructed onset times
            ticks_per_beat: Ticks per beat for analysis
            
        Returns:
            Whether syncopation is preserved
        """
        if not onsets1 or not onsets2:
            return False
        
        # Calculate off-beat notes (syncopation)
        off_beat1 = sum(1 for onset in onsets1 if onset % ticks_per_beat != 0)
        off_beat2 = sum(1 for onset in onsets2 if onset % ticks_per_beat != 0)
        
        sync_ratio1 = off_beat1 / len(onsets1)
        sync_ratio2 = off_beat2 / len(onsets2)
        
        # Check if syncopation ratio is preserved (within 20% tolerance)
        return abs(sync_ratio1 - sync_ratio2) < 0.2
    
    def _analyze_tempo_stability(
        self,
        tempo_changes1: List[TempoChange],
        tempo_changes2: List[TempoChange]
    ) -> float:
        """
        Analyze tempo stability between original and reconstructed.
        
        Args:
            tempo_changes1: Original tempo changes
            tempo_changes2: Reconstructed tempo changes
            
        Returns:
            Tempo stability score (0-1)
        """
        if len(tempo_changes1) != len(tempo_changes2):
            return 0.5  # Different number of tempo changes
        
        if not tempo_changes1:
            return 1.0  # No tempo changes to compare
        
        # Compare tempo values
        tempo_diffs = []
        for t1, t2 in zip(tempo_changes1, tempo_changes2):
            if t1.tempo > 0:
                diff_ratio = abs(t1.tempo - t2.tempo) / t1.tempo
                tempo_diffs.append(diff_ratio)
        
        if not tempo_diffs:
            return 1.0
        
        avg_diff = statistics.mean(tempo_diffs)
        # Convert to stability score (1.0 = perfect, 0.0 = very unstable)
        stability = max(0.0, 1.0 - avg_diff)
        
        return stability
    
    def _calculate_timing_drift(
        self,
        onsets1: List[int],
        onsets2: List[int]
    ) -> float:
        """
        Calculate cumulative timing drift.
        
        Args:
            onsets1: Original onset times
            onsets2: Reconstructed onset times
            
        Returns:
            Average timing drift in ticks
        """
        if not onsets1 or not onsets2:
            return 0.0
        
        # Match onsets and calculate drift
        min_len = min(len(onsets1), len(onsets2))
        if min_len == 0:
            return 0.0
        
        drifts = [abs(onsets2[i] - onsets1[i]) for i in range(min_len)]
        return statistics.mean(drifts) if drifts else 0.0
    
    def _extract_chord_progressions(
        self,
        tracks: List[Instrument]
    ) -> List[Set[int]]:
        """
        Extract simplified chord progressions from tracks.
        
        Args:
            tracks: Instrument tracks
            
        Returns:
            List of chord sets (pitch classes at each time)
        """
        if not tracks:
            return []
        
        # Collect all notes with their timing
        all_notes = []
        for track in tracks:
            all_notes.extend(track.notes)
        
        if not all_notes:
            return []
        
        # Sort by start time
        all_notes.sort(key=lambda n: n.start)
        
        # Group notes that start close together (within a beat)
        chord_progressions = []
        current_chord = set()
        current_time = all_notes[0].start if all_notes else 0
        time_threshold = 10  # Ticks threshold for grouping
        
        for note in all_notes:
            if note.start - current_time <= time_threshold:
                current_chord.add(note.pitch % 12)  # Use pitch class
            else:
                if current_chord:
                    chord_progressions.append(current_chord)
                current_chord = {note.pitch % 12}
                current_time = note.start
        
        if current_chord:
            chord_progressions.append(current_chord)
        
        return chord_progressions
    
    def _calculate_chord_similarity(
        self,
        chords1: List[Set[int]],
        chords2: List[Set[int]]
    ) -> float:
        """
        Calculate similarity between chord progressions.
        
        Args:
            chords1: First chord progression
            chords2: Second chord progression
            
        Returns:
            Similarity score (0-1)
        """
        if not chords1 or not chords2:
            return 0.0
        
        # Compare chord sequences
        min_len = min(len(chords1), len(chords2))
        max_len = max(len(chords1), len(chords2))
        
        if max_len == 0:
            return 1.0
        
        matches = 0
        for i in range(min_len):
            # Calculate Jaccard similarity for each chord pair
            if chords1[i] and chords2[i]:
                intersection = len(chords1[i] & chords2[i])
                union = len(chords1[i] | chords2[i])
                if union > 0:
                    similarity = intersection / union
                    matches += similarity
        
        # Account for length difference
        length_penalty = min_len / max_len
        
        return (matches / min_len) * length_penalty if min_len > 0 else 0.0
    
    def _calculate_interval_distribution(
        self,
        tracks: List[Instrument]
    ) -> Dict[int, int]:
        """
        Calculate distribution of melodic intervals.
        
        Args:
            tracks: Instrument tracks
            
        Returns:
            Dictionary of interval counts
        """
        intervals = []
        
        for track in tracks:
            if len(track.notes) < 2:
                continue
            
            # Sort notes by time then pitch
            sorted_notes = sorted(track.notes, key=lambda n: (n.start, n.pitch))
            
            for i in range(len(sorted_notes) - 1):
                interval = abs(sorted_notes[i+1].pitch - sorted_notes[i].pitch)
                intervals.append(interval)
        
        return dict(Counter(intervals))
    
    def _check_voicing_preservation(
        self,
        tracks1: List[Instrument],
        tracks2: List[Instrument]
    ) -> bool:
        """
        Check if chord voicings are preserved.
        
        Args:
            tracks1: Original tracks
            tracks2: Reconstructed tracks
            
        Returns:
            Whether voicings are preserved
        """
        # Simplified check: compare average intervals in chords
        avg_interval1 = self._calculate_average_chord_interval(tracks1)
        avg_interval2 = self._calculate_average_chord_interval(tracks2)
        
        if avg_interval1 == 0 or avg_interval2 == 0:
            return True  # No chords to compare
        
        # Check if average intervals are similar (within 20%)
        diff_ratio = abs(avg_interval1 - avg_interval2) / avg_interval1
        return diff_ratio < 0.2
    
    def _calculate_average_chord_interval(
        self,
        tracks: List[Instrument]
    ) -> float:
        """Calculate average interval in chords."""
        intervals = []
        
        for track in tracks:
            # Find simultaneous notes
            for i, note1 in enumerate(track.notes):
                for note2 in track.notes[i+1:]:
                    # Check if notes overlap in time
                    if (note1.start <= note2.start < note1.end or
                        note2.start <= note1.start < note2.end):
                        interval = abs(note2.pitch - note1.pitch)
                        intervals.append(interval)
        
        return statistics.mean(intervals) if intervals else 0.0
    
    def _check_harmonic_rhythm_preservation(
        self,
        chords1: List[Set[int]],
        chords2: List[Set[int]]
    ) -> bool:
        """
        Check if harmonic rhythm (rate of chord changes) is preserved.
        
        Args:
            chords1: Original chord progression
            chords2: Reconstructed chord progression
            
        Returns:
            Whether harmonic rhythm is preserved
        """
        if not chords1 or not chords2:
            return True
        
        # Compare rate of chord changes
        rate1 = len(chords1)
        rate2 = len(chords2)
        
        if rate1 == 0:
            return rate2 == 0
        
        # Check if rates are similar (within 20%)
        rate_diff = abs(rate1 - rate2) / rate1
        return rate_diff < 0.2
    
    def _calculate_track_polyphony(self, track: Instrument) -> float:
        """Calculate average polyphony for a track."""
        if not track.notes:
            return 0.0
        
        # Sample at regular intervals
        max_time = max(note.end for note in track.notes)
        sample_points = range(0, max_time, 100)  # Sample every 100 ticks
        
        polyphony_samples = []
        for time_point in sample_points:
            active_notes = sum(1 for note in track.notes
                             if note.start <= time_point < note.end)
            polyphony_samples.append(active_notes)
        
        return statistics.mean(polyphony_samples) if polyphony_samples else 0.0
    
    def _calculate_overall_fidelity(self, metrics: MusicalFeatureMetrics) -> float:
        """
        Calculate overall musical fidelity score.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Overall fidelity score (0-1)
        """
        # Weight different aspects
        weights = {
            'pitch_distribution': 0.25,
            'rhythm': 0.25,
            'harmony': 0.25,
            'structure': 0.25
        }
        
        scores = {
            'pitch_distribution': (
                metrics.pitch_analysis.distribution_similarity * 0.4 +
                metrics.pitch_analysis.pitch_class_distribution_similarity * 0.4 +
                metrics.pitch_analysis.octave_distribution_similarity * 0.2
            ),
            'rhythm': (
                metrics.rhythm_analysis.rhythm_similarity * 0.4 +
                metrics.rhythm_analysis.tempo_stability * 0.3 +
                (1.0 if metrics.rhythm_analysis.groove_preserved else 0.7) * 0.3
            ),
            'harmony': (
                metrics.harmonic_analysis.chord_similarity * 0.4 +
                metrics.harmonic_analysis.interval_similarity * 0.3 +
                metrics.harmonic_analysis.polyphony_preservation * 0.3
            ),
            'structure': (
                (1.0 if metrics.pitch_analysis.range_preserved else 0.5) * 0.5 +
                (1.0 if metrics.harmonic_analysis.voicing_preserved else 0.7) * 0.5
            )
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        return max(0.0, min(1.0, overall))
    
    def _calculate_feature_scores(self, metrics: MusicalFeatureMetrics) -> Dict[str, float]:
        """
        Calculate individual feature preservation scores.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Dictionary of feature scores
        """
        return {
            'pitch_distribution': metrics.pitch_analysis.distribution_similarity,
            'pitch_class_preservation': metrics.pitch_analysis.pitch_class_distribution_similarity,
            'octave_preservation': metrics.pitch_analysis.octave_distribution_similarity,
            'rhythm_similarity': metrics.rhythm_analysis.rhythm_similarity,
            'tempo_stability': metrics.rhythm_analysis.tempo_stability,
            'groove_preservation': 1.0 if metrics.rhythm_analysis.groove_preserved else 0.0,
            'chord_similarity': metrics.harmonic_analysis.chord_similarity,
            'interval_preservation': metrics.harmonic_analysis.interval_similarity,
            'polyphony_preservation': metrics.harmonic_analysis.polyphony_preservation,
            'voicing_preservation': 1.0 if metrics.harmonic_analysis.voicing_preserved else 0.0
        }
    
    def _generate_preservation_warnings(self, metrics: MusicalFeatureMetrics) -> List[str]:
        """
        Generate warnings about preservation issues.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Pitch warnings
        if metrics.pitch_analysis.distribution_similarity < 0.8:
            warnings.append(f"Pitch distribution poorly preserved (similarity: {metrics.pitch_analysis.distribution_similarity:.2f})")
        
        if abs(metrics.pitch_analysis.tessitura_shift) > 3:
            warnings.append(f"Significant tessitura shift detected: {metrics.pitch_analysis.tessitura_shift} semitones")
        
        if not metrics.pitch_analysis.range_preserved:
            warnings.append("Pitch range not preserved")
        
        # Rhythm warnings
        if metrics.rhythm_analysis.rhythm_similarity < 0.7:
            warnings.append(f"Rhythm patterns poorly preserved (similarity: {metrics.rhythm_analysis.rhythm_similarity:.2f})")
        
        if not metrics.rhythm_analysis.groove_preserved:
            warnings.append("Rhythmic groove not preserved")
        
        if metrics.rhythm_analysis.tempo_stability < 0.9:
            warnings.append(f"Tempo instability detected (stability: {metrics.rhythm_analysis.tempo_stability:.2f})")
        
        if metrics.rhythm_analysis.timing_drift > 10:
            warnings.append(f"Significant timing drift: {metrics.rhythm_analysis.timing_drift:.1f} ticks average")
        
        # Harmony warnings
        if metrics.harmonic_analysis.chord_similarity < 0.7:
            warnings.append(f"Chord progressions poorly preserved (similarity: {metrics.harmonic_analysis.chord_similarity:.2f})")
        
        if not metrics.harmonic_analysis.voicing_preserved:
            warnings.append("Chord voicings not preserved")
        
        if metrics.harmonic_analysis.polyphony_preservation < 0.8:
            warnings.append(f"Polyphony not well preserved (preservation: {metrics.harmonic_analysis.polyphony_preservation:.2f})")
        
        return warnings