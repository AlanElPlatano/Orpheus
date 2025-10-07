"""
MIDI file comparison logic for round-trip validation.

This module handles high-level comparison between original and reconstructed
MIDI files, including track matching, global property comparison, and
coordination of detailed note-level analysis.

Strategy-aware key signature comparison to stop false warnings.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from miditoolkit import MidiFile, Instrument
from .validation_metrics import RoundTripMetrics, TrackComparison, DEFAULT_VALIDATION_TOLERANCES
from .note_matcher import NoteMatcher

logger = logging.getLogger(__name__)


class MidiComparator:
    """
    Handles comprehensive comparison between original and reconstructed MIDI files.
    
    This class orchestrates the comparison process, matching tracks between files
    and coordinating detailed note-level analysis through the NoteMatcher.
    """
    
    def __init__(
        self,
        note_matcher: Optional[NoteMatcher] = None,
        tolerances: Optional[Dict[str, Union[int, float]]] = None
    ):
        """
        Initialize the MIDI comparator.
        
        Args:
            note_matcher: NoteMatcher instance for detailed note comparison
            tolerances: Dictionary of tolerance thresholds
        """
        self.tolerances = tolerances or DEFAULT_VALIDATION_TOLERANCES.copy()
        self.note_matcher = note_matcher or NoteMatcher(self.tolerances)
        
    def compare_files(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        strategy: str = "REMI",
        detailed: bool = True
    ) -> RoundTripMetrics:
        """
        Compare original and reconstructed MIDI files comprehensively.
        
        This is the main public method that coordinates all comparison aspects.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            strategy: Tokenization strategy used (affects key signature handling)
            detailed: Whether to perform detailed note-level comparison
            
        Returns:
            RoundTripMetrics with complete comparison results
        """
        logger.info(f"Starting MIDI comparison with strategy: {strategy}")
        
        metrics = RoundTripMetrics(tokenization_strategy=strategy)
        
        # Compare global properties for each strategy
        self._compare_global_properties(original, reconstructed, metrics, strategy)
        
        # Compare tracks
        metrics.track_comparisons = self._compare_tracks(
            original.instruments,
            reconstructed.instruments,
            detailed
        )
        
        # Aggregate statistics from track comparisons
        metrics.aggregate_from_tracks()
        
        logger.info(f"Comparison complete. Overall accuracy: {metrics.overall_accuracy:.2%}")
        return metrics
    
    def _compare_global_properties(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        metrics: RoundTripMetrics,
        strategy: str = "REMI"
    ) -> None:
        """
        Compare global MIDI properties (tempo, time signatures, etc.).
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            metrics: Metrics object to update with warnings
            strategy: Tokenization strategy (affects key signature handling)
        """
        # Compare PPQ (Pulses Per Quarter note)
        if original.ticks_per_beat != reconstructed.ticks_per_beat:
            logger.warning(f"PPQ mismatch: {original.ticks_per_beat} vs {reconstructed.ticks_per_beat}")
        
        # Compare tempo changes
        self._compare_tempo_changes(original, reconstructed)
        
        # Compare time signatures
        self._compare_time_signatures(original, reconstructed)
        
        # Compare key signatures if available (strategy-aware)
        if hasattr(original, 'key_signature_changes') and hasattr(reconstructed, 'key_signature_changes'):
            self._compare_key_signatures(original, reconstructed, strategy)
    
    def _compare_tempo_changes(self, original: MidiFile, reconstructed: MidiFile) -> None:
        """Compare tempo changes between MIDI files."""
        orig_tempos = original.tempo_changes
        recon_tempos = reconstructed.tempo_changes
        
        if len(orig_tempos) != len(recon_tempos):
            logger.warning(f"Tempo change count mismatch: {len(orig_tempos)} vs {len(recon_tempos)}")
            return
        
        tempo_tolerance = self.tolerances.get('tempo_bpm_diff', 1.0)
        
        for i, (orig_tempo, recon_tempo) in enumerate(zip(orig_tempos, recon_tempos)):
            # Check timing
            time_diff = abs(orig_tempo.time - recon_tempo.time)
            if time_diff > self.tolerances.get('note_start_tick', 1):
                logger.warning(f"Tempo change {i} timing mismatch: {time_diff} ticks")
            
            # Check tempo value
            tempo_diff = abs(orig_tempo.tempo - recon_tempo.tempo)
            if tempo_diff > tempo_tolerance:
                logger.warning(f"Tempo change {i} value mismatch: {tempo_diff:.1f} BPM")
    
    def _compare_time_signatures(self, original: MidiFile, reconstructed: MidiFile) -> None:
        """Compare time signature changes between MIDI files."""
        orig_sigs = original.time_signature_changes
        recon_sigs = reconstructed.time_signature_changes
        
        if len(orig_sigs) != len(recon_sigs):
            logger.warning(f"Time signature count mismatch: {len(orig_sigs)} vs {len(recon_sigs)}")
            return
        
        for i, (orig_sig, recon_sig) in enumerate(zip(orig_sigs, recon_sigs)):
            if (orig_sig.numerator != recon_sig.numerator or 
                orig_sig.denominator != recon_sig.denominator):
                logger.warning(f"Time signature {i} mismatch: "
                             f"{orig_sig.numerator}/{orig_sig.denominator} vs "
                             f"{recon_sig.numerator}/{recon_sig.denominator}")
    
    # Strategy-aware key signature comparison
    # Not all tokenization strategies support key signatures, this function applies logic to detect this
    def _compare_key_signatures(
        self, 
        original: MidiFile, 
        reconstructed: MidiFile,
        strategy: str = "REMI"
    ) -> None:
        """
        Compare key signature changes between MIDI files with strategy awareness.
        
        Some tokenization strategies (REMI, TSD, CPWord) do not support key signatures
        by design. This method avoids false warnings for those strategies.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file  
            strategy: Tokenization strategy used
        """
        orig_keys = original.key_signature_changes
        recon_keys = reconstructed.key_signature_changes
        
        # Strategies that don't support key signatures (by design)
        strategies_without_key_sigs = ["REMI", "TSD", "CPWord"]
        
        if strategy in strategies_without_key_sigs:
            # This is EXPECTED behavior for these strategies
            if len(orig_keys) > 0:
                logger.info(
                    f"{strategy} tokenizer does not support key signature preservation. "
                    f"Original MIDI had {len(orig_keys)} key signature(s) that cannot be preserved. "
                    f"This is expected behavior and not an error."
                )
            # Don't log any warning - this is normal
            return
        
        # For strategies that DO support key signatures (MIDI-Like, Structured, Octuple)
        if len(orig_keys) != len(recon_keys):
            logger.warning(
                f"Key signature count mismatch for {strategy}: "
                f"{len(orig_keys)} original vs {len(recon_keys)} reconstructed"
            )
            return
        
        # Compare individual key signatures
        for i, (orig_key, recon_key) in enumerate(zip(orig_keys, recon_keys)):
            if orig_key.key_number != recon_key.key_number:
                logger.warning(
                    f"Key signature {i} mismatch: "
                    f"{orig_key.key_number} vs {recon_key.key_number}"
                )
    
    def _compare_tracks(
        self,
        original_tracks: List[Instrument],
        reconstructed_tracks: List[Instrument],
        detailed: bool
    ) -> List[TrackComparison]:
        """
        Compare individual tracks between MIDI files.
        
        Args:
            original_tracks: Original instrument tracks
            reconstructed_tracks: Reconstructed instrument tracks
            detailed: Whether to perform detailed note comparison
            
        Returns:
            List of TrackComparison objects
        """
        logger.debug(f"Comparing {len(original_tracks)} original tracks with "
                    f"{len(reconstructed_tracks)} reconstructed tracks")
        
        # Match tracks between original and reconstructed
        track_pairs = self._match_tracks(original_tracks, reconstructed_tracks)
        comparisons = []
        
        for orig_idx, recon_idx in track_pairs:
            comparison = self._compare_track_pair(
                original_tracks, reconstructed_tracks,
                orig_idx, recon_idx, detailed
            )
            comparisons.append(comparison)
        
        return comparisons
    
    def _match_tracks(
        self,
        original_tracks: List[Instrument],
        reconstructed_tracks: List[Instrument]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Match tracks between original and reconstructed MIDI.
        
        Uses a sophisticated matching algorithm based on:
        - Instrument program and drum status
        - Track names
        - Note count similarity
        
        Args:
            original_tracks: Original instrument tracks
            reconstructed_tracks: Reconstructed instrument tracks
            
        Returns:
            List of (original_index, reconstructed_index) pairs
        """
        pairs = []
        used_recon = set()
        
        # Try to match by multiple criteria
        for i, orig_track in enumerate(original_tracks):
            best_match = self._find_best_track_match(
                orig_track, reconstructed_tracks, used_recon
            )
            
            if best_match is not None:
                pairs.append((i, best_match))
                used_recon.add(best_match)
            else:
                pairs.append((i, None))  # No match found
        
        # Add any unmatched reconstructed tracks
        for j in range(len(reconstructed_tracks)):
            if j not in used_recon:
                pairs.append((None, j))
        
        return pairs
    
    def _find_best_track_match(
        self,
        orig_track: Instrument,
        recon_tracks: List[Instrument],
        used_indices: set
    ) -> Optional[int]:
        """
        Find the best matching reconstructed track for an original track.
        
        Args:
            orig_track: Original track to match
            recon_tracks: List of reconstructed tracks
            used_indices: Set of already matched track indices
            
        Returns:
            Index of best matching track or None
        """
        best_match = None
        best_score = 0
        min_required_score = 5  # Minimum score to consider a match
        
        for j, recon_track in enumerate(recon_tracks):
            if j in used_indices:
                continue
            
            score = self._calculate_track_similarity(orig_track, recon_track)
            
            if score > best_score and score >= min_required_score:
                best_score = score
                best_match = j
        
        return best_match
    
    def _calculate_track_similarity(
        self,
        orig_track: Instrument,
        recon_track: Instrument
    ) -> float:
        """
        Calculate similarity score between two tracks.
        
        Args:
            orig_track: Original track
            recon_track: Reconstructed track
            
        Returns:
            Similarity score (higher is more similar)
        """
        score = 0.0
        
        # Drum status match (very important)
        if orig_track.is_drum == recon_track.is_drum:
            score += 10.0
        
        # Program match (important for non-drum tracks)
        if not orig_track.is_drum and orig_track.program == recon_track.program:
            score += 5.0
        
        # Name similarity
        if orig_track.name and recon_track.name:
            if orig_track.name.lower() == recon_track.name.lower():
                score += 3.0
            elif orig_track.name.lower() in recon_track.name.lower() or \
                 recon_track.name.lower() in orig_track.name.lower():
                score += 1.5
        
        # Note count similarity
        if orig_track.notes and recon_track.notes:
            note_ratio = min(len(orig_track.notes), len(recon_track.notes)) / \
                        max(len(orig_track.notes), len(recon_track.notes))
            score += note_ratio * 2.0
        elif not orig_track.notes and not recon_track.notes:
            score += 1.0  # Both empty
        
        return score
    
    def _compare_track_pair(
        self,
        original_tracks: List[Instrument],
        reconstructed_tracks: List[Instrument],
        orig_idx: Optional[int],
        recon_idx: Optional[int],
        detailed: bool
    ) -> TrackComparison:
        """
        Compare a pair of matched tracks.
        
        Args:
            original_tracks: List of original tracks
            reconstructed_tracks: List of reconstructed tracks
            orig_idx: Index of original track (None if extra track)
            recon_idx: Index of reconstructed track (None if missing track)
            detailed: Whether to perform detailed note comparison
            
        Returns:
            TrackComparison object with results
        """
        if orig_idx is None:
            # Extra track in reconstructed
            recon_track = reconstructed_tracks[recon_idx]
            return TrackComparison(
                track_index=recon_idx,
                track_name=recon_track.name or f"Track_{recon_idx}",
                original_note_count=0,
                reconstructed_note_count=len(recon_track.notes),
                extra_notes=len(recon_track.notes)
            )
        
        orig_track = original_tracks[orig_idx]
        
        if recon_idx is None:
            # Missing track in reconstructed
            return TrackComparison(
                track_index=orig_idx,
                track_name=orig_track.name or f"Track_{orig_idx}",
                original_note_count=len(orig_track.notes),
                reconstructed_note_count=0,
                missing_notes=len(orig_track.notes)
            )
        
        # Both tracks exist - perform detailed comparison
        recon_track = reconstructed_tracks[recon_idx]
        
        comparison = TrackComparison(
            track_index=orig_idx,
            track_name=orig_track.name or f"Track_{orig_idx}",
            original_note_count=len(orig_track.notes),
            reconstructed_note_count=len(recon_track.notes),
            program_match=(orig_track.program == recon_track.program),
            is_drum_match=(orig_track.is_drum == recon_track.is_drum)
        )
        
        if detailed and orig_track.notes:
            # Detailed note-by-note comparison
            note_comparisons = self.note_matcher.match_notes(
                orig_track.notes,
                recon_track.notes
            )
            comparison.note_comparisons = note_comparisons
            
            # Count errors from detailed comparison
            self._count_track_errors(comparison)
        else:
            # Quick comparison without detailed matching
            comparison.missing_notes = max(0, len(orig_track.notes) - len(recon_track.notes))
            comparison.extra_notes = max(0, len(recon_track.notes) - len(orig_track.notes))
        
        return comparison
    
    def _count_track_errors(self, comparison: TrackComparison) -> None:
        """
        Count different types of errors from note comparisons.
        
        Args:
            comparison: TrackComparison to update with error counts
        """
        start_tolerance = self.tolerances.get('note_start_tick', 1)
        velocity_tolerance = self.tolerances.get('velocity_bin', 1)
        
        for nc in comparison.note_comparisons:
            if nc.is_missing:
                comparison.missing_notes += 1
            elif nc.is_extra:
                comparison.extra_notes += 1
            else:
                if abs(nc.start_diff) > start_tolerance:
                    comparison.timing_errors += 1
                if abs(nc.velocity_diff) > velocity_tolerance:
                    comparison.velocity_errors += 1
    
    def get_comparison_summary(
        self,
        metrics: RoundTripMetrics
    ) -> Dict[str, Union[int, float]]:
        """
        Generate a summary of comparison results.
        
        Args:
            metrics: RoundTripMetrics with comparison data
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_tracks_original': len([tc for tc in metrics.track_comparisons 
                                        if tc.original_note_count > 0]),
            'total_tracks_reconstructed': len([tc for tc in metrics.track_comparisons 
                                             if tc.reconstructed_note_count > 0]),
            'perfectly_matched_tracks': len([tc for tc in metrics.track_comparisons 
                                           if tc.accuracy_score == 1.0]),
            'avg_track_accuracy': sum(tc.accuracy_score for tc in metrics.track_comparisons) / 
                                max(len(metrics.track_comparisons), 1),
            'tracks_with_missing_notes': len([tc for tc in metrics.track_comparisons 
                                            if tc.missing_notes > 0]),
            'tracks_with_extra_notes': len([tc for tc in metrics.track_comparisons 
                                          if tc.extra_notes > 0]),
            'total_notes_original': metrics.total_notes_original,
            'total_notes_reconstructed': metrics.total_notes_reconstructed,
            'overall_accuracy': metrics.overall_accuracy
        }