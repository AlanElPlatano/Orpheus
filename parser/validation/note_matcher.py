"""
Note matching algorithms for MIDI round-trip validation.

This module provides sophisticated algorithms for matching notes between
original and reconstructed MIDI files, handling timing tolerances and
edge cases in the comparison process.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union, Tuple
from miditoolkit import Note
from .validation_metrics import NoteComparison, DEFAULT_VALIDATION_TOLERANCES

logger = logging.getLogger(__name__)


class NoteMatcher:
    """
    Handles sophisticated note matching between original and reconstructed MIDI.
    
    This class implements algorithms for finding the best matches between notes
    in two MIDI sequences, accounting for timing tolerances and handling edge
    cases like missing or extra notes.
    """
    
    def __init__(self, tolerances: Optional[Dict[str, Union[int, float]]] = None):
        """
        Initialize the note matcher.
        
        Args:
            tolerances: Dictionary of tolerance thresholds for matching
        """
        self.tolerances = tolerances or DEFAULT_VALIDATION_TOLERANCES.copy()
        
    def match_notes(
        self,
        original_notes: List[Note],
        reconstructed_notes: List[Note]
    ) -> List[NoteComparison]:
        """
        Match notes between original and reconstructed sequences.
        
        This is the main public method that coordinates the matching process.
        
        Args:
            original_notes: Notes from the original MIDI track
            reconstructed_notes: Notes from the reconstructed MIDI track
            
        Returns:
            List of NoteComparison objects with matching results
        """
        if not original_notes and not reconstructed_notes:
            return []
        
        logger.debug(f"Matching {len(original_notes)} original notes with "
                    f"{len(reconstructed_notes)} reconstructed notes")
        
        # Sort notes by start time and pitch for consistent matching
        orig_sorted = sorted(original_notes, key=lambda n: (n.start, n.pitch))
        recon_sorted = sorted(reconstructed_notes, key=lambda n: (n.start, n.pitch))
        
        # Build index for faster matching
        recon_index = self._build_note_index(recon_sorted)
        matched_recon = set()
        comparisons = []
        
        # Match each original note
        for orig_note in orig_sorted:
            best_match_idx = self._find_best_note_match(
                orig_note,
                recon_sorted,
                recon_index,
                matched_recon
            )
            
            if best_match_idx is not None:
                # Found a match
                recon_note = recon_sorted[best_match_idx]
                matched_recon.add(best_match_idx)
                
                comparison = self._create_note_comparison(orig_note, recon_note)
            else:
                # Missing note
                comparison = NoteComparison(
                    original_note=orig_note,
                    reconstructed_note=None,
                    is_missing=True
                )
            
            comparisons.append(comparison)
        
        # Add extra notes (in reconstructed but not matched)
        for i, recon_note in enumerate(recon_sorted):
            if i not in matched_recon:
                comparison = NoteComparison(
                    original_note=recon_note,  # Use recon as placeholder
                    reconstructed_note=recon_note,
                    is_extra=True
                )
                comparisons.append(comparison)
        
        logger.debug(f"Matching complete: {len(comparisons)} comparisons created")
        return comparisons
    
    def _build_note_index(self, notes: List[Note]) -> Dict[int, List[int]]:
        """
        Build index of notes by pitch for faster matching.
        
        Args:
            notes: List of notes to index
            
        Returns:
            Dictionary mapping pitch to list of note indices
        """
        index = defaultdict(list)
        for i, note in enumerate(notes):
            index[note.pitch].append(i)
        return dict(index)
    
    def _find_best_note_match(
        self,
        orig_note: Note,
        recon_notes: List[Note],
        recon_index: Dict[int, List[int]],
        matched: Set[int]
    ) -> Optional[int]:
        """
        Find the best matching reconstructed note for an original note.
        
        Uses a multi-stage matching approach:
        1. Same pitch candidates first
        2. Nearby pitch candidates if no same-pitch matches
        3. Timing-based scoring to select best match
        
        Args:
            orig_note: Original note to match
            recon_notes: List of all reconstructed notes
            recon_index: Index of reconstructed notes by pitch
            matched: Set of already matched reconstructed note indices
            
        Returns:
            Index of best matching note or None if no suitable match
        """
        candidates = self._get_match_candidates(orig_note, recon_index, matched)
        
        if not candidates:
            return None
        
        # Find best match based on timing and other factors
        return self._select_best_candidate(orig_note, recon_notes, candidates)
    
    def _get_match_candidates(
        self,
        orig_note: Note,
        recon_index: Dict[int, List[int]],
        matched: Set[int]
    ) -> List[int]:
        """
        Get candidate notes for matching based on pitch proximity.
        
        Args:
            orig_note: Original note to match
            recon_index: Index of reconstructed notes by pitch
            matched: Set of already matched indices
            
        Returns:
            List of candidate note indices
        """
        candidates = []
        
        # First priority: exact pitch match
        if orig_note.pitch in recon_index:
            for idx in recon_index[orig_note.pitch]:
                if idx not in matched:
                    candidates.append(idx)
        
        # Second priority: nearby pitches (for pitch bend effects or quantization)
        if not candidates:
            for pitch_offset in [-1, 1, -2, 2]:  # Check nearby semitones
                alt_pitch = orig_note.pitch + pitch_offset
                if alt_pitch in recon_index:
                    for idx in recon_index[alt_pitch]:
                        if idx not in matched:
                            candidates.append(idx)
        
        return candidates
    
    def _select_best_candidate(
        self,
        orig_note: Note,
        recon_notes: List[Note],
        candidates: List[int]
    ) -> Optional[int]:
        """
        Select the best candidate based on timing and similarity.
        
        Args:
            orig_note: Original note to match
            recon_notes: List of reconstructed notes
            candidates: List of candidate indices
            
        Returns:
            Index of best candidate or None
        """
        if not candidates:
            return None
        
        start_tolerance = self.tolerances.get('note_start_tick', 1)
        search_window = start_tolerance * 10  # Reasonable search window
        
        best_idx = None
        best_score = float('inf')
        
        for idx in candidates:
            recon_note = recon_notes[idx]
            score = self._calculate_match_score(orig_note, recon_note)
            
            # Only consider candidates within reasonable timing window
            time_diff = abs(orig_note.start - recon_note.start)
            if time_diff <= search_window and score < best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def _calculate_match_score(self, orig_note: Note, recon_note: Note) -> float:
        """
        Calculate similarity score between two notes (lower is better).
        
        Args:
            orig_note: Original note
            recon_note: Reconstructed note
            
        Returns:
            Similarity score (lower means more similar)
        """
        # Weight different aspects of similarity
        weights = {
            'start_time': 1.0,
            'pitch': 10.0,  # Pitch mismatches are heavily penalized
            'velocity': 0.1,
            'duration': 0.5
        }
        
        # Calculate individual differences
        start_diff = abs(orig_note.start - recon_note.start)
        pitch_diff = abs(orig_note.pitch - recon_note.pitch)
        velocity_diff = abs(orig_note.velocity - recon_note.velocity)
        duration_diff = abs((orig_note.end - orig_note.start) - 
                          (recon_note.end - recon_note.start))
        
        # Weighted score
        score = (
            weights['start_time'] * start_diff +
            weights['pitch'] * pitch_diff +
            weights['velocity'] * velocity_diff +
            weights['duration'] * duration_diff
        )
        
        return score
    
    def _create_note_comparison(
        self,
        orig_note: Note,
        recon_note: Note
    ) -> NoteComparison:
        """
        Create a NoteComparison object for matched notes.
        
        Args:
            orig_note: Original note
            recon_note: Reconstructed note
            
        Returns:
            NoteComparison with calculated differences
        """
        return NoteComparison(
            original_note=orig_note,
            reconstructed_note=recon_note,
            pitch_match=(orig_note.pitch == recon_note.pitch),
            velocity_diff=abs(orig_note.velocity - recon_note.velocity),
            start_diff=abs(orig_note.start - recon_note.start),
            duration_diff=abs((orig_note.end - orig_note.start) - 
                            (recon_note.end - recon_note.start))
        )
    
    def get_matching_stats(self, comparisons: List[NoteComparison]) -> Dict[str, int]:
        """
        Calculate statistics about the matching results.
        
        Args:
            comparisons: List of note comparisons
            
        Returns:
            Dictionary with matching statistics
        """
        stats = {
            'total_comparisons': len(comparisons),
            'exact_matches': 0,
            'pitch_matches': 0,
            'missing_notes': 0,
            'extra_notes': 0,
            'timing_issues': 0,
            'velocity_issues': 0
        }
        
        start_tolerance = self.tolerances.get('note_start_tick', 1)
        velocity_tolerance = self.tolerances.get('velocity_bin', 1)
        
        for comp in comparisons:
            if comp.is_missing:
                stats['missing_notes'] += 1
            elif comp.is_extra:
                stats['extra_notes'] += 1
            else:
                if comp.pitch_match:
                    stats['pitch_matches'] += 1
                
                if (comp.pitch_match and comp.start_diff <= start_tolerance and 
                    comp.velocity_diff <= velocity_tolerance and comp.duration_diff <= 2):
                    stats['exact_matches'] += 1
                
                if comp.start_diff > start_tolerance:
                    stats['timing_issues'] += 1
                
                if comp.velocity_diff > velocity_tolerance:
                    stats['velocity_issues'] += 1
        
        return stats
    
    def set_tolerance(self, key: str, value: Union[int, float]) -> None:
        """
        Update a specific tolerance value.
        
        Args:
            key: Tolerance parameter name
            value: New tolerance value
        """
        self.tolerances[key] = value
        logger.debug(f"Updated tolerance {key} to {value}")
    
    def get_tolerances(self) -> Dict[str, Union[int, float]]:
        """Get current tolerance settings."""
        return self.tolerances.copy()