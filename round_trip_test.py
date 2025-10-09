"""
Round-trip test script for MIDI parser validation.

Tests the complete pipeline: MIDI → JSON → MIDI
Results can be manually verified in Guitar Pro or other MIDI software.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

from miditoolkit import MidiFile

from midi_parser.config.defaults import MidiParserConfig, get_preset_config
from midi_parser.core.midi_loader import load_and_validate_midi
from midi_parser.core.track_analyzer import analyze_tracks
from midi_parser.core.tokenizer_manager import TokenizerManager
from midi_parser.core.json_serializer import (
    JSONSerializer,
    ProcessingMetadata,
    load_tokenized_json
)
from midi_parser.validation.quality_control_main import validate_tokenization_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoundTripTester:
    """Handles complete round-trip testing workflow."""
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        self.config = config or self._create_default_config()
        self.tokenizer_manager = TokenizerManager(self.config)
        self.serializer = JSONSerializer(self.config)
    
    def _create_default_config(self) -> MidiParserConfig:
        """Create configuration optimized for round-trip testing."""
        base_config = MidiParserConfig()
        
        try:
            corrido_preset = get_preset_config("use_case_presets", "corrido_demo")
            
            base_config.tokenizer.pitch_range = tuple(
                corrido_preset.get("config_overrides", {})
                .get("tokenizer", {})
                .get("pitch_range", [36, 84])
            )
            base_config.tokenizer.beat_resolution = (
                corrido_preset.get("config_overrides", {})
                .get("tokenizer", {})
                .get("beat_resolution", 4)
            )
            base_config.tokenizer.num_velocities = (
                corrido_preset.get("config_overrides", {})
                .get("tokenizer", {})
                .get("num_velocities", 8)
            )
            
            logger.info("Using corrido_demo preset configuration")
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not load corrido_demo preset: {e}. Using defaults.")
        
        return base_config
    
    def midi_to_json(
        self,
        midi_path: Path,
        output_dir: Path
    ) -> Tuple[bool, Optional[Path], Optional[dict]]:
        """
        Convert MIDI file to JSON format.
        
        Returns:
            Tuple of (success, json_path, json_data)
        """
        logger.info(f"Converting MIDI to JSON: {midi_path.name}")
        start_time = time.time()
        
        try:
            midi, metadata, validation = load_and_validate_midi(midi_path, self.config)
            
            if not validation.is_valid:
                logger.error(f"MIDI validation failed: {validation.errors}")
                return False, None, None
            
            track_infos = analyze_tracks(midi, self.config)
            
            if not track_infos:
                logger.error("No valid tracks found in MIDI file")
                return False, None, None
            
            logger.info(f"Found {len(track_infos)} tracks:")
            for track in track_infos:
                logger.info(f"  - Track {track.index}: {track.type} "
                           f"({track.statistics.total_notes} notes, "
                           f"confidence: {track.confidence:.2f})")

            # Tokenize entire MIDI file once (not per-track)
            global_result = self.tokenizer_manager.tokenize_midi(
                midi,
                strategy=self.config.tokenization,
                track_infos=track_infos,  # Pass all tracks
                auto_select=False
            )

            if not global_result.success:
                logger.error(f"Tokenization failed: {global_result.error_message}")
                return False, None, None

            logger.info(f"✓ Tokenized entire MIDI: {global_result.sequence_length} tokens")

            # Create per-track results for metadata (but tokens stay global)
            tokenization_results = [global_result for _ in track_infos]
            
            processing_metadata = ProcessingMetadata(
                processing_time_seconds=time.time() - start_time,
                validation_passed=validation.is_valid
            )
            
            json_data = self.serializer.create_output_json(
                midi_path,
                midi,
                metadata,
                track_infos,
                tokenization_results,
                validation,
                processing_metadata
            )
            
            result = self.serializer.serialize_to_file(
                json_data,
                output_dir,
                midi_path
            )
            
            if result.success:
                logger.info(f"✓ JSON saved: {result.output_path.name}")
                logger.info(f"  Size: {result.file_size_bytes / 1024:.1f}KB")
                logger.info(f"  Processing time: {time.time() - start_time:.2f}s")
                return True, result.output_path, json_data
            else:
                logger.error(f"Failed to save JSON: {result.error_message}")
                return False, None, None
                
        except Exception as e:
            logger.error(f"Error during MIDI to JSON conversion: {e}", exc_info=True)
            return False, None, None
    
    def json_to_midi(
        self,
        json_path: Path,
        output_dir: Path
    ) -> Tuple[bool, Optional[Path]]:
        """
        Convert JSON file back to MIDI format using tokens.
        
        Returns:
            Tuple of (success, midi_path)
        """
        logger.info(f"Converting JSON to MIDI: {json_path.name}")
        
        try:
            json_data = load_tokenized_json(json_path)
            
            strategy = json_data.get("tokenization", "REMI")
            tokenizer = self.tokenizer_manager.create_tokenizer(strategy)

            # Get the global token sequence (not per-track tokens)
            global_tokens = json_data.get("global_tokens", [])

            if not global_tokens:
                logger.error("No global_tokens found in JSON")
                return False, None

            logger.info(f"Reconstructing from {len(global_tokens)} global tokens")

            try:
                from miditok.classes import TokSequence
                import tempfile
                import os

                # Decode the entire sequence once
                tok_sequence = TokSequence(ids=global_tokens)
                score = tokenizer.decode([tok_sequence])

                # Convert symusic Score to miditoolkit MidiFile
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.mid', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    score.dump_midi(tmp_path)
                    reconstructed_midi = MidiFile(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"Failed to decode tokens: {e}")
                return False, None

            # Update track metadata from JSON
            tracks_data = json_data.get("tracks", [])
            total_notes_reconstructed = 0

            for i, instrument in enumerate(reconstructed_midi.instruments):
                if i < len(tracks_data):
                    track_data = tracks_data[i]
                    instrument.name = track_data.get("name", f"Track_{i}")
                    instrument.program = track_data.get("program", 0)
                    instrument.is_drum = track_data.get("is_drum", False)

                total_notes_reconstructed += len(instrument.notes)
                logger.info(f"  ✓ Track {i}: {instrument.name} ({len(instrument.notes)} notes)")
            
            if not reconstructed_midi.instruments:
                logger.error("No instruments in reconstructed MIDI")
                return False, None
            
            output_filename = json_path.stem.replace('.json', '') + '_reconstructed.mid'
            output_path = output_dir / output_filename
            
            reconstructed_midi.dump(str(output_path))
            
            logger.info(f"✓ Reconstructed MIDI saved: {output_path.name}")
            logger.info(f"  Total tracks: {len(reconstructed_midi.instruments)}")
            logger.info(f"  Total notes: {total_notes_reconstructed}")
            
            return True, output_path
            
        except Exception as e:
            logger.error(f"Error during JSON to MIDI conversion: {e}", exc_info=True)
            return False, None
    
    def run_validation_analysis(
        self,
        midi_path: Path
    ) -> Optional[dict]:
        """
        Run validation analysis and return metrics.
        
        Returns:
            Dictionary with validation metrics or None if failed
        """
        logger.info(f"Running validation analysis: {midi_path.name}")
        
        try:
            result = validate_tokenization_pipeline(
                midi_path=midi_path,
                strategy=self.config.tokenization,
                enable_quality_analysis=True,
                quality_gate="permissive",
                use_case="production"
            )
            
            logger.info(f"Validation Status: {'PASSED' if result.validation_passed else 'FAILED'}")
            logger.info(f"Quality Score: {result.overall_score:.1%}")
            
            metrics = {
                "validation_passed": result.validation_passed,
                "quality_score": result.overall_score,
                "errors": result.errors,
                "warnings": result.warnings
            }
            
            if result.round_trip_result:
                _, rt_metrics = result.round_trip_result
                
                logger.info("Round-Trip Metrics:")
                logger.info(f"  - Original notes: {rt_metrics.total_notes_original}")
                logger.info(f"  - Reconstructed notes: {rt_metrics.total_notes_reconstructed}")
                logger.info(f"  - Missing: {rt_metrics.missing_notes} ({rt_metrics.missing_notes_ratio:.1%})")
                logger.info(f"  - Extra: {rt_metrics.extra_notes} ({rt_metrics.extra_notes_ratio:.1%})")
                logger.info(f"  - Timing accuracy: {rt_metrics.timing_accuracy:.1%}")
                logger.info(f"  - Velocity accuracy: {rt_metrics.velocity_accuracy:.1%}")
                
                metrics["round_trip"] = {
                    "original_notes": rt_metrics.total_notes_original,
                    "reconstructed_notes": rt_metrics.total_notes_reconstructed,
                    "missing_notes": rt_metrics.missing_notes,
                    "missing_ratio": rt_metrics.missing_notes_ratio,
                    "extra_notes": rt_metrics.extra_notes,
                    "extra_ratio": rt_metrics.extra_notes_ratio,
                    "timing_accuracy": rt_metrics.timing_accuracy,
                    "velocity_accuracy": rt_metrics.velocity_accuracy
                }
            
            return metrics
                
        except Exception as e:
            logger.error(f"Error during validation analysis: {e}", exc_info=True)
            return None
    
    def compare_midi_files(
        self,
        original_path: Path,
        reconstructed_path: Path
    ) -> dict:
        """
        Compare original and reconstructed MIDI files.
        
        Returns:
            Dictionary with comparison statistics
        """
        logger.info("Comparing original and reconstructed MIDI files")
        
        try:
            original = MidiFile(str(original_path))
            reconstructed = MidiFile(str(reconstructed_path))
            
            comparison = {
                "original": {
                    "tracks": len(original.instruments),
                    "notes": sum(len(inst.notes) for inst in original.instruments),
                    "duration_ticks": original.max_tick,
                    "ppq": original.ticks_per_beat
                },
                "reconstructed": {
                    "tracks": len(reconstructed.instruments),
                    "notes": sum(len(inst.notes) for inst in reconstructed.instruments),
                    "duration_ticks": reconstructed.max_tick,
                    "ppq": reconstructed.ticks_per_beat
                }
            }
            
            comparison["differences"] = {
                "track_count_diff": comparison["reconstructed"]["tracks"] - comparison["original"]["tracks"],
                "note_count_diff": comparison["reconstructed"]["notes"] - comparison["original"]["notes"],
                "note_preservation_ratio": comparison["reconstructed"]["notes"] / max(comparison["original"]["notes"], 1)
            }
            
            logger.info("Comparison results:")
            logger.info(f"  Original: {comparison['original']['tracks']} tracks, "
                       f"{comparison['original']['notes']} notes")
            logger.info(f"  Reconstructed: {comparison['reconstructed']['tracks']} tracks, "
                       f"{comparison['reconstructed']['notes']} notes")
            logger.info(f"  Note preservation: "
                       f"{comparison['differences']['note_preservation_ratio']:.1%}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing MIDI files: {e}", exc_info=True)
            return {}
    
    def run_full_test(
        self,
        input_midi: Path,
        output_dir: Path
    ) -> bool:
        """
        Run complete round-trip test.
        
        Returns:
            True if test completed successfully
        """
        logger.info("="*60)
        logger.info("STARTING ROUND-TRIP TEST")
        logger.info("="*60)
        logger.info(f"Input: {input_midi}")
        logger.info(f"Output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("STEP 1: MIDI → JSON")
        logger.info("="*60)
        
        success, json_path, json_data = self.midi_to_json(input_midi, output_dir)
        
        if not success or not json_path:
            logger.error("❌ MIDI to JSON conversion failed")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("STEP 2: JSON → MIDI (Reconstruction)")
        logger.info("="*60)
        
        success, midi_path = self.json_to_midi(json_path, output_dir)
        
        if not success or not midi_path:
            logger.error("❌ JSON to MIDI conversion failed")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Validation Analysis")
        logger.info("="*60)
        
        metrics = self.run_validation_analysis(input_midi)
        
        if metrics:
            logger.info(f"✓ Validation complete - Overall score: {metrics['quality_score']:.1%}")
        else:
            logger.warning("⚠️  Validation analysis had issues (continuing anyway)")
        
        logger.info("\n" + "="*60)
        logger.info("STEP 4: File Comparison")
        logger.info("="*60)
        
        comparison = self.compare_midi_files(input_midi, midi_path)
        
        logger.info("\n" + "="*60)
        logger.info("✓ ROUND-TRIP TEST COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Original MIDI: {input_midi}")
        logger.info(f"✓ JSON output: {json_path}")
        logger.info(f"✓ Reconstructed MIDI: {midi_path}")
        
        if comparison:
            preservation = comparison.get("differences", {}).get("note_preservation_ratio", 0)
            if preservation >= 0.95:
                logger.info(f"✓ Note preservation: {preservation:.1%} (Excellent)")
            elif preservation >= 0.85:
                logger.info(f"⚠️  Note preservation: {preservation:.1%} (Good, but could be better)")
            else:
                logger.info(f"❌ Note preservation: {preservation:.1%} (Needs improvement)")
        
        logger.info("")
        logger.info("🎸 Next step: Import reconstructed MIDI into Guitar Pro")
        logger.info("   File: " + str(midi_path))
        logger.info("="*60)
        
        return True


def main():
    """Main entry point for round-trip testing."""
    
    project_root = Path(__file__).parent
    source_dir = project_root / "source_midis" / "single"
    output_dir = project_root / "processed"
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    midi_files = list(source_dir.glob("*.mid")) + list(source_dir.glob("*.midi"))
    
    if not midi_files:
        logger.error(f"No MIDI files found in {source_dir}")
        sys.exit(1)
    
    input_midi = midi_files[0]
    logger.info(f"Testing with: {input_midi.name}")
    
    tester = RoundTripTester()
    
    success = tester.run_full_test(input_midi, output_dir)
    
    if success:
        logger.info("\n✓ Round-trip test completed successfully!")
        logger.info("\nYou can now:")
        logger.info("  1. Check the JSON file in processed/")
        logger.info("  2. Import the *_reconstructed.mid file into Guitar Pro")
        sys.exit(0)
    else:
        logger.error("\n✗ Round-trip test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()