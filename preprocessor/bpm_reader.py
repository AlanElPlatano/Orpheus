import warnings
import pretty_midi


def get_tempo_from_midi(midi_file: pretty_midi.PrettyMIDI) -> float:
    """
    Extract tempo from MIDI file using a precise approach and fallbacks.

    Attempts to read tempo changes available via pretty_midi. If unavailable,
    falls back to pretty_midi's tempo estimation. As a final fallback, returns
    the default tempo 120.0 BPM.

    Args:
        midi_file: PrettyMIDI object

    Returns:
        Tempo in BPM as float
    """
    try:
        # Try to get tempo using tempo change data from pretty_midi
        if hasattr(midi_file, '_tick_scales') and midi_file._tick_scales:
            tempo_times, tempos = midi_file.get_tempo_changes()
            if len(tempos) > 0:
                tempo = tempos[0]
                return round(float(tempo), 2)
    except Exception as e:
        warnings.warn(f"Could not extract tempo from tempo changes: {e}")

    # Fallback to pretty_midi estimation
    try:
        tempo = midi_file.estimate_tempo()
        if tempo > 0:
            return round(float(tempo), 2)
    except Exception as e:
        warnings.warn(f"Could not estimate tempo: {e}")

    # Final fallback to default tempo
    warnings.warn("Could not determine tempo, using default 120 BPM")
    return 120.0