from pathlib import Path

from batch import validate_batch


def test_validation():
    """
    Test function to validate MIDI files in the source_midis folder.
    """
    # Get path to source_midis folder
    file_path = Path(__file__).resolve().parent
    file_path = file_path.parent
    source_dir = file_path / "source_midis"
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Please create the source_midis folder and add some MIDI files to test.")
        return
        
    # Validate all files in the directory
    results = validate_batch(str(source_dir), verbose=True)
    
    # Print detailed results for invalid files
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    for file_path, result in results['results'].items():
        file_name = Path(file_path).name
        
        if not result['is_valid']:
            print(f"\n❌ {file_name}")
            print(f"   Issues: {len(result['issues'])}")
            for issue in result['issues']:
                print(f"     • {issue}")
                
        elif result['warnings']:
            print(f"\n⚠️  {file_name}")
            print(f"   Warnings: {len(result['warnings'])}")
            for warning in result['warnings'][:3]:  # Show first 3 warnings
                print(f"     • {warning}")
            if len(result['warnings']) > 3:
                print(f"     • ... and {len(result['warnings']) - 3} more warnings")
                
        else:
            print(f"\n✅ {file_name}")
            if result['stats']:
                stats = result['stats']
                print(f"   {stats['total_notes']} notes, "
                      f"{stats['duration_seconds']:.1f}s, "
                      f"{stats['tempo_bpm']:.1f} BPM")


if __name__ == "__main__":
    test_validation()