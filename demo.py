#!/usr/bin/env python3
"""
Demo script for the Whisper Speaker Diarization system.
Shows what files are available and how to use the system.
"""

import os
import json
from datetime import datetime

def show_available_files():
    """Show available audio files in the data directory."""
    print("=" * 60)
    print("AVAILABLE AUDIO FILES")
    print("=" * 60)
    
    data_dir = "data"
    if os.path.exists(data_dir):
        audio_files = [f for f in os.listdir(data_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if audio_files:
            for i, file in enumerate(audio_files, 1):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"{i}. {file} ({file_size:.1f} MB)")
        else:
            print("No audio files found in data/ directory")
    else:
        print("data/ directory not found")
    
    print()

def show_results():
    """Show existing results if any."""
    print("=" * 60)
    print("EXISTING RESULTS")
    print("=" * 60)
    
    results_dir = "results"
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) 
                       if f.endswith(('.json', '.png'))]
        
        if result_files:
            for file in result_files:
                file_path = os.path.join(results_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"✓ {file} ({file_size:.1f} KB)")
        else:
            print("No results found yet")
    else:
        print("No results directory found")
    
    print()

def show_usage_instructions():
    """Show usage instructions."""
    print("=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("1. QUICK START (No Authentication Required):")
    print("   python simple_diarization_no_auth.py")
    print("   - Provides transcription with basic speaker separation")
    print("   - Works immediately without setup")
    print()
    
    print("2. FULL VERSION (Requires Hugging Face Token):")
    print("   - Follow instructions in SETUP_FULL_VERSION.md")
    print("   - Provides accurate speaker diarization")
    print("   - Requires authentication setup")
    print()
    
    print("3. BATCH PROCESSING:")
    print("   python batch_process.py input_directory output_directory")
    print("   - Process multiple files at once")
    print()
    
    print("4. TEST INSTALLATION:")
    print("   python test_installation.py")
    print("   - Verify all dependencies are working")
    print()

def show_file_structure():
    """Show the current file structure."""
    print("=" * 60)
    print("PROJECT STRUCTURE")
    print("=" * 60)
    
    files = [
        "simple_diarization_no_auth.py - No-auth version (ready to use)",
        "simple_diarization.py - Full version (requires setup)",
        "batch_process.py - Batch processing script",
        "test_installation.py - Installation test script",
        "requirements.txt - Python dependencies",
        "README.md - Complete documentation",
        "SETUP_FULL_VERSION.md - Full version setup guide",
        "data/ - Audio files directory",
        "results/ - Output directory"
    ]
    
    for file in files:
        print(f"• {file}")
    
    print()

def show_sample_output():
    """Show a sample of the output format."""
    print("=" * 60)
    print("SAMPLE OUTPUT FORMAT")
    print("=" * 60)
    
    sample_output = {
        "timestamp": datetime.now().isoformat(),
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 5.2,
                "text": "Hello, how are you today?",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "how", "start": 0.6, "end": 0.8}
                ]
            }
        ]
    }
    
    print("JSON Output:")
    print(json.dumps(sample_output, indent=2))
    print()
    print("Files Generated:")
    print("• transcription_results.json - Detailed results")
    print("• transcription_plot.png - Timeline visualization")

def main():
    """Main demo function."""
    print("🎤 WHISPER SPEAKER DIARIZATION DEMO")
    print("=" * 60)
    print("This system provides local speaker diarization using Whisper")
    print("and pyannote.audio, inspired by the Hugging Face space.")
    print()
    
    show_available_files()
    show_results()
    show_file_structure()
    show_usage_instructions()
    show_sample_output()
    
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run the no-auth version to test: python simple_diarization_no_auth.py")
    print("2. Check the results/ directory for output files")
    print("3. For full features, follow SETUP_FULL_VERSION.md")
    print("4. Read README.md for complete documentation")
    print()
    print("Happy transcribing! 🎵")

if __name__ == "__main__":
    main() 
