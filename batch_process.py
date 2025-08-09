#!/usr/bin/env python3
"""
Batch processing script for speaker diarization.
Process multiple audio files with different configurations.
"""

import os
import argparse
import json
from datetime import datetime
from simple_diarization import SimpleSpeakerDiarization

def process_single_file(audio_path, output_dir, whisper_model="large-v3", 
                       diarization_model="pyannote/speaker-diarization-3.1"):
    """
    Process a single audio file with speaker diarization.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save results
        whisper_model: Whisper model size
        diarization_model: Diarization model name
        
    Returns:
        success: Boolean indicating if processing was successful
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        
        # Initialize diarizer with specified models
        diarizer = SimpleSpeakerDiarization(
            whisper_model=whisper_model,
            diarization_model=diarization_model
        )
        
        # Process the audio file
        results = diarizer.process_audio(audio_path, output_dir)
        
        # Print summary
        print(f"\nResults for {os.path.basename(audio_path)}:")
        print(f"- Transcription segments: {len(results)}")
        print(f"- Output directory: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False

def process_directory(input_dir, output_base_dir, whisper_model="large-v3",
                    diarization_model="pyannote/speaker-diarization-3.1",
                    file_extensions=('.wav', '.mp3', '.flac', '.m4a')):
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Directory containing audio files
        output_base_dir: Base directory for output
        whisper_model: Whisper model size
        diarization_model: Diarization model name
        file_extensions: List of audio file extensions to process
        
    Returns:
        results: Dictionary with processing results
    """
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return {}
    
    # Find all audio files
    audio_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in file_extensions):
            audio_files.append(os.path.join(input_dir, file))
    
    if not audio_files:
        print(f"No audio files found in '{input_dir}'")
        return {}
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    results = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_base_directory': output_base_dir,
        'whisper_model': whisper_model,
        'diarization_model': diarization_model,
        'files_processed': 0,
        'files_successful': 0,
        'files_failed': 0,
        'file_results': []
    }
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nProcessing file {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
        
        # Create output directory for this file
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = os.path.join(output_base_dir, file_name)
        
        # Process the file
        success = process_single_file(
            audio_file, 
            output_dir, 
            whisper_model, 
            diarization_model
        )
        
        # Record results
        file_result = {
            'file': audio_file,
            'output_directory': output_dir,
            'success': success
        }
        results['file_results'].append(file_result)
        
        if success:
            results['files_successful'] += 1
        else:
            results['files_failed'] += 1
        
        results['files_processed'] += 1
    
    return results

def save_batch_results(results, output_path="batch_results.json"):
    """
    Save batch processing results to JSON file.
    
    Args:
        results: Dictionary with processing results
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch results saved to: {output_path}")

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Batch process audio files for speaker diarization')
    
    parser.add_argument('input_dir', help='Directory containing audio files')
    parser.add_argument('output_dir', help='Base directory for output files')
    parser.add_argument('--whisper-model', default='large-v3', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3'],
                       help='Whisper model size (default: large-v3)')
    parser.add_argument('--diarization-model', default='pyannote/speaker-diarization-3.1',
                       help='Diarization model name')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.wav', '.mp3', '.flac', '.m4a'],
                       help='Audio file extensions to process')
    parser.add_argument('--save-results', action='store_true',
                       help='Save batch processing results to JSON')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BATCH SPEAKER DIARIZATION PROCESSING")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Whisper model: {args.whisper_model}")
    print(f"Diarization model: {args.diarization_model}")
    print(f"File extensions: {args.extensions}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    results = process_directory(
        args.input_dir,
        args.output_dir,
        args.whisper_model,
        args.diarization_model,
        args.extensions
    )
    
    # Print summary
    if results:
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files processed: {results['files_processed']}")
        print(f"Successful: {results['files_successful']}")
        print(f"Failed: {results['files_failed']}")
        print(f"Success rate: {results['files_successful']/results['files_processed']*100:.1f}%")
        
        if args.save_results:
            results_path = os.path.join(args.output_dir, "batch_results.json")
            save_batch_results(results, results_path)
        
        print("\nDetailed results:")
        for file_result in results['file_results']:
            status = "✓" if file_result['success'] else "✗"
            print(f"{status} {os.path.basename(file_result['file'])}")
    
    print("\nBatch processing completed!")

if __name__ == "__main__":
    main() 