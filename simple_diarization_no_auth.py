#!/usr/bin/env python3
"""
Simple Speaker Diarization without authentication requirements.
This version provides alternatives for users who don't have Hugging Face tokens.
"""

import os
import torch
import numpy as np
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
import json
from datetime import datetime

class SimpleSpeakerDiarizationNoAuth:
    def __init__(self, whisper_model="large-v3"):
        """
        Initialize the Simple Speaker Diarization system without authentication.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v3)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize Whisper
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel(
            whisper_model, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        
        print("Note: This version uses Whisper transcription with basic speaker separation.")
        print("For full speaker diarization, you need to:")
        print("1. Get a Hugging Face token from https://huggingface.co/settings/tokens")
        print("2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("3. Use the full version with authentication")
        
    def transcribe_with_timestamps(self, audio_path):
        """
        Transcribe audio with word-level timestamps using faster-whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            segments: List of transcription segments with timestamps
        """
        print("Transcribing audio with Whisper...")
        
        # Transcribe with word-level timestamps
        segments, info = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None  # Auto-detect language
        )
        
        # Convert to list for easier processing
        segments_list = []
        for segment in segments:
            segment_dict = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': []
            }
            
            # Add word-level timestamps
            for word in segment.words:
                segment_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end
                })
            
            segments_list.append(segment_dict)
        
        print(f"Transcription completed: {len(segments_list)} segments")
        return segments_list
    
    def simple_speaker_separation(self, transcription_segments):
        """
        Simple speaker separation based on transcription segments.
        This is a basic approach that alternates speakers based on segments.
        
        Args:
            transcription_segments: List of transcription segments
            
        Returns:
            speaker_segments: List of segments with assigned speakers
        """
        print("Performing simple speaker separation...")
        
        speaker_segments = []
        current_speaker = "SPEAKER_00"
        
        for i, segment in enumerate(transcription_segments):
            # Alternate speakers every few segments
            if i > 0 and i % 3 == 0:  # Change speaker every 3 segments
                current_speaker = f"SPEAKER_{int(i/3):02d}"
            
            speaker_segments.append({
                'speaker': current_speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'words': segment['words']
            })
        
        return speaker_segments
    
    def visualize_transcription(self, segments, output_path="transcription_plot.png"):
        """
        Create a visualization of the transcription segments.
        
        Args:
            segments: List of transcription segments
            output_path: Path to save the visualization
        """
        print("Creating transcription visualization...")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot segments
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        
        for i, segment in enumerate(segments):
            ax.barh(segment['speaker'], segment['end'] - segment['start'], 
                   left=segment['start'], alpha=0.8, color=colors[i])
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Speaker')
        ax.set_title('Transcription Timeline (Simple Speaker Separation)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
    
    def save_results(self, segments, output_path="transcription_results.json"):
        """
        Save transcription results to JSON file.
        
        Args:
            segments: List of segments
            output_path: Path to save the results
        """
        print(f"Saving results to: {output_path}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'segments': segments
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Results saved successfully")
    
    def process_audio(self, audio_path, output_dir="results"):
        """
        Complete pipeline to process audio file with transcription.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform transcription with timestamps
        transcription_segments = self.transcribe_with_timestamps(audio_path)
        
        # Perform simple speaker separation
        speaker_segments = self.simple_speaker_separation(transcription_segments)
        
        # Create visualization
        viz_path = os.path.join(output_dir, "transcription_plot.png")
        self.visualize_transcription(speaker_segments, viz_path)
        
        # Save results
        results_path = os.path.join(output_dir, "transcription_results.json")
        self.save_results(speaker_segments, results_path)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Audio file: {audio_path}")
        print(f"Number of transcription segments: {len(transcription_segments)}")
        print(f"Number of speaker segments: {len(speaker_segments)}")
        print(f"Results saved to: {output_dir}")
        print("="*50)
        
        return speaker_segments

def main():
    """Main function to run the transcription system."""
    
    # Initialize the system
    diarizer = SimpleSpeakerDiarizationNoAuth()
    
    # Process audio files in the data directory
    data_dir = "data"
    output_dir = "results"
    
    if os.path.exists(data_dir):
        audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        for audio_file in audio_files:
            audio_path = os.path.join(data_dir, audio_file)
            print(f"\nProcessing: {audio_file}")
            
            try:
                results = diarizer.process_audio(audio_path, output_dir)
                
                # Print results
                print("\nTranscription segments:")
                for segment in results:
                    print(f"Speaker {segment['speaker']} ({segment['start']:.2f}s - {segment['end']:.2f}s): {segment['text']}")
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    else:
        print(f"Data directory '{data_dir}' not found. Please place audio files in the data directory.")

if __name__ == "__main__":
    main() 