#!/usr/bin/env python3
"""
Real-time Speaker Diarization with chunked processing.
This version processes audio in small chunks for faster, more responsive transcription.
"""

import os
import torch
import numpy as np
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
import json
from datetime import datetime
import librosa
import soundfile as sf
from tqdm import tqdm
import threading
import time

class RealtimeSpeakerDiarization:
    def __init__(self, whisper_model="base", chunk_duration=5, overlap=1):
        """
        Initialize the Real-time Speaker Diarization system.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v3)
            chunk_duration: Duration of each audio chunk in seconds
            overlap: Overlap between chunks in seconds
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        
        print(f"Using device: {self.device}")
        print(f"Chunk duration: {chunk_duration}s, Overlap: {overlap}s")
        
        # Initialize Whisper with smaller model for faster processing
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel(
            whisper_model, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        
        # Performance tracking
        self.processing_times = []
        self.total_audio_duration = 0
        
    def load_audio(self, audio_path):
        """Load audio file and return audio data and sample rate."""
        print(f"Loading audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        self.total_audio_duration = len(audio) / sr
        print(f"Audio duration: {self.total_audio_duration:.2f} seconds")
        return audio, sr
    
    def create_audio_chunks(self, audio, sr):
        """Split audio into overlapping chunks."""
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap * sr)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        chunk_times = []
        
        for start_sample in range(0, len(audio) - chunk_samples + 1, step_samples):
            end_sample = start_sample + chunk_samples
            chunk = audio[start_sample:end_sample]
            
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            chunks.append(chunk)
            chunk_times.append((start_time, end_time))
        
        print(f"Created {len(chunks)} chunks for processing")
        return chunks, chunk_times
    
    def transcribe_chunk(self, chunk, chunk_time, chunk_index):
        """Transcribe a single audio chunk."""
        start_time = time.time()
        
        # Convert numpy array to temporary file for Whisper
        temp_path = f"temp_chunk_{chunk_index}.wav"
        sf.write(temp_path, chunk, 16000)
        
        # Transcribe with word-level timestamps
        segments, info = self.whisper_model.transcribe(
            temp_path,
            word_timestamps=True,
            language=None
        )
        
        # Convert to list and adjust timestamps
        segments_list = []
        chunk_start, chunk_end = chunk_time
        
        for segment in segments:
            # Adjust timestamps to global time
            global_start = chunk_start + segment.start
            global_end = chunk_start + segment.end
            
            segment_dict = {
                'start': global_start,
                'end': global_end,
                'text': segment.text.strip(),
                'words': []
            }
            
            # Add word-level timestamps
            for word in segment.words:
                segment_dict['words'].append({
                    'word': word.word,
                    'start': chunk_start + word.start,
                    'end': chunk_start + word.end
                })
            
            segments_list.append(segment_dict)
        
        # Clean up temp file
        os.remove(temp_path)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Calculate speed ratio
        chunk_duration = chunk_end - chunk_start
        speed_ratio = chunk_duration / processing_time if processing_time > 0 else 0
        
        print(f"Chunk {chunk_index + 1}: {chunk_duration:.1f}s processed in {processing_time:.2f}s ({speed_ratio:.1f}x real-time)")
        
        return segments_list
    
    def merge_overlapping_segments(self, all_segments):
        """Merge segments that overlap due to chunk processing."""
        if not all_segments:
            return all_segments
        
        merged = []
        current_segment = all_segments[0].copy()
        
        for next_segment in all_segments[1:]:
            # Check for overlap
            if (next_segment['start'] <= current_segment['end'] + 0.5 and  # 0.5s tolerance
                next_segment['text'].strip()):
                
                # Merge overlapping segments
                current_segment['end'] = max(current_segment['end'], next_segment['end'])
                current_segment['text'] += ' ' + next_segment['text']
                
                # Merge words
                current_segment['words'].extend(next_segment['words'])
            else:
                # No overlap, save current and start new
                if current_segment['text'].strip():
                    merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        if current_segment['text'].strip():
            merged.append(current_segment)
        
        return merged
    
    def simple_speaker_separation(self, transcription_segments):
        """Simple speaker separation based on transcription segments."""
        print("Performing simple speaker separation...")
        
        speaker_segments = []
        current_speaker = "SPEAKER_00"
        
        for i, segment in enumerate(transcription_segments):
            # Alternate speakers every few segments
            if i > 0 and i % 3 == 0:
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
        """Create a visualization of the transcription segments."""
        print("Creating transcription visualization...")
        
        fig, ax = plt.subplots(figsize=(15, 5))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        
        for i, segment in enumerate(segments):
            ax.barh(segment['speaker'], segment['end'] - segment['start'], 
                   left=segment['start'], alpha=0.8, color=colors[i])
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Speaker')
        ax.set_title('Real-time Transcription Timeline')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
    
    def save_results(self, segments, output_path="transcription_results.json", timing_info=None):
        """Save transcription results to JSON file."""
        print(f"Saving results to: {output_path}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'segments': segments,
            'timing': timing_info,
            'performance': {
                'average_processing_time': np.mean(self.processing_times),
                'total_processing_time': sum(self.processing_times),
                'speed_ratio': self.total_audio_duration / sum(self.processing_times) if self.processing_times else 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Results saved successfully")
    
    def process_audio_realtime(self, audio_path, output_dir="results"):
        """Process audio file with real-time chunked processing."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Start timing
        start_time = datetime.now()
        
        # Load and chunk audio
        audio, sr = self.load_audio(audio_path)
        chunks, chunk_times = self.create_audio_chunks(audio, sr)
        
        # Process chunks with progress bar
        all_segments = []
        print("\nProcessing chunks:")
        
        for i, (chunk, chunk_time) in enumerate(tqdm(zip(chunks, chunk_times), total=len(chunks))):
            segments = self.transcribe_chunk(chunk, chunk_time, i)
            all_segments.extend(segments)
        
        # Merge overlapping segments
        print("Merging overlapping segments...")
        merged_segments = self.merge_overlapping_segments(all_segments)
        
        # End timing
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        # Calculate timing metrics
        timing_info = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': processing_duration,
            'audio_duration': self.total_audio_duration,
            'speed_ratio': self.total_audio_duration / processing_duration if processing_duration > 0 else 0
        }
        
        # Perform simple speaker separation
        speaker_segments = self.simple_speaker_separation(merged_segments)
        
        # Separate directory for output per audio file
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        audio_output_dir = os.path.join(output_dir, audio_filename)
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Create visualization
        viz_path = os.path.join(audio_output_dir, "transcription_plot.png")
        self.visualize_transcription(speaker_segments, viz_path)
        
        # Save results with timing information
        results_path = os.path.join(audio_output_dir, "transcription_results.json")
        self.save_results(speaker_segments, results_path, timing_info)
        
        # Print performance summary
        print("\n" + "="*60)
        print("REAL-TIME PROCESSING COMPLETE")
        print("="*60)
        print(f"Audio file: {audio_path}")
        print(f"Audio duration: {self.total_audio_duration:.2f} seconds")
        print(f"Processing time: {processing_duration:.2f} seconds")
        print(f"Speed ratio: {timing_info['speed_ratio']:.1f}x real-time")
        print(f"Average chunk processing time: {np.mean(self.processing_times):.2f}s")
        print(f"Number of chunks processed: {len(chunks)}")
        print(f"Number of transcription segments: {len(merged_segments)}")
        print(f"Number of speaker segments: {len(speaker_segments)}")
        print(f"Results saved to: {audio_output_dir}")
        print("="*60)
        
        return speaker_segments

def main():
    """Main function to run the real-time transcription system."""
    
    # Initialize the system with smaller chunks for faster processing
    diarizer = RealtimeSpeakerDiarization(
        whisper_model="base",  # Use base model for faster processing
        chunk_duration=5,      # 5-second chunks
        overlap=1              # 1-second overlap
    )
    
    # Process audio files in the data directory
    data_dir = "data"
    output_dir = "results"
    
    if os.path.exists(data_dir):
        audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        for audio_file in audio_files:
            audio_path = os.path.join(data_dir, audio_file)
            print(f"\n{'='*60}")
            print(f"Processing: {audio_file}")
            print(f"{'='*60}")
            
            try:
                results = diarizer.process_audio_realtime(audio_path, output_dir)
                
                # Print first few results as preview
                print("\nTranscription preview:")
                for i, segment in enumerate(results[:5]):  # Show first 5 segments
                    print(f"Speaker {segment['speaker']} ({segment['start']:.2f}s - {segment['end']:.2f}s): {segment['text']}")
                
                if len(results) > 5:
                    print(f"... and {len(results) - 5} more segments")
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    else:
        print(f"Data directory '{data_dir}' not found. Please place audio files in the data directory.")

if __name__ == "__main__":
    main() 