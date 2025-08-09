import os
import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import matplotlib.pyplot as plt
import json
from datetime import datetime

class SimpleSpeakerDiarization:
    def __init__(self, whisper_model="large-v3", 
                 diarization_model="pyannote/speaker-diarization-3.1"):
        """
        Initialize the Simple Speaker Diarization system.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v3)
            diarization_model: Hugging Face model name for speaker diarization
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
        
        # Initialize Speaker Diarization
        print("Loading Speaker Diarization model...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            diarization_model,
            use_auth_token=os.getenv("HF_TOKEN")  # Set to your HF token if needed
        )
        self.diarization_pipeline.to(torch.device(self.device))
        
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
            language="auto"
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
    
    def perform_speaker_diarization(self, audio_path):
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            diarization: Diarization result
        """
        print("Performing speaker diarization...")
        
        # Run diarization pipeline
        with ProgressHook() as hook:
            diarization = self.diarization_pipeline(audio_path, hook=hook)
        
        print("Speaker diarization completed")
        return diarization
    
    def align_transcription_with_speakers(self, transcription_segments, diarization):
        """
        Align transcription segments with speaker diarization.
        
        Args:
            transcription_segments: List of transcription segments with timestamps
            diarization: Speaker diarization result
            
        Returns:
            aligned_segments: List of segments with speaker and text
        """
        print("Aligning transcription with speaker diarization...")
        
        aligned_segments = []
        
        # Get speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        # Align transcription segments with speaker segments
        for trans_segment in transcription_segments:
            trans_start = trans_segment['start']
            trans_end = trans_segment['end']
            
            # Find which speaker segment overlaps most with this transcription segment
            best_speaker = None
            max_overlap = 0
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(trans_start, speaker_seg['start'])
                overlap_end = min(trans_end, speaker_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_seg['speaker']
            
            # If no clear overlap, assign to the closest speaker
            if best_speaker is None:
                min_distance = float('inf')
                for speaker_seg in speaker_segments:
                    # Calculate distance to segment center
                    speaker_center = (speaker_seg['start'] + speaker_seg['end']) / 2
                    trans_center = (trans_start + trans_end) / 2
                    distance = abs(speaker_center - trans_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = speaker_seg['speaker']
            
            aligned_segments.append({
                'speaker': best_speaker,
                'start': trans_start,
                'end': trans_end,
                'text': trans_segment['text'],
                'words': trans_segment['words']
            })
        
        return aligned_segments
    
    def visualize_diarization(self, diarization, output_path="diarization_plot.png"):
        """
        Create a visualization of the speaker diarization.
        
        Args:
            diarization: Diarization result
            output_path: Path to save the visualization
        """
        print("Creating diarization visualization...")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot speaker segments
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            ax.barh(speaker, turn.end - turn.start, left=turn.start, 
                   alpha=0.8, label=speaker)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Speaker')
        ax.set_title('Speaker Diarization Timeline')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
    
    def save_results(self, aligned_segments, output_path="diarization_results.json"):
        """
        Save diarization results to JSON file.
        
        Args:
            aligned_segments: List of aligned segments
            output_path: Path to save the results
        """
        print(f"Saving results to: {output_path}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'segments': aligned_segments
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Results saved successfully")
    
    def process_audio(self, audio_path, output_dir="results"):
        """
        Complete pipeline to process audio file with transcription and diarization.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform transcription with timestamps
        transcription_segments = self.transcribe_with_timestamps(audio_path)
        
        # Perform diarization
        diarization = self.perform_speaker_diarization(audio_path)
        
        # Align transcription with speakers
        aligned_segments = self.align_transcription_with_speakers(
            transcription_segments, diarization
        )
        
        # Create visualization
        viz_path = os.path.join(output_dir, "diarization_plot.png")
        self.visualize_diarization(diarization, viz_path)
        
        # Save results
        results_path = os.path.join(output_dir, "diarization_results.json")
        self.save_results(aligned_segments, results_path)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Audio file: {audio_path}")
        print(f"Number of transcription segments: {len(transcription_segments)}")
        print(f"Number of aligned segments: {len(aligned_segments)}")
        print(f"Results saved to: {output_dir}")
        print("="*50)
        
        return aligned_segments


def main():
    """Main function to run the speaker diarization system."""

    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize the system
    diarizer = SimpleSpeakerDiarization()
    
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
                print("\nAligned segments:")
                for segment in results:
                    print(f"Speaker {segment['speaker']} ({segment['start']:.2f}s - {segment['end']:.2f}s): {segment['text']}")
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    else:
        print(f"Data directory '{data_dir}' not found. Please place audio files in the data directory.")

if __name__ == "__main__":
    main() 