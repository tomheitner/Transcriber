import os
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class WhisperSpeakerDiarization:
    def __init__(self, whisper_model="openai/whisper-large-v3", 
                 diarization_model="pyannote/speaker-diarization-3.1"):
        """
        Initialize the Whisper Speaker Diarization system.
        
        Args:
            whisper_model: Hugging Face model name for Whisper
            diarization_model: Hugging Face model name for speaker diarization
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize Whisper
        print("Loading Whisper model...")
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
        self.whisper_model.to(self.device)
        
        # Initialize Speaker Diarization
        print("Loading Speaker Diarization model...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            diarization_model,
            use_auth_token=os.getenv("HF_TOKEN")  # Set to your HF token if needed
        )
        self.diarization_pipeline.to(torch.device(self.device))
        
    def load_audio(self, audio_path):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
        """
        print(f"Loading audio from: {audio_path}")
        
        # Load audio using librosa
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        print(f"Audio loaded: {len(audio)} samples, {sample_rate} Hz")
        return audio, sample_rate
    
    def transcribe_audio(self, audio, sample_rate):
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            transcription: Whisper transcription result
        """
        print("Transcribing audio with Whisper...")
        
        # Prepare input for Whisper
        inputs = self.whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.whisper_model.generate(
                inputs["input_features"],
                max_length=448,
                return_dict_in_generate=True,
                output_scores=True,
                return_timestamps=True
            )
        
        # Decode transcription
        transcription = self.whisper_processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        
        # Extract timestamps if available
        timestamps = []
        if hasattr(generated_ids, 'sequences_timestamps'):
            timestamps = generated_ids.sequences_timestamps[0]
        
        print(f"Transcription completed: {len(transcription)} characters")
        return transcription, timestamps
    
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
    
    def align_transcription_with_speakers(self, transcription, timestamps, diarization):
        """
        Align transcription segments with speaker diarization.
        
        Args:
            transcription: Whisper transcription
            timestamps: Whisper timestamps
            diarization: Speaker diarization result
            
        Returns:
            aligned_segments: List of segments with speaker and text
        """
        print("Aligning transcription with speaker diarization...")
        
        aligned_segments = []
        
        # For now, we'll create a simple alignment
        # In a more sophisticated implementation, you'd use the timestamps
        # to match transcription segments with speaker segments
        
        # Get speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        # Simple alignment - split transcription by speakers
        # This is a simplified approach; for better results, use Whisper's timestamps
        num_speakers = len(set([seg['speaker'] for seg in speaker_segments]))
        
        # Split transcription into segments based on speaker count
        words = transcription.split()
        words_per_speaker = len(words) // num_speakers
        
        for i, speaker_seg in enumerate(speaker_segments):
            start_idx = i * words_per_speaker
            end_idx = (i + 1) * words_per_speaker if i < num_speakers - 1 else len(words)
            
            segment_text = " ".join(words[start_idx:end_idx])
            
            aligned_segments.append({
                'speaker': speaker_seg['speaker'],
                'start': speaker_seg['start'],
                'end': speaker_seg['end'],
                'text': segment_text
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
        
        # Load audio
        audio, sample_rate = self.load_audio(audio_path)
        
        # Perform transcription
        transcription, timestamps = self.transcribe_audio(audio, sample_rate)
        
        # Perform diarization
        diarization = self.perform_speaker_diarization(audio_path)
        
        # Align transcription with speakers
        aligned_segments = self.align_transcription_with_speakers(
            transcription, timestamps, diarization
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
        print(f"Transcription length: {len(transcription)} characters")
        print(f"Number of speaker segments: {len(aligned_segments)}")
        print(f"Results saved to: {output_dir}")
        print("="*50)
        
        return aligned_segments

def main():
    """Main function to run the speaker diarization system."""
    
    from dotenv import load_dotenv
    load_dotenv()
    
    print(os.getenv("HF_TOKEN"))

    # Initialize the system
    diarizer = WhisperSpeakerDiarization()
    
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