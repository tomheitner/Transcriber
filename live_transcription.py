#!/usr/bin/env python3
"""
Live Real-time Transcription with microphone input.
This version processes live audio from microphone for immediate transcription.
"""

import os
import torch
import numpy as np
from faster_whisper import WhisperModel
import pyaudio
import wave
import threading
import queue
import time
import json
from datetime import datetime

class LiveTranscription:
    def __init__(self, whisper_model="tiny", chunk_duration=3, save_output=True, output_dir="live_sessions"):
        """
        Initialize the Live Transcription system.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v3)
            chunk_duration: Duration of each audio chunk in seconds
            save_output: Whether to save transcription to files
            output_dir: Directory to save transcription files
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chunk_duration = chunk_duration
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.save_output = save_output
        self.output_dir = output_dir
        
        # Create output directory if saving is enabled
        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Session tracking
        self.session_start_time = None
        self.session_transcriptions = []
        self.session_audio_chunks = []
        
        print(f"Using device: {self.device}")
        print(f"Chunk duration: {chunk_duration}s")
        print(f"Save output: {save_output}")
        if save_output:
            print(f"Output directory: {output_dir}")
        
        # Initialize Whisper with tiny model for fastest processing
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel(
            whisper_model, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper expects 16kHz
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input."""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Start recording from microphone."""
        print("Starting live transcription...")
        print("Speak into your microphone. Press Ctrl+C to stop.")
        
        # Initialize session
        self.session_start_time = datetime.now()
        self.session_transcriptions = []
        self.session_audio_chunks = []
        
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
        # Start processing threads
        self.audio_thread = threading.Thread(target=self.process_audio_chunks)
        self.transcription_thread = threading.Thread(target=self.process_transcriptions)
        
        self.audio_thread.start()
        self.transcription_thread.start()
        
        try:
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and cleanup."""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        self.p.terminate()
        print("Recording stopped.")
        
        # Save session if enabled
        if self.save_output and self.session_transcriptions:
            self.save_session()
    
    def save_session(self):
        """Save the current session to files."""
        if not self.session_transcriptions:
            print("No transcriptions to save.")
            return
        
        session_end_time = datetime.now()
        session_duration = (session_end_time - self.session_start_time).total_seconds()
        
        # Create session filename
        session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save transcriptions to JSON
        json_filename = f"live_session_{session_id}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        session_data = {
            'session_id': session_id,
            'start_time': self.session_start_time.isoformat(),
            'end_time': session_end_time.isoformat(),
            'duration_seconds': session_duration,
            'total_transcriptions': len(self.session_transcriptions),
            'transcriptions': self.session_transcriptions
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Save transcriptions to text file
        txt_filename = f"live_session_{session_id}.txt"
        txt_path = os.path.join(self.output_dir, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Live Transcription Session\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {session_duration:.2f} seconds\n")
            f.write(f"Total Transcriptions: {len(self.session_transcriptions)}\n")
            f.write("="*50 + "\n\n")
            
            for i, trans in enumerate(self.session_transcriptions, 1):
                f.write(f"[{i:03d}] {trans['timestamp']} - {trans['text']}\n")
        
        print(f"\nSession saved:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
        print(f"  Duration: {session_duration:.2f} seconds")
        print(f"  Transcriptions: {len(self.session_transcriptions)}")
    
    def process_audio_chunks(self):
        """Process audio chunks from the queue."""
        audio_buffer = []
        chunk_samples = int(self.chunk_duration * self.RATE)
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_buffer.extend(audio_data)
                
                # Process when we have enough samples
                while len(audio_buffer) >= chunk_samples:
                    chunk = np.array(audio_buffer[:chunk_samples])
                    audio_buffer = audio_buffer[chunk_samples:]
                    
                    # Save chunk to temporary file
                    temp_path = f"temp_live_{int(time.time())}.wav"
                    import soundfile as sf
                    sf.write(temp_path, chunk, self.RATE)
                    
                    # Add to transcription queue
                    self.transcription_queue.put(temp_path)
                    
            except queue.Empty:
                continue
    
    def process_transcriptions(self):
        """Process transcription requests."""
        while self.is_recording or not self.transcription_queue.empty():
            try:
                temp_path = self.transcription_queue.get(timeout=0.1)
                
                # Transcribe chunk
                segments, info = self.whisper_model.transcribe(
                    temp_path,
                    word_timestamps=True,
                    language=None
                )
                
                # Print results and save to session
                for segment in segments:
                    if segment.text.strip():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] {segment.text.strip()}")
                        
                        # Save to session
                        if self.save_output:
                            transcription_entry = {
                                'timestamp': timestamp,
                                'text': segment.text.strip(),
                                'start': segment.start,
                                'end': segment.end,
                                'words': [
                                    {
                                        'word': word.word,
                                        'start': word.start,
                                        'end': word.end
                                    } for word in segment.words
                                ] if hasattr(segment, 'words') else []
                            }
                            self.session_transcriptions.append(transcription_entry)
                
                # Clean up temp file
                os.remove(temp_path)
                
            except queue.Empty:
                continue

def main():
    """Main function to run live transcription."""
    print("🎤 LIVE REAL-TIME TRANSCRIPTION")
    print("="*50)
    print("This will transcribe your microphone input in real-time.")
    print("Press Ctrl+C to stop recording.")
    print("Transcriptions will be saved to 'live_sessions' directory.")
    print("="*50)
    
    # Initialize live transcription with saving enabled
    live_transcriber = LiveTranscription(
        whisper_model="tiny",      # Use tiny model for fastest processing
        chunk_duration=3,          # 3-second chunks for quick response
        save_output=True,          # Enable saving transcriptions
        output_dir="live_sessions" # Directory to save files
    )
    
    # Start live transcription
    live_transcriber.start_recording()

if __name__ == "__main__":
    main() 