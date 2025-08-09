# Whisper Speaker Diarization

A local implementation of speaker diarization using Whisper for transcription and pyannote.audio for speaker identification, inspired by the [Xenova/whisper-speaker-diarization](https://huggingface.co/spaces/Xenova/whisper-speaker-diarization) Hugging Face space.

## Features

- **Whisper Transcription**: High-quality speech-to-text using faster-whisper
- **Speaker Diarization**: Identify and separate different speakers in audio
- **Timestamp Alignment**: Align transcription segments with speaker segments
- **Visualization**: Generate timeline plots of speaker segments
- **JSON Export**: Save results in structured JSON format
- **Multi-format Support**: Supports WAV, MP3, FLAC, M4A audio files

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (if not already installed):
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) or use Chocolatey: `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

4. **Optional: Set up Hugging Face token** (for private models):
   ```bash
   # Create a .env file or set environment variable
   export HF_="your_huggingface_token_here"
   ```

## Usage

### Quick Start

1. **Place your audio files** in the `data/` directory
2. **Run the diarization script**:
   ```bash
   python simple_diarization.py
   ```

### Programmatic Usage

```python
from simple_diarization import SimpleSpeakerDiarization

# Initialize the system
diarizer = SimpleSpeakerDiarization()

# Process a single audio file
results = diarizer.process_audio("path/to/your/audio.wav", "output_directory")

# Print results
for segment in results:
    print(f"Speaker {segment['speaker']} ({segment['start']:.2f}s - {segment['end']:.2f}s): {segment['text']}")
```

### Configuration Options

You can customize the models used:

```python
# Use different Whisper model sizes
diarizer = SimpleSpeakerDiarization(whisper_model="base")  # tiny, base, small, medium, large, large-v3

# Use different diarization model
diarizer = SimpleSpeakerDiarization(diarization_model="pyannote/speaker-diarization-3.1")
```

## Output

The system generates several output files in the `results/` directory:

- **`diarization_plot.png`**: Timeline visualization of speaker segments
- **`diarization_results.json`**: Structured JSON with aligned transcription and speaker information

### JSON Output Format

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, how are you today?",
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5
        },
        {
          "word": "how",
          "start": 0.6,
          "end": 0.8
        }
      ]
    }
  ]
}
```

## Performance Tips

1. **Use GPU**: The system automatically detects and uses CUDA if available
2. **Model Size**: Smaller Whisper models (tiny, base) are faster but less accurate
3. **Audio Quality**: Higher quality audio files produce better results
4. **File Size**: Very large files may take longer to process

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use smaller Whisper model or process shorter audio files
2. **FFmpeg not found**: Ensure FFmpeg is installed and in your system PATH
3. **Model download errors**: Check your internet connection and Hugging Face token
4. **Audio format issues**: Convert audio to WAV format if problems persist

### Error Messages

- **"No module named 'faster_whisper'"**: Run `pip install -r requirements.txt`
- **"FFmpeg not found"**: Install FFmpeg and ensure it's in your PATH
- **"CUDA out of memory"**: Use CPU mode or smaller models

## Advanced Usage

### Custom Audio Processing

```python
# Process specific audio file
audio_path = "custom_audio.wav"
output_dir = "custom_results"

diarizer = SimpleSpeakerDiarization()
results = diarizer.process_audio(audio_path, output_dir)
```

### Batch Processing

```python
import os
from simple_diarization import SimpleSpeakerDiarization

diarizer = SimpleSpeakerDiarization()

# Process all audio files in a directory
audio_dir = "audio_files"
output_base = "results"

for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(('.wav', '.mp3', '.flac')):
        audio_path = os.path.join(audio_dir, audio_file)
        output_dir = os.path.join(output_base, audio_file.replace('.', '_'))
        results = diarizer.process_audio(audio_path, output_dir)
```

## Model Information

- **Whisper Models**: Based on OpenAI's Whisper architecture, optimized with faster-whisper
- **Speaker Diarization**: Uses pyannote.audio pipeline for speaker identification
- **Language Support**: Automatic language detection with Whisper

## License

This implementation is for educational and research purposes. Please respect the licenses of the underlying models:
- Whisper: MIT License
- pyannote.audio: MIT License

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

## Acknowledgments

- [Xenova](https://huggingface.co/spaces/Xenova/whisper-speaker-diarization) for the original Hugging Face space
- [OpenAI](https://openai.com/research/whisper) for the Whisper model
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized Whisper inference 