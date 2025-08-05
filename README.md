# Whisper Speaker Diarization - First Release

A local implementation of speaker diarization using Whisper for transcription, inspired by the [Xenova/whisper-speaker-diarization](https://huggingface.co/spaces/Xenova/whisper-speaker-diarization) Hugging Face space.

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the system**:
   ```bash
   python diarization.py
   ```

3. **Check results**:
   - Look in the `results/` folder for output files
   - Run `python demo.py` to see what's available

## ✨ Features

- **Whisper Transcription**: High-quality speech-to-text using faster-whisper
- **Basic Speaker Separation**: Simple speaker assignment based on segments
- **Timestamp Alignment**: Word-level timestamps for precise timing
- **Visualization**: Generate timeline plots of speaker segments
- **JSON Export**: Save results in structured JSON format
- **Multi-format Support**: Supports WAV, MP3, FLAC, M4A audio files
- **Multi-language Support**: English, Hebrew, and auto-detection

## 📁 Project Structure

```
speaker_diarization_release/
├── diarization.py  # Main script (ready to use)
├── demo.py                        # System overview and demo
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── data/                         # Audio files directory
│   ├── EN - Emily and I 2.wav
│   ├── HE - Emily and I.wav
│   └── Hebrew.wav
└── results/                      # Output directory
    ├── transcription_results.json
    └── transcription_plot.png
```

## 🎯 What's Included

### Working Features:
- ✅ **Transcription with timestamps**
- ✅ **Basic speaker separation**
- ✅ **Visualization**
- ✅ **JSON export**
- ✅ **Multi-language support**

### Sample Output:
- **JSON Results**: Detailed transcription with speaker segments
- **Timeline Plot**: Visual representation of speaker segments
- **Console Output**: Real-time processing information

## 🔧 Usage

### Basic Usage:
```bash
python diarization.py
```

### Demo:
```bash
python demo.py
```

### Programmatic Usage:
```python
from diarization import SimpleSpeakerDiarizationNoAuth

# Initialize
diarizer = SimpleSpeakerDiarizationNoAuth()

# Process audio
results = diarizer.process_audio("path/to/audio.wav", "output_dir")
```

## 📊 Output Format

### JSON Structure:
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
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "how", "start": 0.6, "end": 0.8}
      ]
    }
  ]
}
```

## 🎵 Supported Audio

- **Formats**: WAV, MP3, FLAC, M4A
- **Languages**: English, Hebrew, and auto-detection
- **Quality**: Works with various audio qualities

## 🛠️ Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, for faster processing)

## 📈 Performance

- **CPU Mode**: Works on any machine
- **GPU Mode**: Automatically detected and used if available
- **Memory**: Efficient processing with streaming

## 🔮 Future Versions

This release includes the no-authentication version that works immediately. Future versions will include:

- Full speaker diarization with authentication
- Advanced speaker identification
- Batch processing capabilities
- Web interface

## 📝 License

This implementation is for educational and research purposes. Please respect the licenses of the underlying models:
- Whisper: MIT License
- faster-whisper: MIT License

## 🙏 Acknowledgments

- [Xenova](https://huggingface.co/spaces/Xenova/whisper-speaker-diarization) for the original Hugging Face space
- [OpenAI](https://openai.com/research/whisper) for the Whisper model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized Whisper inference

---

**Ready to use!** 🎤 Just run `python diarization.py` to get started. 