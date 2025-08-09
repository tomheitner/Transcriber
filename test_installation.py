#!/usr/bin/env python3
"""
Test script to verify the speaker diarization installation.
This script checks if all dependencies are properly installed.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import faster_whisper
        print(f"✓ faster-whisper {faster_whisper.__version__}")
    except ImportError as e:
        print(f"✗ faster-whisper: {e}")
        return False
    
    try:
        import pyannote.audio
        print(f"✓ pyannote.audio")
    except ImportError as e:
        print(f"✗ pyannote.audio: {e}")
        return False
    
    try:
        import librosa
        print(f"✓ librosa {librosa.__version__}")
    except ImportError as e:
        print(f"✗ librosa: {e}")
        return False
    
    try:
        import soundfile
        print(f"✓ soundfile {soundfile.__version__}")
    except ImportError as e:
        print(f"✗ soundfile: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA not available, will use CPU")
            return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_ffmpeg():
    """Test if FFmpeg is available."""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ FFmpeg available")
            return True
        else:
            print("✗ FFmpeg not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ FFmpeg not found in PATH")
        return False

def test_audio_files():
    """Test if audio files are available."""
    data_dir = "data"
    if os.path.exists(data_dir):
        audio_files = [f for f in os.listdir(data_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        if audio_files:
            print(f"✓ Found {len(audio_files)} audio files in data/")
            for file in audio_files:
                print(f"  - {file}")
            return True
        else:
            print("⚠ No audio files found in data/ directory")
            return True
    else:
        print("⚠ data/ directory not found")
        return True

def test_model_loading():
    """Test if models can be loaded (this will download them if needed)."""
    print("\nTesting model loading (this may take a few minutes on first run)...")
    
    try:
        from simple_diarization import SimpleSpeakerDiarization
        
        # Try to initialize with a smaller model for faster testing
        print("Loading Whisper model...")
        diarizer = SimpleSpeakerDiarization(whisper_model="base")
        print("✓ Models loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("SPEAKER DIARIZATION INSTALLATION TEST")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("CUDA Test", test_cuda),
        ("FFmpeg Test", test_ffmpeg),
        ("Audio Files Test", test_audio_files),
        ("Model Loading Test", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your installation is ready.")
        print("\nYou can now run:")
        print("  python simple_diarization.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install FFmpeg: https://ffmpeg.org/download.html")
        print("3. Check your internet connection for model downloads")

if __name__ == "__main__":
    main() 