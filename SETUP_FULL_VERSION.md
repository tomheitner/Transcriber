# Setup Guide for Full Speaker Diarization

This guide will help you set up the full speaker diarization system with authentication.

## Step 1: Get Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token (you'll need it later)

## Step 2: Accept Model Terms

1. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Accept" to accept the model terms
3. You may also need to accept terms for related models

## Step 3: Set Environment Variable

Set your Hugging Face token as an environment variable:

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"
```

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=your_token_here
```

**Linux/macOS:**
```bash
export HF_TOKEN="your_token_here"
```

## Step 4: Update the Code

Edit `simple_diarization.py` and change this line:

```python
# Change this line:
use_auth_token=None  # Set to your HF token if needed

# To this:
use_auth_token=os.getenv("HF_TOKEN")  # Use environment variable
```

## Step 5: Test the Full Version

Run the full version:
```bash
python simple_diarization.py
```

## Alternative: Direct Token Usage

If you prefer to hardcode the token (not recommended for security), you can modify the code directly:

```python
self.diarization_pipeline = Pipeline.from_pretrained(
    diarization_model,
    use_auth_token="your_token_here"  # Replace with your token
)
```

## Troubleshooting

### Common Issues:

1. **"Could not download pipeline"**: Make sure you've accepted the model terms
2. **"Authentication failed"**: Check your token is correct
3. **"Model not found"**: Ensure you're using the correct model name

### Model Alternatives:

If the default model doesn't work, you can try these alternatives:
- `pyannote/speaker-diarization-3.0`
- `pyannote/speaker-diarization-2.1`

## Performance Tips

1. **GPU Usage**: The system will automatically use CUDA if available
2. **Model Size**: Larger models are more accurate but slower
3. **Memory**: Speaker diarization requires significant RAM

## Expected Output

With the full version, you should get:
- Accurate speaker identification
- Proper speaker segmentation
- High-quality diarization plots
- Detailed JSON results with speaker labels

## Comparison

| Feature | No-Auth Version | Full Version |
|---------|----------------|--------------|
| Transcription | ✅ | ✅ |
| Basic Speaker Separation | ✅ | ✅ |
| Accurate Speaker Diarization | ❌ | ✅ |
| Hugging Face Token Required | ❌ | ✅ |
| Setup Complexity | Low | Medium |

The no-auth version provides transcription with basic speaker separation, while the full version offers accurate speaker diarization using advanced AI models. 