# PISA-Clap

PISA-Clap is an audio recording and classification project that utilizes the CLAP (Contrastive Language-Audio Pretraining) model to analyze audio. It features both a continuous background recording mode and a user-friendly web interface powered by Gradio.

## Features

- **Continuous Audio Analysis**: Run in a loop to continuously record audio, classify it, and clean up temporary audio files.
- **Web Interface**: Easy-to-use Gradio web app to upload or record audio clips for instant classification.
- **CLAP Integration**: Powered by Hugging Face Transformers for state-of-the-art audio-text understanding.

## Requirements

The project relies on several key libraries:
- `torch`
- `librosa`
- `transformers`
- `sounddevice`
- `soundfile`
- `openai`
- `litellm`
- `gradio` (required for UI)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supersam300/PISA-Clap.git
   cd PISA-Clap
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install all dependencies:
   ```bash
   pip install -r req.txt
   pip install gradio  # Make sure Gradio is installed for the web interface
   ```

## Usage

You can run the application in two different modes. 

### 1. Gradio Web Interface (Default)
To start the web interface where you can intuitively record your microphone or upload an audio file:
```bash
python main.py
```
This will launch a local web server (usually at `http://127.0.0.1:7860`).

### 2. Continuous Recording Loop
To run the script continuously in the terminal (records audio, processes it using the CLAP model, and deletes local `.wav` files after 10 seconds):
```bash
python main.py --loop
```

## Project Structure

- `main.py`: The entry point for the application. Handles mode selection (Gradio vs Loop).
- `audio_recs/`: Directory handling the storage and logic of audio recordings. Contains `record_audio.py`.
- `model/`: Directory containing the ML logic. Includes `clap.py` and data `augmentation.py`.
- `req.txt`: Required pip dependencies.

## License

[Add appropriate license info here]
