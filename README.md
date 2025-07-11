# Voice Cloning Model

Clone any voice from a short audio clip and synthesize speech from custom text 

---

## Features
- **Voice embedding cloning** via Resemblyzer
- **End-to-end speech generation** using a fine-tuned VITS model
- **Streamlit web interface** for real-time voice synthesis
- **Fallback TTS** using pyttsx3 if no audio is provided

---

##  How It Works
### 1. Preprocessing
- **Input audio:**
  - Converted to mono
  - Resampled to 16kHz
  - Normalized and passed to Resemblyzer to extract a 256-dimension voice embedding
- **Input text:**
  - Cleaned using custom text cleaners
  - Tokenized and optionally interleaved with blank tokens

### 2. Model Training
- Uses the VITS (Variational Inference TTS) architecture
- Trained on a custom dataset with (text, audio_path, speaker_id)
- Optimized to predict mel-spectrograms from text and speaker features
- Output waveform is generated from learned spectrograms

### 3. Synthesis
- **Input:** text + (reference audio or speaker ID)
- Model generates a waveform aligned with the input voice and text
- Saved as `cloned_voice.wav` for download or playback

---

##  Tech Stack
| Component      | Tech Used                |
|---------------|--------------------------|
| Frontend      | Streamlit                |
| Backend       | PyTorch, Resemblyzer     |
| TTS Model     | VITS (fine-tuned)        |
| Fallback TTS  | pyttsx3                  |
| File Handling | librosa, soundfile       |

---

##  Folder Structure
```
voice-clone/
├── app.py               # Streamlit Web UI
├── synthesize.py        # Core synthesis logic
├── models/              # VITS model definition
├── configs/             # JSON config for model
├── utils.py             # Utility functions
├── text/                # Text cleaners and processing
├── data/                # Training data (optional)
└── fine_tuned.pth       # Pretrained or fine-tuned model
```

---

##  Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/your-username/voice-clone-app.git
cd voice-clone-app

# 2. Setup virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the web app
streamlit run app.py

#5. Run the synthesis directly without the web interface, use:
python synthesize.py
-
```

---

##  Usage
- Upload reference audio (`.wav`)
- Enter custom text for synthesis
- Choose voice embedding or speaker ID (Currently FALSE)
- Generate and play cloned voice
- Fallback to TTS if audio is missing

---

##  Hackathon Impact
This project demonstrates how AI voice synthesis can be personalized in real-time. Potential applications include:
- Voice assistants in regional accents
- Accessibility tools for speech impairment
- AI-generated content in any voice

---


