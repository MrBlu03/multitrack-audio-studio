# Multitrack Audio Studio & Transcriber

An automated, studio-grade multitrack audio restoration, bleed gating, voice leveling, and speech transcription suite designed for multi-microphone tabletop, podcast, and video productions (Star Wars 5e / Cyberpunk RED / actual plays).

![Interface](web/preview.png) *(Modern hardware-accelerated DAW workstation interface)*

---

## Features

### 1. Fully Automated Acoustic Bleed Removal & Self-Tuning
- **Dynamic Cross-Track Bleed Matrix**: Uses `BleedCalibrator` to analyze inter-track acoustic crosstalk, VoIP network latency windows, and ambient room noise floors from the provided stems.
- **Zero Knob Twiddling**: Gating thresholds, attack/hold/release envelopes, and cancellation matrices self-calibrate based on the session audio.
- **Enforced Studio DSP Chain**:
  $$\text{Restorative EQ} \longrightarrow \text{Soft Tanh Limiter} \longrightarrow \text{Dialogue Silence Gate} \longrightarrow \text{RNNoise Neural Denoise} \longrightarrow \text{-18.0 LUFS Loudness Match}$$

### 2. Tailored Acoustic Restoration Profiles
- **Timmy / Magnus (`t6_laptop_fan`)**: Multi-window median noise profiling, dynamic spectral expander, desk reflection filter, warmth lift (+2.5 dB), and presence polish.
- **Robin (`t1_room_echo`)**: Room reverberation suppressor, 350ms soft tail compressor, and 500ms dialogue hold envelope.
- **Mathew / Salova (`t5_bleed_gate`)**: Multi-speaker cross-bleed suppression with VoIP jitter compensation.
- **Blu GM (`t3_reference`)**: Studio broadcast transparent levelling and bus compression.
- **Umbra / Rati (`t7_rati_clarity`)**: Dynamic presence shelf and high-frequency intelligibility boost.

### 3. Faster-Whisper Multitrack Transcription & Dialogue Merger
- Multi-track conversational speech-to-text with turn merging.
- Inter-word timestamp alignment and speaker label assignment.
- Exports to `.txt`, `.srt` subtitles, and `.json` timeline markers for DaVinci Resolve & Adobe Premiere Pro.

### 4. Hardware-Accelerated Desktop Workstation UI
- Built with **FastAPI**, **WebSockets**, and an embedded **WebView2 (Chromium)** native desktop window (`pywebview`) — identical in ergonomics to Electron/Tauri with zero browser tab popups.
- High-density DAW mixer rack with simulated 8-segment LED peak meters, channel mutes/solos, and real-time processing telemetry.

---

## Quick Start

### Option A: Standalone Executable (Windows)
Run `Launch_Multitrack_Studio.bat` or open `MultitrackAudioStudio.exe`.

### Option B: Run from Source
1. Clone the repository:
   ```bash
   git clone git@github.com:MrBlu03/multitrack-audio-studio.git
   cd multitrack-audio-studio
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the studio:
   ```bash
   python gui.py
   ```

---

## Build Standalone Executable
To package into a single standalone `.exe` using PyInstaller:
```bash
pyinstaller --noconfirm MultitrackAudioStudio.spec
```

---

## License
MIT License. Created for actual-play tabletop podcast audio engineering.
