@echo off
title Multitrack Audio & Transcription Studio v4.0
cd /d "%~dp0"
echo ==============================================================
echo Launching Multitrack Audio & Transcription Studio...
echo ==============================================================
if exist MultitrackAudioStudio.exe (
    start "" MultitrackAudioStudio.exe
) else (
    python gui.py
)

