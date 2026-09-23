#!/usr/bin/env python3
"""
FastAPI Backend Server for Multitrack Audio & Transcription Studio
==================================================================
Bridges the modern hardware-accelerated Webview interface with:
- Automated Multi-Track Session DSP Mastering
- Faster-Whisper Speech-to-Text Transcription & Script Merging
- Native Windows File/Folder Browser Dialogs
- Real-Time 60fps WebSocket Progress & Console Streaming
"""

import os
import sys
import re
import time
import json
import asyncio
import threading
import subprocess
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Import core DSP engines and campaign configurations
from multitrack_bleed_gate import (
    get_campaign_config,
    profile_id_to_str,
    profile_str_to_id,
    auto_detect_session_tracks,
    process_automated_session,
    process_laptop_mic_file,
    process_vocal_restoration_file,
    PROFILE_CHOICES,
    CAMPAIGN_CONFIGS,
    _CUDA_AVAILABLE,
    _CUDA_DEVICE,
)

# Import video ingestion engine
from video_ingest import (
    probe_video_streams,
    extract_video_audio_stems,
    predict_video_stem_mapping,
    is_video_file,
    get_default_video_track_mapping,
    VIDEO_EXTENSIONS,
)


# Import transcriber
try:
    from multitrack_transcriber import (
        MultitrackTranscriber,
        export_txt,
        export_srt,
        export_json,
        export_moderation_report,
        export_timeline_markers_davinci,
        export_timeline_markers_premiere,
        setup_cuda_dll_paths,
    )
    setup_cuda_dll_paths()
    HAS_TRANSCRIBER = True
except Exception:
    HAS_TRANSCRIBER = False

app = FastAPI(title="Multitrack Audio Studio Backend")

# ---------------------------------------------------------------------------
# Global Session State
# ---------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.campaign_mode = "sw5e"
        self.config = get_campaign_config(self.campaign_mode)
        
        # 6 Auto-Master Slots
        self.auto_slots: Dict[int, Dict[str, Any]] = {}
        for s in range(1, 7):
            s_cfg = self.config["slots"].get(s, {})
            self.auto_slots[s] = {
                "slot": s,
                "label": s_cfg.get("label", f"Slot {s}:"),
                "player": s_cfg.get("player", f"Speaker {s}"),
                "character": s_cfg.get("character", ""),
                "speaker": s_cfg.get("speaker", f"Speaker {s}"),
                "path": "",
                "filename": "",
                "profile_id": s_cfg.get("profile_id", "t3_reference"),
                "profile_name": profile_id_to_str(s_cfg.get("profile_id", "t3_reference")),
                "active": s_cfg.get("active", True),
                "duration_sec": 0.0,
            }
            
        self.session_source_name = ""
        self.output_dir = ""
        self.export_format = "MP3 (320 kbps Broadcast)"
        self.apply_silence_gate = True
        self.apply_ai_denoise = True
        self.ai_denoise_strength = 1.0
        self.apply_normalization = True
        self.target_lufs = -18.0
        
        # Transcriber Settings
        self.whisper_model = "small"
        self.merge_gap = 1.2
        self.enable_moderation = True
        self.transcribe_prompt = self.config["prompt"]
        self.export_srt = True
        self.export_json = False
        self.auto_mute_flags = False
        
        # Video Ingestion State
        self.video_source_file = ""
        self.video_streams_info: List[Dict[str, Any]] = []
        self.is_extracting_video = False

        # Execution State
        self.is_processing = False
        self.is_cancelled = False
        self.last_output_file: Optional[str] = None
        self.progress_percent = 0.0
        self.status_message = "Ready. Drop a 4K video, session folder, or select audio tracks."
        self.active_clients: List[WebSocket] = []
        self._lock = threading.Lock()

state = SessionState()


# ---------------------------------------------------------------------------
# WebSocket Broadcast Helper
# ---------------------------------------------------------------------------
async def broadcast_ws(data: Dict[str, Any]):
    disconnected = []
    for client in state.active_clients:
        try:
            await client.send_json(data)
        except Exception:
            disconnected.append(client)
    for d in disconnected:
        if d in state.active_clients:
            state.active_clients.remove(d)

def ws_emit_sync(data: Dict[str, Any]):
    """Thread-safe synchronous emitter for background worker callbacks."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_ws(data), loop)
    except Exception:
        pass


def log_broadcast(msg: str):
    """Broadcast log line to WebSocket clients and stdout."""
    print(msg)
    ws_emit_sync({"type": "log", "message": msg})


# ---------------------------------------------------------------------------
# REST API Models & Endpoints
# ---------------------------------------------------------------------------
class CampaignSwitchReq(BaseModel):
    mode: str

class SlotUpdateReq(BaseModel):
    slot: int
    path: Optional[str] = None
    profile_id: Optional[str] = None
    active: Optional[bool] = None

class SettingsUpdateReq(BaseModel):
    output_dir: Optional[str] = None
    export_format: Optional[str] = None
    apply_silence_gate: Optional[bool] = None
    apply_ai_denoise: Optional[bool] = None
    ai_denoise_strength: Optional[float] = None
    apply_normalization: Optional[bool] = None
    target_lufs: Optional[float] = None
    whisper_model: Optional[str] = None
    merge_gap: Optional[float] = None
    enable_moderation: Optional[bool] = None
    transcribe_prompt: Optional[str] = None
    export_srt: Optional[bool] = None
    export_json: Optional[bool] = None

class StartMasterReq(BaseModel):
    preview_sec: Optional[float] = None

class VideoIngestReq(BaseModel):
    video_path: str
    custom_mapping: Optional[Dict[int, int]] = None

class VideoProbeReq(BaseModel):
    video_path: str



@app.get("/api/state")
def get_state():
    return {
        "campaign_mode": state.campaign_mode,
        "campaign_name": state.config["name"],
        "campaign_icon": state.config["icon"],
        "session_source_name": state.session_source_name,
        "slots": state.auto_slots,
        "profile_choices": [{"id": profile_str_to_id(c), "name": c} for c in PROFILE_CHOICES],
        "output_dir": state.output_dir,

        "export_format": state.export_format,
        "apply_silence_gate": state.apply_silence_gate,
        "apply_ai_denoise": state.apply_ai_denoise,
        "ai_denoise_strength": state.ai_denoise_strength,
        "apply_normalization": state.apply_normalization,
        "target_lufs": state.target_lufs,
        "whisper_model": state.whisper_model,
        "merge_gap": state.merge_gap,
        "enable_moderation": state.enable_moderation,
        "transcribe_prompt": state.transcribe_prompt,
        "export_srt": state.export_srt,
        "export_json": state.export_json,
        "is_processing": state.is_processing,
        "progress_percent": state.progress_percent,
        "status_message": state.status_message,
        "last_output_file": state.last_output_file,
        "video_source_file": state.video_source_file,
        "video_streams_info": state.video_streams_info,
        "is_extracting_video": state.is_extracting_video,
        "cuda_available": _CUDA_AVAILABLE,
        "cuda_device": _CUDA_DEVICE,
        "has_transcriber": HAS_TRANSCRIBER,
    }


@app.post("/api/campaign")
def set_campaign(req: CampaignSwitchReq):
    mode = req.mode.lower()
    if mode not in CAMPAIGN_CONFIGS:
        return JSONResponse({"error": "Invalid campaign mode"}, status_code=400)
    
    state.campaign_mode = mode
    state.config = get_campaign_config(mode)
    state.transcribe_prompt = state.config["prompt"]
    
    for s in range(1, 7):
        s_cfg = state.config["slots"].get(s, {})
        state.auto_slots[s]["label"] = s_cfg.get("label", f"Slot {s}:")
        state.auto_slots[s]["player"] = s_cfg.get("player", f"Speaker {s}")
        state.auto_slots[s]["character"] = s_cfg.get("character", "")
        state.auto_slots[s]["speaker"] = s_cfg.get("speaker", f"Speaker {s}")
        def_prof = s_cfg.get("profile_id", "t3_reference")
        state.auto_slots[s]["profile_id"] = def_prof
        state.auto_slots[s]["profile_name"] = profile_id_to_str(def_prof)
        state.auto_slots[s]["active"] = s_cfg.get("active", True)
            
    log_broadcast(f"[⚡] Switched Campaign Mode: {state.config['icon']} {state.config['name']}")
    return get_state()


@app.post("/api/update-slot")
def update_slot(req: SlotUpdateReq):
    if req.slot not in state.auto_slots:
        return JSONResponse({"error": "Invalid slot"}, status_code=400)
    slot_info = state.auto_slots[req.slot]
    if req.path is not None:
        slot_info["path"] = req.path
        slot_info["filename"] = os.path.basename(req.path) if req.path else ""
    if req.profile_id is not None:
        slot_info["profile_id"] = req.profile_id
        slot_info["profile_name"] = profile_id_to_str(req.profile_id)
    if req.active is not None:
        slot_info["active"] = req.active
    return slot_info


@app.post("/api/update-settings")
def update_settings(req: SettingsUpdateReq):
    if req.output_dir is not None: state.output_dir = req.output_dir
    if req.export_format is not None: state.export_format = req.export_format
    if req.apply_silence_gate is not None: state.apply_silence_gate = req.apply_silence_gate
    if req.apply_ai_denoise is not None: state.apply_ai_denoise = req.apply_ai_denoise
    if req.ai_denoise_strength is not None: state.ai_denoise_strength = req.ai_denoise_strength
    if req.apply_normalization is not None: state.apply_normalization = req.apply_normalization
    if req.target_lufs is not None: state.target_lufs = req.target_lufs
    if req.whisper_model is not None: state.whisper_model = req.whisper_model
    if req.merge_gap is not None: state.merge_gap = req.merge_gap
    if req.enable_moderation is not None: state.enable_moderation = req.enable_moderation
    if req.transcribe_prompt is not None: state.transcribe_prompt = req.transcribe_prompt
    if req.export_srt is not None: state.export_srt = req.export_srt
    if req.export_json is not None: state.export_json = req.export_json
    return {"status": "ok"}


def _apply_detected_tracks(tracks: Dict[int, str], session_name: str, folder_path: Optional[str] = None):
    state.session_source_name = session_name
    detected_count = 0
    first_path = None
    
    for s in range(1, 7):
        p = tracks.get(s, "")
        if p and os.path.isfile(p):
            state.auto_slots[s]["path"] = p
            state.auto_slots[s]["filename"] = os.path.basename(p)
            state.auto_slots[s]["active"] = True
            detected_count += 1
            if not first_path:
                first_path = p
        else:
            state.auto_slots[s]["path"] = ""
            state.auto_slots[s]["filename"] = ""
            
    if first_path:
        base_dir = folder_path if folder_path else os.path.dirname(first_path)
        state.output_dir = os.path.join(base_dir, "Mastered")
        
    log_broadcast(f"[+] Loaded Session '{session_name}' ({detected_count} track(s) mapped)")
    ws_emit_sync({"type": "session_loaded", "name": session_name, "count": detected_count, "state": get_state()})


def _apply_predicted_video_tracks(prediction: Dict[str, Any], video_path: str):
    """Instantly populate rack slot cards and telemetry BEFORE background ffmpeg extraction finishes."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    session_name = f"{base_name} (4K Stems)"
    state.session_source_name = session_name
    state.video_source_file = video_path
    state.is_extracting_video = True
    state.progress_percent = 0.0
    state.status_message = f"Extracting 4K audio stems from {os.path.basename(video_path)}..."
    output_dir = prediction.get("output_dir", "")
    if output_dir:
        state.output_dir = os.path.join(output_dir, "Mastered")

    slots_data = prediction.get("slots", {})
    for s in range(1, 7):
        if s in slots_data:
            s_info = slots_data[s]
            state.auto_slots[s]["path"] = s_info["path"]
            state.auto_slots[s]["filename"] = s_info["filename"]
            state.auto_slots[s]["player"] = s_info.get("player", f"Speaker {s}")
            state.auto_slots[s]["character"] = s_info.get("character", "")
            state.auto_slots[s]["profile_id"] = s_info.get("profile_id", "t3_reference")
            state.auto_slots[s]["profile_name"] = profile_id_to_str(s_info.get("profile_id", "t3_reference"))
            state.auto_slots[s]["active"] = True
        else:
            state.auto_slots[s]["path"] = ""
            state.auto_slots[s]["filename"] = ""
            state.auto_slots[s]["active"] = False

    log_broadcast(f"[+] Instant-Mapped {len(slots_data)} 4K tracks for '{session_name}'. Extraction in progress...")
    ws_emit_sync({"type": "session_loaded", "name": session_name, "count": len(slots_data), "state": get_state()})


def _process_video_ingestion(video_path: str, custom_mapping: Optional[Dict[int, int]] = None):
    """Background worker for extracting audio tracks from a 4K video container."""
    if not is_video_file(video_path):
        log_broadcast(f"[-] Invalid or non-existent video file: {video_path}")
        return False

    state.is_extracting_video = True
    state.video_source_file = video_path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    session_name = f"{base_name} (4K Stems)"
    
    probe = probe_video_streams(video_path)
    state.video_streams_info = probe.get("audio_streams", [])
    num_tracks = probe.get("num_audio_streams", 0)
    
    log_broadcast(f"\n======================================================================")
    log_broadcast(f"🎬 DIRECT 4K VIDEO INGESTION: {os.path.basename(video_path)}")
    log_broadcast(f"Duration: {probe.get('duration_sec', 0):.1f}s | Audio Streams: {num_tracks}")
    log_broadcast(f"Mode: {'Cyberpunk RED' if state.campaign_mode == 'red' else 'Star Wars 5e'}")
    log_broadcast(f"======================================================================\n")

    parent_dir = os.path.dirname(os.path.abspath(video_path))
    stems_dir = os.path.join(parent_dir, f"{base_name}_Stems")

    def _v_prog(p):
        pct = p.get("percent", 0.0)
        state.progress_percent = pct
        state.status_message = p.get("status", "")
        ws_emit_sync({
            "type": "video_progress",
            "percent": pct,
            "speed": p.get("speed", 1.0),
            "status": state.status_message
        })

    def _v_cancel():
        return state.is_cancelled

    try:
        extracted = extract_video_audio_stems(
            video_path=video_path,
            output_dir=stems_dir,
            campaign_mode=state.campaign_mode,
            custom_track_mapping=custom_mapping,
            progress_callback=_v_prog,
            cancel_check=_v_cancel,
            log_func=log_broadcast
        )
        _apply_detected_tracks(extracted, session_name=session_name, folder_path=stems_dir)
        state.status_message = f"4K audio extraction complete ({len(extracted)} stems ready)."
        ws_emit_sync({
            "type": "video_extracted",
            "success": True,
            "video_path": video_path,
            "session_name": session_name,
            "count": len(extracted),
            "state": get_state()
        })
        return True
    except Exception as e:
        log_broadcast(f"[-] Video extraction failed: {e}")
        state.status_message = f"Video extraction failed: {e}"
        ws_emit_sync({"type": "video_extracted", "success": False, "error": str(e), "state": get_state()})
        return False
    finally:
        state.is_extracting_video = False


@app.post("/api/browse-folder")
def browse_folder():
    """Trigger native Windows folder dialog in a separate hidden Tk root."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Multitrack Session Folder")
        root.destroy()
        if folder and os.path.isdir(folder):
            session_name = os.path.basename(folder)
            tracks = auto_detect_session_tracks(folder, mode=state.campaign_mode)
            _apply_detected_tracks(tracks, session_name, folder)
            return get_state()
    except Exception as e:
        log_broadcast(f"[-] Browse folder error: {e}")
    return get_state()


@app.post("/api/browse-files")
def browse_files():
    """Trigger native Windows multi-file open dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Select Multitrack Session Audio Files or Video",
            filetypes=[
                ("Audio & Video Files", "*.mp3 *.wav *.flac *.aiff *.mkv *.mp4 *.mov *.webm *.avi *.m4v"),
                ("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff"),
                ("Video Containers (*.mkv, *.mp4, *.mov)", "*.mkv *.mp4 *.mov *.webm *.avi *.m4v"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        if files:
            files_list = list(files)
            if len(files_list) == 1 and is_video_file(files_list[0]):
                video_file = files_list[0]
                prediction = predict_video_stem_mapping(video_file, campaign_mode=state.campaign_mode)
                if prediction.get("success"):
                    _apply_predicted_video_tracks(prediction, video_file)
                threading.Thread(target=_process_video_ingestion, args=(video_file,), daemon=True).start()
                return get_state()
            session_name = os.path.basename(os.path.dirname(files_list[0]))
            tracks = auto_detect_session_tracks(files_list, mode=state.campaign_mode)
            _apply_detected_tracks(tracks, session_name)
            return get_state()
    except Exception as e:
        log_broadcast(f"[-] Browse files error: {e}")
    return get_state()


@app.post("/api/browse-video")
def browse_video():
    """Trigger native Windows file open dialog specifically for 4K video containers."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        video_file = filedialog.askopenfilename(
            title="Select 4K Video Container (.mkv, .mp4, .mov)",
            filetypes=[
                ("Video Containers (*.mkv, *.mp4, *.mov, *.webm, *.avi)", "*.mkv *.mp4 *.mov *.webm *.avi *.m4v"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        if video_file and is_video_file(video_file):
            prediction = predict_video_stem_mapping(video_file, campaign_mode=state.campaign_mode)
            if prediction.get("success"):
                _apply_predicted_video_tracks(prediction, video_file)
            threading.Thread(target=_process_video_ingestion, args=(video_file,), daemon=True).start()
            return get_state()
    except Exception as e:
        log_broadcast(f"[-] Browse video error: {e}")
    return get_state()


@app.post("/api/ingest-video")
def ingest_video_endpoint(req: VideoIngestReq):
    """Directly ingest a video file given its path (e.g. from drag & drop)."""
    if not is_video_file(req.video_path):
        return JSONResponse({"error": f"Invalid video file path: {req.video_path}"}, status_code=400)
    prediction = predict_video_stem_mapping(req.video_path, campaign_mode=state.campaign_mode, custom_track_mapping=req.custom_mapping)
    if prediction.get("success"):
        _apply_predicted_video_tracks(prediction, req.video_path)
    threading.Thread(target=_process_video_ingestion, args=(req.video_path, req.custom_mapping), daemon=True).start()
    return {"status": "started", "video_path": req.video_path, "state": get_state()}



@app.post("/api/probe-video")
def probe_video_endpoint(req: VideoProbeReq):
    """Inspect video audio streams in milliseconds."""
    probe = probe_video_streams(req.video_path)
    return probe


@app.post("/api/browse-single-file/{slot}")
def browse_single_file(slot: int):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        f = filedialog.askopenfilename(
            title=f"Select Audio Track for Slot {slot}",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff"), ("All Files", "*.*")]
        )
        root.destroy()
        if f and os.path.isfile(f):
            state.auto_slots[slot]["path"] = f
            state.auto_slots[slot]["filename"] = os.path.basename(f)
            state.auto_slots[slot]["active"] = True
            log_broadcast(f"[+] Slot {slot} set to: {os.path.basename(f)}")
            return get_state()
    except Exception as e:
        log_broadcast(f"[-] Browse file error: {e}")
    return get_state()


@app.post("/api/browse-output-dir")
def browse_output_dir():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Mastered Output Directory")
        root.destroy()
        if folder:
            state.output_dir = folder
            return {"output_dir": folder}
    except Exception as e:
        log_broadcast(f"[-] Browse output dir error: {e}")
    return {"output_dir": state.output_dir}


@app.post("/api/clear-slots")
def clear_slots():
    for s in range(1, 7):
        state.auto_slots[s]["path"] = ""
        state.auto_slots[s]["filename"] = ""
    state.session_source_name = ""
    log_broadcast("[*] Cleared all session slots.")
    return get_state()


@app.post("/api/open-output")
def open_output():
    target = state.last_output_file or state.output_dir
    if target and os.path.exists(target):
        try:
            if os.path.isfile(target):
                subprocess.run(["explorer", f"/select,{os.path.abspath(target)}"])
            else:
                os.startfile(os.path.abspath(target))
            return {"status": "ok"}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"error": "Path does not exist"}, status_code=404)


@app.post("/api/cancel")
def cancel_processing():
    if state.is_processing:
        state.is_cancelled = True
        log_broadcast("[-] Cancellation requested by user...")
        return {"status": "cancelling"}
    return {"status": "idle"}


# ---------------------------------------------------------------------------
# Background DSP Execution
# ---------------------------------------------------------------------------
@app.post("/api/start-master")
def start_master(req: StartMasterReq):
    if state.is_processing:
        return JSONResponse({"error": "Processing already in progress"}, status_code=400)
        
    active_slots = {}
    for s, info in state.auto_slots.items():
        if info["active"] and info["path"] and os.path.isfile(info["path"]):
            active_slots[s] = {
                "path": info["path"],
                "profile": info["profile_id"],
                "player": info.get("player", ""),
                "character": info.get("character", ""),
            }
            
    if not active_slots:
        return JSONResponse({"error": "No valid active audio tracks loaded in slots"}, status_code=400)
        
    out_dir = state.output_dir
    if not out_dir:
        first_p = next(iter(active_slots.values()))["path"]
        out_dir = os.path.join(os.path.dirname(first_p), "Mastered")
        state.output_dir = out_dir
        
    os.makedirs(out_dir, exist_ok=True)
    state.is_processing = True
    state.is_cancelled = False
    state.progress_percent = 0.0
    preview_sec = req.preview_sec
    
    def _worker():
        t0 = time.time()
        mode_desc = f"{preview_sec:.0f}s Preview" if (preview_sec and preview_sec > 0) else "Full Session"
        log_broadcast(f"\n======================================================================")
        log_broadcast(f"🚀 STARTING AUTOMATED SESSION MASTERING: {len(active_slots)} TRACK(S) [{mode_desc}]")
        log_broadcast(f"Output Directory: {out_dir}")
        log_broadcast(f"Format: {state.export_format} | Gate: {state.apply_silence_gate} | AI: {state.apply_ai_denoise} ({int(state.ai_denoise_strength*100)}%) | LUFS: {state.target_lufs}")
        log_broadcast(f"======================================================================\n")
        
        def _prog(track_idx, num_tracks, out_fn, track_pct, total_pct, speed, eta_str, status_msg):
            state.progress_percent = round(total_pct, 1)
            state.status_message = f"[{track_idx}/{num_tracks}] {out_fn} ({track_pct:.1f}%) | Speed: {speed:.1f}x | ETA: {eta_str}"
            p_payload = {
                "track_idx": track_idx,
                "num_tracks": num_tracks,
                "total_tracks": num_tracks,
                "track_name": out_fn,
                "track_percent": round(track_pct, 1),
                "total_percent": round(total_pct, 1),
                "pct": round(track_pct, 1),
                "total_pct": round(total_pct, 1),
                "speed": round(speed, 1),
                "eta": eta_str,
                "status": state.status_message,
            }
            ws_emit_sync({
                "type": "progress",
                **p_payload,
                "data": p_payload,
            })
            
        def _cancel():
            return state.is_cancelled
            
        try:
            results = process_automated_session(
                active_slots=active_slots,
                output_dir=out_dir,
                preview_sec=preview_sec,
                export_format="mp3" if "mp3" in state.export_format.lower() else "wav",
                apply_silence_gate=state.apply_silence_gate,
                apply_ai_denoise=state.apply_ai_denoise,
                ai_denoise_strength=state.ai_denoise_strength,
                apply_speech_normalization=state.apply_normalization,
                target_lufs=state.target_lufs,
                progress_callback=_prog,
                cancel_check=_cancel,
                log_func=log_broadcast,
            )
            
            elapsed = time.time() - t0
            if state.is_cancelled:
                log_broadcast("\n[-] Session mastering cancelled.")
                ws_emit_sync({"type": "finish", "success": False, "message": "Cancelled by user."})
            else:
                log_broadcast(f"\n🎉 MASTERING COMPLETE! All tracks exported in {elapsed:.1f}s")
                state.last_output_file = out_dir
                ws_emit_sync({"type": "finish", "success": True, "message": f"Mastering complete in {elapsed:.1f}s", "output": out_dir})
        except Exception as e:
            log_broadcast(f"\n[-] Error during session mastering: {e}")
            ws_emit_sync({"type": "finish", "success": False, "message": str(e)})
        finally:
            state.is_processing = False
            state.is_cancelled = False
            
    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "preview": preview_sec}


@app.post("/api/start-transcribe")
def start_transcribe(req: StartMasterReq):
    if not HAS_TRANSCRIBER:
        return JSONResponse({"error": "Transcriber not available (faster-whisper missing)"}, status_code=400)
    if state.is_processing:
        return JSONResponse({"error": "Processing already in progress"}, status_code=400)
        
    tracks_config = []
    for s, info in state.auto_slots.items():
        if info["active"] and info["path"] and os.path.isfile(info["path"]):
            spk = info.get("speaker") or info.get("player") or f"Speaker {s}"
            tracks_config.append({
                "path": info["path"],
                "speaker": spk,
                "active": True,
            })
            
    if not tracks_config:
        return JSONResponse({"error": "No valid active audio tracks for transcription"}, status_code=400)
        
    out_dir = state.output_dir or os.path.join(os.path.dirname(tracks_config[0]["path"]), "Transcripts")
    os.makedirs(out_dir, exist_ok=True)
    state.is_processing = True
    state.is_cancelled = False
    state.progress_percent = 0.0
    
    def _worker():
        t0 = time.time()
        log_broadcast(f"\n======================================================================")
        log_broadcast(f"📝 STARTING MULTITRACK TRANSCRIPTION: {len(tracks_config)} SPEAKER(S)")
        log_broadcast(f"Model: {state.whisper_model} | Turn Gap: {state.merge_gap:.1f}s | Moderation: {state.enable_moderation}")
        log_broadcast(f"======================================================================\n")
        
        def _prog(p_data):
            if isinstance(p_data, dict):
                pct = p_data.get("progress_pct", 0.0)
                state.progress_percent = round(pct, 1)
                ws_emit_sync({"type": "transcribe_progress", "data": p_data})
                
        def _cancel():
            return state.is_cancelled
            
        try:
            transcriber = MultitrackTranscriber(
                model_size=state.whisper_model,
                device="cuda" if _CUDA_AVAILABLE else "cpu",
                compute_type="float16" if _CUDA_AVAILABLE else "int8",
            )
            
            res = transcriber.transcribe_multitrack_session(
                tracks_config=tracks_config,
                initial_prompt=state.transcribe_prompt,
                pause_merge_sec=state.merge_gap,
                enable_moderation=state.enable_moderation,
                progress_callback=_prog,
                status_callback=log_broadcast,
                cancel_check=_cancel,
            )
            
            session_name = state.session_source_name or "Session"
            txt_path = os.path.join(out_dir, f"{session_name}_Transcript.txt")
            export_txt(res, txt_path, title=f"{session_name} Multitrack Transcript")
            log_broadcast(f"[+] Clean script exported: {txt_path}")
            
            if state.export_srt:
                srt_path = os.path.join(out_dir, f"{session_name}_Subtitles.srt")
                export_srt(res, srt_path)
                log_broadcast(f"[+] SubRip subtitles exported: {srt_path}")
                
            if state.export_json:
                json_path = os.path.join(out_dir, f"{session_name}_Timeline.json")
                export_json(res, json_path)
                log_broadcast(f"[+] Timeline JSON exported: {json_path}")
                
            elapsed = time.time() - t0
            state.last_output_file = txt_path
            log_broadcast(f"\n🎉 TRANSCRIPTION COMPLETE! Processed {len(tracks_config)} tracks in {elapsed:.1f}s")
            ws_emit_sync({"type": "finish", "success": True, "message": f"Transcription complete in {elapsed:.1f}s", "output": txt_path})
        except Exception as e:
            log_broadcast(f"\n[-] Error during transcription: {e}")
            ws_emit_sync({"type": "finish", "success": False, "message": str(e)})
        finally:
            state.is_processing = False
            state.is_cancelled = False
            
    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    state.active_clients.append(websocket)
    try:
        await websocket.send_json({
            "type": "init",
            "state": get_state(),
        })
        while True:
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        if websocket in state.active_clients:
            state.active_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Static Web Assets Mounting
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    bundle_base = sys._MEIPASS
else:
    bundle_base = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(bundle_base, "web")
if not os.path.isdir(web_dir):
    os.makedirs(web_dir, exist_ok=True)

@app.get("/")
def serve_index():
    index_file = os.path.join(web_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "Multitrack Studio Server Running", "web_dir": web_dir})

app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")


def run_server(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
