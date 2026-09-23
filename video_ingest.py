#!/usr/bin/env python3
"""
Video Ingestion and Multitrack Audio Extraction Engine
=====================================================
Direct 4K raw video container ingestion for OBS Studio / ATEM recordings (.mkv, .mp4, .mov).
- Inspects embedded audio streams (0:a:0 .. 0:a:5) in milliseconds via ffprobe without decoding video.
- Extracts all discrete tracks simultaneously in a single pass into pristine 24-bit PCM WAV stems (~100x speed).
- Automatically maps extracted stems to Channels 1-6 according to campaign presets (SW5E vs Cyberpunk RED).
"""

import os
import sys
import re
import json
import time
import subprocess
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable



VIDEO_EXTENSIONS = ('.mkv', '.mp4', '.mov', '.m4v', '.avi', '.webm')


def is_video_file(path: str) -> bool:
    """Check if file exists and has a supported video container extension."""
    if not path or not isinstance(path, str):
        return False
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTENSIONS)


def probe_video_streams(video_path: str) -> Dict[str, Any]:
    """
    Use ffprobe to inspect video and embedded audio streams in milliseconds.
    Does NOT decode the video stream.
    
    Returns dict:
        success: bool
        duration_sec: float
        format_name: str
        video_streams: list of dicts
        audio_streams: list of dicts (with audio_stream_idx 0, 1, 2...)
        is_multichannel_single_stream: bool
        error: Optional[str]
    """
    if not os.path.isfile(video_path):
        return {"success": False, "error": f"File not found: {video_path}"}

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,channels,channel_layout,sample_rate,duration,bit_rate:stream_tags=title,language",
        "-show_entries", "format=duration,size,format_name",
        "-of", "json",
        video_path
    ]

    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
            timeout=15.0
        )
        if proc.returncode != 0:
            return {"success": False, "error": proc.stderr.strip() or "ffprobe failed"}

        data = json.loads(proc.stdout)
    except Exception as e:
        return {"success": False, "error": f"ffprobe execution error: {e}"}

    streams = data.get("streams", [])
    format_info = data.get("format", {})
    
    duration_sec = 0.0
    try:
        duration_sec = float(format_info.get("duration", 0.0))
    except (ValueError, TypeError):
        pass

    video_streams = []
    audio_streams = []
    audio_idx = 0

    for s in streams:
        c_type = s.get("codec_type", "").lower()
        if c_type == "video":
            video_streams.append({
                "index": s.get("index"),
                "codec_name": s.get("codec_name"),
            })
        elif c_type == "audio":
            tags = s.get("tags", {}) or {}
            title = tags.get("title", tags.get("handler_name", ""))
            
            s_dur = duration_sec
            try:
                if "duration" in s:
                    s_dur = float(s["duration"])
            except (ValueError, TypeError):
                pass
            if s_dur > duration_sec:
                duration_sec = s_dur

            channels = int(s.get("channels", 2))
            sample_rate = int(s.get("sample_rate", 48000))

            audio_streams.append({
                "audio_stream_idx": audio_idx,
                "stream_index": s.get("index"),
                "codec_name": s.get("codec_name", "unknown"),
                "channels": channels,
                "channel_layout": s.get("channel_layout", "stereo" if channels == 2 else ("mono" if channels == 1 else f"{channels}ch")),
                "sample_rate": sample_rate,
                "duration_sec": s_dur,
                "title": title,
                "language": tags.get("language", "und")
            })
            audio_idx += 1

    is_multichannel_single = (len(audio_streams) == 1 and audio_streams[0]["channels"] >= 4)

    return {
        "success": True,
        "video_path": video_path,
        "filename": os.path.basename(video_path),
        "duration_sec": duration_sec,
        "format_name": format_info.get("format_name", ""),
        "video_streams": video_streams,
        "audio_streams": audio_streams,
        "num_audio_streams": len(audio_streams),
        "is_multichannel_single_stream": is_multichannel_single,
        "error": None
    }


def get_default_video_track_mapping(campaign_mode: str = "sw5e", num_streams: int = 8) -> Dict[int, Dict[str, Any]]:
    """
    Default mapping from video audio streams (0-based) to rack slot numbers (1-based),
    including player names and recommended acoustic profiles.
    
    Supports:
    - MeldStudio 8-Track Layout (num_streams >= 7):
        T1 (0:a:0): Master mixdown (ignored for voice slots)
        T2 (0:a:1): Music (exported as Track_Music.wav)
        T3..T8: Discrete player microphones
    - Standard OBS 6-Track Layout (num_streams <= 6):
        1-to-1 mapping to slots 1-6
    """
    mode = campaign_mode.lower() if campaign_mode else "sw5e"
    is_red = ("red" in mode or "cyber" in mode)

    # -------------------------------------------------------------------------
    # MeldStudio 8-Track Layout (Active Recording Setup)
    # -------------------------------------------------------------------------
    if num_streams >= 7:
        if is_red:
            return {
                0: {"slot": None, "player": "Mixdown", "character": "Full Mix", "profile_id": "skip", "active": False, "role": "mixdown"},
                1: {"slot": None, "player": "Music", "character": "BGM", "profile_id": "skip", "active": False, "role": "music"},
                2: {"slot": 1, "player": "Robin", "character": "QBall", "profile_id": "t1_room_echo", "active": True, "role": "voice"},
                3: {"slot": 2, "player": "Blu", "character": "GM", "profile_id": "t3_reference", "active": True, "role": "voice"},
                4: {"slot": 4, "player": "Rati", "character": "Umbra", "profile_id": "t7_rati_clarity", "active": True, "role": "voice"},
                5: {"slot": 3, "player": "Marc", "character": "Ryu", "profile_id": "t4_megaphone", "active": True, "role": "voice"},
                6: {"slot": 5, "player": "Timmy", "character": "Magnus", "profile_id": "t6_laptop_fan", "active": True, "role": "voice"},
                7: {"slot": 6, "player": "-", "character": "Empty", "profile_id": "skip", "active": False, "role": "empty"},
            }
        else:  # sw5e
            return {
                0: {"slot": None, "player": "Mixdown", "character": "Full Mix", "profile_id": "skip", "active": False, "role": "mixdown"},
                1: {"slot": None, "player": "Music", "character": "BGM", "profile_id": "skip", "active": False, "role": "music"},
                2: {"slot": 1, "player": "Robin", "character": "Cratebreaker", "profile_id": "t1_room_echo", "active": True, "role": "voice"},
                3: {"slot": 2, "player": "Tino", "character": "It'Mir", "profile_id": "t2_muffled", "active": True, "role": "voice"},
                4: {"slot": 3, "player": "Blu", "character": "Caelen", "profile_id": "t3_reference", "active": True, "role": "voice"},
                5: {"slot": 4, "player": "Marc", "character": "GM", "profile_id": "t4_megaphone", "active": True, "role": "voice"},
                6: {"slot": 5, "player": "Mathew", "character": "Salova", "profile_id": "t5_bleed_gate", "active": True, "role": "voice"},
                7: {"slot": 6, "player": "Timmy", "character": "Belial", "profile_id": "t6_laptop_fan", "active": True, "role": "voice"},
            }

    # -------------------------------------------------------------------------
    # Standard OBS 6-Track Layout (Fallback)
    # -------------------------------------------------------------------------
    if is_red:
        return {
            0: {"slot": 1, "player": "Robin", "character": "QBall", "profile_id": "t1_room_echo", "active": True, "role": "voice"},
            1: {"slot": 2, "player": "Blu", "character": "GM", "profile_id": "t3_reference", "active": True, "role": "voice"},
            2: {"slot": 3, "player": "Marc", "character": "Ryu", "profile_id": "t4_megaphone", "active": True, "role": "voice"},
            3: {"slot": 4, "player": "Rati", "character": "Umbra", "profile_id": "t7_rati_clarity", "active": True, "role": "voice"},
            4: {"slot": 5, "player": "Timmy", "character": "Magnus", "profile_id": "t6_laptop_fan", "active": True, "role": "voice"},
            5: {"slot": 6, "player": "-", "character": "Inactive", "profile_id": "skip", "active": False, "role": "empty"},
        }
    else:  # sw5e
        return {
            0: {"slot": 1, "player": "Robin", "character": "Cratebreaker", "profile_id": "t1_room_echo", "active": True, "role": "voice"},
            1: {"slot": 2, "player": "Tino", "character": "It'Mir", "profile_id": "t2_muffled", "active": True, "role": "voice"},
            2: {"slot": 3, "player": "Blu", "character": "Caelen", "profile_id": "t3_reference", "active": True, "role": "voice"},
            3: {"slot": 4, "player": "Marc", "character": "GM", "profile_id": "t4_megaphone", "active": True, "role": "voice"},
            4: {"slot": 5, "player": "Mathew", "character": "Salova", "profile_id": "t5_bleed_gate", "active": True, "role": "voice"},
            5: {"slot": 6, "player": "Timmy", "character": "Belial", "profile_id": "t6_laptop_fan", "active": True, "role": "voice"},
        }


def predict_video_stem_mapping(
    video_path: str,
    campaign_mode: str = "sw5e",
    custom_track_mapping: Optional[Dict[int, int]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Instantly (<50ms) predict all slot assignments, player names, and destination WAV paths
    WITHOUT decoding or extracting audio. Used for immediate UI feedback.
    """
    probe = probe_video_streams(video_path)
    if not probe.get("success"):
        return {"success": False, "error": probe.get("error")}

    audio_streams = probe.get("audio_streams", [])
    num_streams = len(audio_streams)
    if num_streams == 0:
        return {"success": False, "error": "No audio tracks found inside video container"}

    if not output_dir:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        parent_dir = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(parent_dir, f"{base_name}_Stems")

    default_mapping = get_default_video_track_mapping(campaign_mode, num_streams=num_streams)
    is_multichannel = probe.get("is_multichannel_single_stream", False)

    slots: Dict[int, Dict[str, Any]] = {}
    music_path = None

    if not is_multichannel:
        for i in range(num_streams):
            info = default_mapping.get(i, {})
            slot = info.get("slot")
            if custom_track_mapping and i in custom_track_mapping:
                slot = custom_track_mapping[i]
            role = info.get("role", "voice")
            player_name = info.get("player", f"Track_{i+1}")
            character = info.get("character", "")
            profile_id = info.get("profile_id", "t3_reference")

            if role == "music":
                music_path = os.path.join(output_dir, "Track_Music.wav")
            elif slot is not None and slot >= 1 and role == "voice":
                clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '', player_name.replace(' ', '_'))
                out_filename = f"Track_{slot:02d}_{clean_name}.wav"
                out_path = os.path.join(output_dir, out_filename)
                slots[slot] = {
                    "slot": slot,
                    "stream_idx": i,
                    "path": out_path,
                    "filename": out_filename,
                    "player": player_name,
                    "character": character,
                    "profile_id": profile_id,
                    "active": True
                }
    else:
        num_channels = min(audio_streams[0]["channels"], 6)
        for ch_idx in range(num_channels):
            slot = ch_idx + 1
            info = default_mapping.get(ch_idx, {})
            player_name = info.get("player", f"Speaker {slot}")
            character = info.get("character", "")
            profile_id = info.get("profile_id", "t3_reference")
            clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '', player_name.replace(' ', '_'))
            out_filename = f"Track_{slot:02d}_{clean_name}.wav"
            out_path = os.path.join(output_dir, out_filename)
            slots[slot] = {
                "slot": slot,
                "stream_idx": 0,
                "channel_idx": ch_idx,
                "path": out_path,
                "filename": out_filename,
                "player": player_name,
                "character": character,
                "profile_id": profile_id,
                "active": True
            }

    return {
        "success": True,
        "video_path": video_path,
        "output_dir": output_dir,
        "duration_sec": probe.get("duration_sec", 0.0),
        "num_audio_streams": num_streams,
        "slots": slots,
        "music_path": music_path,
        "is_multichannel_single_stream": is_multichannel
    }


def extract_video_audio_stems(
    video_path: str,
    output_dir: Optional[str] = None,
    campaign_mode: str = "sw5e",
    custom_track_mapping: Optional[Dict[int, int]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    log_func: Callable[[str], None] = print
) -> Dict[int, str]:
    """
    Extract all discrete audio tracks from a 4K video container in a SINGLE pass.
    Outputs pristine 24-bit PCM WAV (48 kHz) stems.
    
    Args:
        video_path: Path to .mkv, .mp4, or .mov container.
        output_dir: Destination folder for WAV stems (defaults to <video_name>_Stems).
        campaign_mode: 'sw5e' or 'red'.
        custom_track_mapping: Optional dict mapping stream_idx (0..5) -> slot_idx (1..6).
        progress_callback: Callback receiving {"percent": float, "speed": float, "status": str}.
        cancel_check: Callable returning True if cancelled.
        log_func: Logging function.
        
    Returns:
        Dict mapping slot_number (1..6) to extracted WAV file path.
    """
    probe = probe_video_streams(video_path)
    if not probe.get("success"):
        raise RuntimeError(f"Cannot inspect video file: {probe.get('error')}")

    audio_streams = probe.get("audio_streams", [])
    if not audio_streams:
        raise RuntimeError(f"No audio tracks found inside video: {video_path}")

    total_duration = probe.get("duration_sec", 0.0)

    num_streams = len(audio_streams)
    default_mapping = get_default_video_track_mapping(campaign_mode, num_streams=num_streams)

    # Determine output folder
    if not output_dir:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        parent_dir = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(parent_dir, f"{base_name}_Stems")
    os.makedirs(output_dir, exist_ok=True)

    extracted_slots: Dict[int, str] = {}

    
    # Case 1: Multiple discrete audio streams (MeldStudio 8-track or OBS multi-track)
    if not probe.get("is_multichannel_single_stream"):
        log_func(f"[Video Ingest] Found {num_streams} discrete audio streams in video container.")

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-progress", "pipe:1",
            "-nostats",
            "-i", video_path,
            "-vn"  # Skip decoding 4K video completely
        ]

        mapped_count = 0
        for i in range(num_streams):
            info = default_mapping.get(i, {})
            slot = info.get("slot")
            if custom_track_mapping and i in custom_track_mapping:
                slot = custom_track_mapping[i]
            role = info.get("role", "voice")
            player_name = info.get("player", f"Track_{i+1}")

            if role == "music":
                out_filename = "Track_Music.wav"
                out_path = os.path.join(output_dir, out_filename)
                cmd.extend([
                    "-map", f"0:a:{i}",
                    "-c:a", "pcm_s24le",
                    "-ar", "48000",
                    out_path
                ])
                log_func(f"  → Stream {i} (T{i+1}): Exporting BGM stem: {out_filename}")
            elif slot is not None and slot >= 1 and role == "voice":
                clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '', player_name.replace(' ', '_'))
                out_filename = f"Track_{slot:02d}_{clean_name}.wav"
                out_path = os.path.join(output_dir, out_filename)
                extracted_slots[slot] = out_path
                mapped_count += 1

                cmd.extend([
                    "-map", f"0:a:{i}",
                    "-c:a", "pcm_s24le",
                    "-ar", "48000",
                    out_path
                ])
                log_func(f"  → Stream {i} (T{i+1}): Channel {slot} [{player_name}] → {out_filename}")
            elif role == "mixdown":
                log_func(f"  → Stream {i} (T{i+1}): Master Mixdown (skipped for discrete processing)")
            elif role == "empty":
                log_func(f"  → Stream {i} (T{i+1}): Empty track (skipped)")

        log_func(f"[Video Ingest] Starting single-pass extraction of {mapped_count} voice stem(s) (24-bit PCM WAV)...")

    # Case 2: Multichannel single stream (e.g. 5.1/hexaphonic single audio stream)
    else:
        num_channels = min(audio_streams[0]["channels"], 6)
        log_func(f"[Video Ingest] Found single {num_channels}-channel audio stream. Splitting discrete channels...")

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-progress", "pipe:1",
            "-nostats",
            "-i", video_path,
            "-vn"
        ]

        filter_parts = []
        for i in range(num_channels):
            filter_parts.append(f"[0:a:0]pan=mono|c0=c{i}[a{i}]")
        cmd.extend(["-filter_complex", ";".join(filter_parts)])

        for i in range(num_channels):
            slot = stream_to_slot.get(i, i + 1)
            player_info = default_mapping.get(i, {})
            player_name = player_info.get("player", f"Track_{i+1}")
            clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '', player_name.replace(' ', '_'))
            
            out_filename = f"Track_{slot:02d}_{clean_name}.wav"
            out_path = os.path.join(output_dir, out_filename)
            extracted_slots[slot] = out_path

            cmd.extend([
                "-map", f"[a{i}]",
                "-c:a", "pcm_s24le",
                "-ar", "48000",
                out_path
            ])

    # Check if stems were already fully extracted previously
    all_exist = all(os.path.isfile(p) and os.path.getsize(p) > 1024 for p in extracted_slots.values())
    if all_exist and len(extracted_slots) > 0:
        log_func(f"[Video Ingest] All {len(extracted_slots)} stems already extracted in: {output_dir}")
        if progress_callback:
            progress_callback({
                "percent": 100.0,
                "current_sec": round(total_duration, 1),
                "total_sec": round(total_duration, 1),
                "speed": 100.0,
                "status": "Stems ready (cached)."
            })
        return extracted_slots

    # Execute ffmpeg with real-time progress parsing
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    t_start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        startupinfo=startupinfo,
        bufsize=1
    )

    stderr_lines = []
    def _drain_stderr():
        try:
            for l in iter(proc.stderr.readline, ''):
                if l:
                    stderr_lines.append(l)
        except Exception:
            pass

    err_t = threading.Thread(target=_drain_stderr, daemon=True)
    err_t.start()

    out_time_us = 0
    cur_speed = 1.0

    while True:
        if cancel_check and cancel_check():
            proc.kill()
            log_func("[-] Video stem extraction cancelled.")
            raise RuntimeError("Extraction cancelled by user")

        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("out_time_us="):
            try:
                out_time_us = int(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("speed="):
            speed_str = line.split("=")[1].replace("x", "").strip()
            try:
                cur_speed = float(speed_str)
            except ValueError:
                pass
        elif line.startswith("progress="):
            current_sec = out_time_us / 1_000_000.0
            pct = (current_sec / total_duration * 100.0) if total_duration > 0 else 0.0
            pct = min(100.0, max(0.0, pct))
            
            if progress_callback:
                progress_callback({
                    "percent": round(pct, 1),
                    "current_sec": round(current_sec, 1),
                    "total_sec": round(total_duration, 1),
                    "speed": round(cur_speed, 1),
                    "status": f"Extracting audio: {pct:.1f}% ({cur_speed:.1f}x real-time)"
                })

    proc.wait()
    err_t.join(timeout=1.0)
    stderr_output = "".join(stderr_lines)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg extraction failed (code {proc.returncode}): {stderr_output.strip()}")

    elapsed = time.time() - t_start
    log_func(f"[Video Ingest] Extracted {len(extracted_slots)} stems in {elapsed:.1f}s (~{cur_speed:.1f}x real-time).")

    if progress_callback:
        progress_callback({
            "percent": 100.0,
            "current_sec": round(total_duration, 1),
            "total_sec": round(total_duration, 1),
            "speed": round(cur_speed, 1),
            "status": "Extraction complete."
        })

    return extracted_slots

