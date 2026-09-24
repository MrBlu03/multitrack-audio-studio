#!/usr/bin/env python3
"""
DaVinci Resolve Timeline Generator
==================================
Generates native DaVinci Resolve sequence timelines (.fcpxml and .xml)
with pre-cropped, scaled, and translated 4K multicam video tracks and
synchronized mastered audio stems.

Campaign Presets:
- Cyberpunk RED (V1: Robin, V2: Blu, V3: Marc, V4: Rati, V5: Timmy, V6: Roll20)
- Star Wars 5e (V1: Robin, V2: Tino, V3: Blu, V4: Marc, V5: Mathew, V6: Timmy, V7: Roll20)

Both formats (FCPXML v1.9 and FCP 7 XML) are supported for 1-click import into DaVinci Resolve:
File > Import > Timeline...
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom


RESOLVE_CAMERA_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "red": [
        {"track": "V1", "name": "Robin", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 1920.0, "pos_y": -1080.0, "color": "Purple", "slot": 1},
        {"track": "V2", "name": "Blu", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": -1920.0, "pos_y": -1080.0, "color": "Cyan", "slot": 2},
        {"track": "V3", "name": "Marc", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 0.0, "pos_y": 0.0, "color": "Orange", "slot": 3},
        {"track": "V4", "name": "Rati", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": -1920.0, "pos_y": 0.0, "color": "Pink", "slot": 4},
        {"track": "V5", "name": "Timmy", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 1920.0, "pos_y": 0.0, "color": "Yellow", "slot": 5},
        {"track": "V6", "name": "Roll 20", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": -1920.0, "pos_y": 1080.0, "color": "Blue", "slot": None},
    ],
    "sw5e": [
        {"track": "V1", "name": "Robin", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 1920.0, "pos_y": -1080.0, "color": "Purple", "slot": 1},
        {"track": "V2", "name": "Tino", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 0.0, "pos_y": -1080.0, "color": "Yellow", "slot": 2},
        {"track": "V3", "name": "Blu", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": -1920.0, "pos_y": -1080.0, "color": "Cyan", "slot": 3},
        {"track": "V4", "name": "Marc", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 0.0, "pos_y": 0.0, "color": "Orange", "slot": 4},
        {"track": "V5", "name": "Mathew", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": -1920.0, "pos_y": 0.0, "color": "Green", "slot": 5},
        {"track": "V6", "name": "Timmy", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": 1920.0, "pos_y": 0.0, "color": "Sand", "slot": 6},
        {"track": "V7", "name": "Roll 20", "zoom_x": 3.0, "zoom_y": 3.0, "pos_x": -1920.0, "pos_y": 1080.0, "color": "Blue", "slot": None},
    ]
}


def get_media_duration_seconds(file_path: str) -> float:
    """Probe media file duration in seconds using ffprobe or soundfile fallback."""
    if not file_path or not os.path.isfile(file_path):
        return 0.0
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
        val = float(out.decode().strip())
        if val > 0:
            return val
    except Exception:
        pass

    try:
        import soundfile as sf
        info = sf.info(file_path)
        return float(info.duration)
    except Exception:
        pass

    return 3600.0  # Safe default 1 hour if unmeasurable


def format_rational_time(seconds: float, timebase: int = 30) -> str:
    """Format duration in rational seconds (e.g. '12345/30s') for FCPXML."""
    frames = int(round(seconds * timebase))
    return f"{frames}/{timebase}s"


def generate_resolve_fcpxml(
    video_path: Optional[str],
    audio_tracks: Dict[str, str],
    output_path: str,
    campaign_mode: str = "red",
    timeline_name: Optional[str] = None,
    fps: int = 30,
    width: int = 3840,
    height: int = 2160
) -> str:
    """
    Generate standard FCPXML v1.9 with:
    - Stacked video tracks V1..V6/V7 pre-cropped with exact Resolve Inspector Transform values.
    - Mastered audio stems mapped to dedicated dialogue roles / audio tracks.
    """
    campaign = campaign_mode.lower() if campaign_mode in RESOLVE_CAMERA_PRESETS else "red"
    cam_preset = RESOLVE_CAMERA_PRESETS[campaign]

    if not timeline_name:
        if video_path and os.path.isfile(video_path):
            timeline_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            timeline_name = f"{campaign.upper()} Multicam Timeline"

    # Determine timeline duration
    duration_sec = 0.0
    if video_path and os.path.isfile(video_path):
        duration_sec = get_media_duration_seconds(video_path)
    if duration_sec <= 0.0:
        for a_path in audio_tracks.values():
            if a_path and os.path.isfile(a_path):
                dur = get_media_duration_seconds(a_path)
                if dur > duration_sec:
                    duration_sec = dur

    if duration_sec <= 0.0:
        duration_sec = 3600.0

    duration_str = format_rational_time(duration_sec, fps)
    frame_dur_str = f"1/{fps}s"

    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")

    # Format definition (e.g. 4K UHD 3840x2160 @ 30fps)
    ET.SubElement(
        resources, "format",
        id="r_fmt",
        name=f"FFVideoFormat{width}x{height}p{fps}",
        frameDuration=frame_dur_str,
        width=str(width),
        height=str(height)
    )

    # Video Asset
    vid_asset_id = "r_video"
    has_video = video_path and os.path.isfile(video_path)
    if has_video:
        ET.SubElement(
            resources, "asset",
            id=vid_asset_id,
            name=os.path.basename(video_path),
            src=Path(os.path.abspath(video_path)).as_uri(),
            format="r_fmt",
            duration=duration_str,
            hasVideo="1",
            hasAudio="1"
        )

    # Audio Assets
    audio_asset_ids = {}
    for idx, (label, a_path) in enumerate(audio_tracks.items()):
        if a_path and os.path.isfile(a_path):
            a_id = f"r_audio_{idx + 1}"
            audio_asset_ids[label] = a_id
            ET.SubElement(
                resources, "asset",
                id=a_id,
                name=os.path.basename(a_path),
                src=Path(os.path.abspath(a_path)).as_uri(),
                duration=duration_str,
                hasAudio="1",
                hasVideo="0"
            )

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=timeline_name)
    project = ET.SubElement(event, "project", name=timeline_name)
    sequence = ET.SubElement(
        project, "sequence",
        format="r_fmt",
        duration=duration_str,
        tcStart="0s"
    )
    spine = ET.SubElement(sequence, "spine")

    # Primary Video Clip (V1)
    v1_cfg = cam_preset[0]
    if has_video:
        primary_clip = ET.SubElement(
            spine, "asset-clip",
            ref=vid_asset_id,
            offset="0s",
            name=f"{v1_cfg['track']}: {v1_cfg['name']}",
            duration=duration_str,
            start="0s",
            format="r_fmt"
        )
        ET.SubElement(
            primary_clip, "adjust-transform",
            enabled="1",
            position=f"{v1_cfg['pos_x']:.2f} {v1_cfg['pos_y']:.2f}",
            scale=f"{v1_cfg['zoom_x']:.2f} {v1_cfg['zoom_y']:.2f}",
            rotation="0"
        )

        # Connected Video Tracks: V2, V3, V4... stacked on lanes 1, 2, 3...
        for lane_idx, cfg in enumerate(cam_preset[1:], start=1):
            conn_video = ET.SubElement(
                primary_clip, "asset-clip",
                lane=str(lane_idx),
                offset="0s",
                name=f"{cfg['track']}: {cfg['name']}",
                duration=duration_str,
                start="0s",
                ref=vid_asset_id,
                format="r_fmt"
            )
            ET.SubElement(
                conn_video, "adjust-transform",
                enabled="1",
                position=f"{cfg['pos_x']:.2f} {cfg['pos_y']:.2f}",
                scale=f"{cfg['zoom_x']:.2f} {cfg['zoom_y']:.2f}",
                rotation="0"
            )

        # Connected Audio Tracks: stacked on negative lanes -1, -2, -3...
        for lane_idx, (label, a_id) in enumerate(audio_asset_ids.items(), start=1):
            role_tag = f"dialogue.{lane_idx}" if "music" not in label.lower() else "music.1"
            ET.SubElement(
                primary_clip, "asset-clip",
                lane=str(-lane_idx),
                offset="0s",
                name=label,
                duration=duration_str,
                start="0s",
                ref=a_id,
                role=role_tag
            )
    else:
        # Audio-only timeline if no 4K video supplied
        first_audio_label = next(iter(audio_asset_ids)) if audio_asset_ids else None
        if first_audio_label:
            first_aid = audio_asset_ids[first_audio_label]
            primary_clip = ET.SubElement(
                spine, "asset-clip",
                ref=first_aid,
                offset="0s",
                name=first_audio_label,
                duration=duration_str,
                start="0s",
                role="dialogue.1"
            )
            for lane_idx, (label, a_id) in enumerate(list(audio_asset_ids.items())[1:], start=1):
                role_tag = f"dialogue.{lane_idx + 1}" if "music" not in label.lower() else "music.1"
                ET.SubElement(
                    primary_clip, "asset-clip",
                    lane=str(-lane_idx),
                    offset="0s",
                    name=label,
                    duration=duration_str,
                    start="0s",
                    ref=a_id,
                    role=role_tag
                )

    raw_xml = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(raw_xml)
    pretty_xml = parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    # Add DOCTYPE fcpxml
    if "<!DOCTYPE fcpxml>" not in pretty_xml:
        pretty_xml = pretty_xml.replace(
            '<fcpxml version="1.9">',
            '<!DOCTYPE fcpxml>\n<fcpxml version="1.9">'
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    return output_path


def generate_resolve_fcp7_xml(
    video_path: Optional[str],
    audio_tracks: Dict[str, str],
    output_path: str,
    campaign_mode: str = "red",
    timeline_name: Optional[str] = None,
    fps: int = 30,
    width: int = 3840,
    height: int = 2160
) -> str:
    """
    Generate classic FCP 7 XML (xmeml v5) with:
    - Discrete <track> elements for V1..V6/V7 with Basic Motion scaling/centering.
    - Discrete <track> elements for A1..A6/A7 with mastered stems.
    """
    campaign = campaign_mode.lower() if campaign_mode in RESOLVE_CAMERA_PRESETS else "red"
    cam_preset = RESOLVE_CAMERA_PRESETS[campaign]

    if not timeline_name:
        if video_path and os.path.isfile(video_path):
            timeline_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            timeline_name = f"{campaign.upper()} Multicam Timeline"

    duration_sec = 0.0
    if video_path and os.path.isfile(video_path):
        duration_sec = get_media_duration_seconds(video_path)
    if duration_sec <= 0.0:
        for a_path in audio_tracks.values():
            if a_path and os.path.isfile(a_path):
                dur = get_media_duration_seconds(a_path)
                if dur > duration_sec:
                    duration_sec = dur
    if duration_sec <= 0.0:
        duration_sec = 3600.0

    total_frames = int(round(duration_sec * fps))

    root = ET.Element("xmeml", version="5")
    seq = ET.SubElement(root, "sequence", id="sequence-1")
    ET.SubElement(seq, "name").text = timeline_name
    ET.SubElement(seq, "duration").text = str(total_frames)

    rate = ET.SubElement(seq, "rate")
    ET.SubElement(rate, "timebase").text = str(fps)
    ET.SubElement(rate, "ntsc").text = "FALSE"

    media = ET.SubElement(seq, "media")
    video_elem = ET.SubElement(media, "video")

    # Format
    fmt = ET.SubElement(video_elem, "format")
    sc = ET.SubElement(fmt, "samplecharacteristics")
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)
    ET.SubElement(sc, "pixelaspectratio").text = "square"
    sc_rate = ET.SubElement(sc, "rate")
    ET.SubElement(sc_rate, "timebase").text = str(fps)
    ET.SubElement(sc_rate, "ntsc").text = "FALSE"

    has_video = video_path and os.path.isfile(video_path)

    # Video Tracks
    if has_video:
        video_abs_path = os.path.abspath(video_path)
        video_path_url = Path(video_abs_path).as_uri()

        for idx, cfg in enumerate(cam_preset, start=1):
            track_elem = ET.SubElement(video_elem, "track")
            clip_id = f"clipitem-v{idx}"
            ci = ET.SubElement(track_elem, "clipitem", id=clip_id)
            ET.SubElement(ci, "name").text = f"{cfg['track']}: {cfg['name']}"
            ET.SubElement(ci, "duration").text = str(total_frames)
            ET.SubElement(ci, "start").text = "0"
            ET.SubElement(ci, "end").text = str(total_frames)
            ET.SubElement(ci, "in").text = "0"
            ET.SubElement(ci, "out").text = str(total_frames)

            # Master file element
            file_elem = ET.SubElement(ci, "file", id="file-master-video")
            ET.SubElement(file_elem, "name").text = os.path.basename(video_path)
            ET.SubElement(file_elem, "pathurl").text = video_path_url
            f_rate = ET.SubElement(file_elem, "rate")
            ET.SubElement(f_rate, "timebase").text = str(fps)
            ET.SubElement(f_rate, "ntsc").text = "FALSE"
            ET.SubElement(file_elem, "duration").text = str(total_frames)

            # Basic Motion filter
            filt = ET.SubElement(ci, "filter")
            eff = ET.SubElement(filt, "effect")
            ET.SubElement(eff, "name").text = "Basic Motion"
            ET.SubElement(eff, "effectid").text = "basic"
            ET.SubElement(eff, "effecttype").text = "motion"
            ET.SubElement(eff, "mediatype").text = "video"

            # Scale parameter (300%)
            p_scale = ET.SubElement(eff, "parameter")
            ET.SubElement(p_scale, "parameterid").text = "scale"
            ET.SubElement(p_scale, "name").text = "Scale"
            ET.SubElement(p_scale, "value").text = str(int(cfg["zoom_x"] * 100))

            # Center position parameter
            norm_x = (cfg["pos_x"] / (width / 2.0)) * 100.0 if width > 0 else 0.0
            norm_y = (cfg["pos_y"] / (height / 2.0)) * 100.0 if height > 0 else 0.0
            p_center = ET.SubElement(eff, "parameter")
            ET.SubElement(p_center, "parameterid").text = "center"
            ET.SubElement(p_center, "name").text = "Center"
            val_elem = ET.SubElement(p_center, "value")
            ET.SubElement(val_elem, "horiz").text = f"{norm_x:.1f}"
            ET.SubElement(val_elem, "vert").text = f"{norm_y:.1f}"

    # Audio Tracks
    audio_elem = ET.SubElement(media, "audio")
    for idx, (label, a_path) in enumerate(audio_tracks.items(), start=1):
        if a_path and os.path.isfile(a_path):
            a_abs = os.path.abspath(a_path)
            a_uri = Path(a_abs).as_uri()

            a_track = ET.SubElement(audio_elem, "track")
            ci_a = ET.SubElement(a_track, "clipitem", id=f"clipitem-a{idx}")
            ET.SubElement(ci_a, "name").text = label
            ET.SubElement(ci_a, "duration").text = str(total_frames)
            ET.SubElement(ci_a, "start").text = "0"
            ET.SubElement(ci_a, "end").text = str(total_frames)
            ET.SubElement(ci_a, "in").text = "0"
            ET.SubElement(ci_a, "out").text = str(total_frames)

            f_elem = ET.SubElement(ci_a, "file", id=f"file-a{idx}")
            ET.SubElement(f_elem, "name").text = os.path.basename(a_path)
            ET.SubElement(f_elem, "pathurl").text = a_uri
            fr = ET.SubElement(f_elem, "rate")
            ET.SubElement(fr, "timebase").text = str(fps)
            ET.SubElement(fr, "ntsc").text = "FALSE"
            ET.SubElement(f_elem, "duration").text = str(total_frames)

    raw_xml = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(raw_xml)
    pretty_xml = parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    if "<!DOCTYPE xmeml>" not in pretty_xml:
        pretty_xml = pretty_xml.replace(
            '<xmeml version="5">',
            '<!DOCTYPE xmeml>\n<xmeml version="5">'
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    return output_path


def auto_generate_resolve_timelines(
    session_dir: str,
    mastered_dir: Optional[str] = None,
    video_path: Optional[str] = None,
    campaign_mode: str = "red",
    log_func: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Auto-detect session video and mastered audio files and export BOTH:
    1. <Session>_Resolve_Timeline.fcpxml (FCPXML v1.9)
    2. <Session>_Resolve_Timeline.xml (FCP 7 XML)
    
    Returns dict mapping 'fcpxml' and 'xml' to absolute file paths.
    """
    _log = log_func or print

    if not mastered_dir:
        mastered_dir = os.path.join(session_dir, "Mastered")
        if not os.path.isdir(mastered_dir):
            mastered_dir = session_dir

    # Search for video if not explicitly provided
    if not video_path:
        candidates = [os.path.join(session_dir, f) for f in os.listdir(session_dir)
                      if f.lower().endswith(('.mp4', '.mov', '.mkv'))]
        if candidates:
            video_path = candidates[0]
        else:
            meld_dir = r"E:\Recordings\Meld"
            if os.path.isdir(meld_dir):
                m_files = [os.path.join(meld_dir, f) for f in os.listdir(meld_dir)
                           if f.lower().endswith(('.mp4', '.mov', '.mkv'))]
                if m_files:
                    m_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    video_path = m_files[0]

    campaign = campaign_mode.lower() if campaign_mode in RESOLVE_CAMERA_PRESETS else "red"
    cam_preset = RESOLVE_CAMERA_PRESETS[campaign]
    audio_tracks: Dict[str, str] = {}
    # Collect audio files (prefer Mastered, fallback to session_dir)
    audio_files = []
    if os.path.isdir(mastered_dir):
        audio_files.extend([os.path.join(mastered_dir, f) for f in os.listdir(mastered_dir)
                           if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))])
    if not audio_files and os.path.isdir(session_dir):
        audio_files.extend([os.path.join(session_dir, f) for f in os.listdir(session_dir)
                           if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))])

    if audio_files:
        files = audio_files
        for cfg in cam_preset:
            slot_num = cfg.get("slot")
            player = cfg["name"].lower()
            if not slot_num:
                continue
            chosen = None
            for fp in files:
                fl = os.path.basename(fp).lower()
                if (
                    f"track_{slot_num}" in fl or
                    f"track_a0{slot_num}" in fl or
                    f"track_{slot_num:02d}" in fl or
                    player in fl
                ):
                    chosen = fp
                    break
            if chosen:
                audio_tracks[f"A{slot_num}: {cfg['name']} (Mastered)"] = chosen

        music_candidates = [
            os.path.join(mastered_dir, "Track_Music.wav"),
            os.path.join(session_dir, "Track_Music.wav")
        ]
        for mp in music_candidates:
            if os.path.isfile(mp):
                audio_tracks["Music (Stem)"] = mp
                break

    session_name = os.path.basename(os.path.normpath(session_dir))
    out_dir = mastered_dir if os.path.isdir(mastered_dir) else session_dir
    base_name = f"{session_name}_Resolve_Timeline"
    fcpxml_path = os.path.join(out_dir, f"{base_name}.fcpxml")
    xml_path = os.path.join(out_dir, f"{base_name}.xml")

    _log(f"[Resolve Exporter] Generating DaVinci Resolve timelines for '{session_name}' ({campaign.upper()})...")
    generate_resolve_fcpxml(
        video_path=video_path,
        audio_tracks=audio_tracks,
        output_path=fcpxml_path,
        campaign_mode=campaign,
        timeline_name=f"{session_name} Multicam"
    )
    generate_resolve_fcp7_xml(
        video_path=video_path,
        audio_tracks=audio_tracks,
        output_path=xml_path,
        campaign_mode=campaign,
        timeline_name=f"{session_name} Multicam"
    )

    _log(f"[Resolve Exporter] [OK] FCPXML exported: {fcpxml_path}")
    _log(f"[Resolve Exporter] [OK] FCP 7 XML exported: {xml_path}")

    return {
        "fcpxml": fcpxml_path,
        "xml": xml_path,
        "video_path": video_path or "",
        "audio_tracks": audio_tracks
    }
