#!/usr/bin/env python3
"""
Graphical User Interface for Multitrack Bleed Suppressor & Studio Audio Restorer
=================================================================================
Dual-Tool Audio Suite:
1. 🎙️ Multitrack Bleed Suppressor: Activity-aware spectral gating to eliminate
   acoustic bleed from lossy MP3/Discord recordings while preserving speech.
2. 💻 Laptop Mic Restorer: Eliminates internal cooling fan whine & chassis rumble,
   removes hollow desk reflections, and restores warmth, air, and broadcast leveling.

Author: Antigravity DSP Tools
"""

import os
import sys
import re
import math
import time
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Dict, Any

import soundfile as sf
import numpy as np

# Ensure stdout and stderr are valid in both windowed GUI and console modes
if sys.platform == "win32" and sys.stdout is None:
    import ctypes
    if ctypes.windll.kernel32.AttachConsole(-1):
        try:
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        except Exception:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
    else:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import core DSP engines
try:
    from multitrack_bleed_gate import (
        AudioSource,
        BleedCalibrator,
        StreamingMultitrackBleedGate,
        LaptopMicEnhancer,
        process_laptop_mic_file,
        DialogueSafeSilenceGate,
        EnsembleVocalRestorer,
        process_vocal_restoration_file,
        export_as_mp3,
        format_time,
        DEFAULT_SLOT_PROFILES,
        CAMPAIGN_CONFIGS,
        get_campaign_config,
        PROFILE_CHOICES,
        profile_str_to_id,
        profile_id_to_str,
        auto_detect_session_tracks,
        process_automated_session
    )
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from multitrack_bleed_gate import (
        AudioSource,
        BleedCalibrator,
        StreamingMultitrackBleedGate,
        LaptopMicEnhancer,
        process_laptop_mic_file,
        DialogueSafeSilenceGate,
        EnsembleVocalRestorer,
        process_vocal_restoration_file,
        export_as_mp3,
        format_time,
        DEFAULT_SLOT_PROFILES,
        CAMPAIGN_CONFIGS,
        get_campaign_config,
        PROFILE_CHOICES,
        profile_str_to_id,
        profile_id_to_str,
        auto_detect_session_tracks,
        process_automated_session
    )

# Import multitrack speech-to-text transcriber
try:
    from multitrack_transcriber import (
        MultitrackTranscriber,
        export_txt,
        export_srt,
        export_json,
        export_moderation_report,
        export_timeline_markers_davinci,
        export_timeline_markers_premiere,
        format_timestamp as format_ts_transcribe,
        setup_cuda_dll_paths,
        correct_campaign_vocabulary,
        correct_transcript_file,
    )
    setup_cuda_dll_paths()
    HAS_TRANSCRIBER = True
except Exception as e:
    HAS_TRANSCRIBER = False

# Try importing windnd for native Windows drag-and-drop
try:
    import windnd
    HAS_WINDND = True
except ImportError:
    HAS_WINDND = False


def natural_sort_key(s: str):
    """Sort strings with numbers naturally (e.g. track1, track2, track10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class BleedGateGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multitrack Audio & Transcription Studio v4.0 (Automated Chain & Transcriber)")
        self.root.geometry("1060x880")
        self.root.minsize(960, 740)
        self._apply_dark_titlebar()

        # Campaign State: Default "sw5e", supports toggle to "red"
        self.campaign_mode_var = tk.StringVar(value="sw5e")
        cfg = get_campaign_config("sw5e")

        # Tab 0: 1-Click Session Auto-Master state
        self.auto_slots = {}
        for s in range(1, 7):
            s_cfg = cfg["slots"].get(s, {})
            self.auto_slots[s] = {
                "path_var": tk.StringVar(value=""),
                "prof_var": tk.StringVar(value=profile_id_to_str(s_cfg.get("profile_id", "t3_reference"))),
                "label_var": tk.StringVar(value=s_cfg.get("label", f"Slot {s}:"))
            }
        self.auto_out_dir_var = tk.StringVar(value="")
        self.auto_export_fmt_var = tk.StringVar(value="MP3 (320 kbps Broadcast)")
        self.auto_silence_gate_var = tk.BooleanVar(value=True)
        self.auto_ai_denoise_var = tk.BooleanVar(value=True)
        self.auto_ai_strength_var = tk.StringVar(value="100% (Maximum Elimination)")
        self.auto_normalize_var = tk.BooleanVar(value=True)
        self.auto_target_lufs_var = tk.StringVar(value="-18 LUFS (Default - Broadcast Dialogue / Equal Volume)")
        self.auto_session_dir: Optional[str] = None

        # Tab 1: Multitrack Bleed state
        self.track_files: List[str] = []
        self.is_multichannel = False
        self.bleed_silence_gate_var = tk.BooleanVar(value=False)
        self.bleed_export_fmt_var = tk.StringVar(value="WAV (24-bit PCM)")

        # Tab 2: Laptop Mic state
        self.laptop_file: Optional[str] = None
        self.laptop_out_path_var = tk.StringVar(value="")
        self.laptop_silence_gate_var = tk.BooleanVar(value=False)
        self.laptop_ai_denoise_var = tk.BooleanVar(value=True)
        self.laptop_ai_strength_var = tk.StringVar(value="100% (Maximum Elimination)")
        self.laptop_export_fmt_var = tk.StringVar(value="WAV (24-bit PCM)")

        # Tab 3: Studio Voice Restorer state
        self.studio_file: Optional[str] = None
        self.studio_out_path_var = tk.StringVar(value="")
        self.studio_profile_var = tk.StringVar(value="Track 1: Room Echo & Slapback Suppressor (Warmth + Dry Studio)")
        self.studio_silence_gate_var = tk.BooleanVar(value=True)
        self.studio_ai_denoise_var = tk.BooleanVar(value=False)
        self.studio_ai_strength_var = tk.StringVar(value="100% (Maximum Elimination)")
        self.studio_thresh_var = tk.StringVar(value="-34 dB")
        self.studio_hold_var = tk.StringVar(value="280 ms")
        self.studio_floor_var = tk.StringVar(value="Pure Digital Silence (-inf dB)")
        self.studio_export_fmt_var = tk.StringVar(value="MP3 (320 kbps Broadcast)")

        # Tab 4: 📝 Multitrack Transcriber state
        self.transcribe_slots = {}
        for s in range(1, 7):
            s_cfg = cfg["slots"].get(s, {})
            self.transcribe_slots[s] = {
                "path_var": tk.StringVar(value=""),
                "name_var": tk.StringVar(value=s_cfg.get("speaker", f"Speaker {s}")),
                "active_var": tk.BooleanVar(value=s_cfg.get("active", True)),
                "label_var": tk.StringVar(value=s_cfg.get("label", f"Slot {s}:"))
            }
        self.transcribe_model_var = tk.StringVar(value="large-v3 (Default - Highest Accuracy Benchmark)")
        self.transcribe_device_var = tk.StringVar(value="⚡ Auto (NVIDIA GPU CUDA -> Multi-Core CPU Fallback)")
        self.transcribe_prompt_var = tk.StringVar(value=cfg["prompt"])
        self.transcribe_gap_var = tk.StringVar(value="1.2s (Conversational Flow - Snappy Turns)")
        self.transcribe_out_file_var = tk.StringVar(value="")
        self.transcribe_export_srt_var = tk.BooleanVar(value=True)
        self.transcribe_export_json_var = tk.BooleanVar(value=False)
        self.transcribe_moderation_var = tk.BooleanVar(value=True)
        self.transcribe_automute_var = tk.BooleanVar(value=False)
        self.transcribe_flagged_terms_var = tk.StringVar(value="nigger, nigga, faggot, retard")
        self.last_transcript_file: Optional[str] = None
        self.subtitle_lbl_var = tk.StringVar(
            value=f"🎛️ Automated Audio Chain ({cfg['name']}) & 📝 Speech-to-Text Transcriber"
        )

        # Common processing state
        self.is_processing = False
        self.cancel_requested = False
        self.worker_thread: Optional[threading.Thread] = None
        self.last_output_file: Optional[str] = None
        self.msg_queue: queue.Queue = queue.Queue()

        self._setup_styles()
        self._build_ui()
        self._check_queue()

        # Register Drag and Drop
        if HAS_WINDND:
            try:
                windnd.hook_dropfiles(self.root, func=self._on_files_dropped)
            except Exception as e:
                print(f"[-] Note: windnd hook failed: {e}")

    def _apply_dark_titlebar(self):
        """Enable Windows 10/11 native immersive dark titlebar."""
        if sys.platform != "win32":
            return
        import ctypes
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            val = ctypes.c_int(1)
            # DWMWA_USE_IMMERSIVE_DARK_MODE (20 for Win11 / Win10 20H1+, 19 for older Win10)
            res = ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
            if res != 0:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(val), ctypes.sizeof(val))
        except Exception:
            pass

    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Refined Studio DAW Dark Mode Palette (Sleek, High Contrast, Low Eye-Strain)
        self.bg_color = "#0f1117"          # Deep midnight canvas
        self.card_bg = "#181b24"           # Elevated matte card surface
        self.panel_bg = "#222736"          # Interactive elevated panel
        self.border_color = "#282f42"      # Subtle modern border
        self.input_bg = "#13161f"          # Recessed dark input background
        self.accent_cyan = "#38bdf8"       # Electric Sapphire (active/headers/progress)
        self.accent_green = "#10b981"      # Studio Emerald (primary actions/success)
        self.accent_pink = "#f43f5e"       # Studio Rose (cancel/warnings)
        self.accent_amber = "#f59e0b"      # Studio Amber (preview/attention)
        self.text_color = "#e2e8f0"        # Clean readable slate
        self.text_dim = "#94a3b8"          # Muted slate for descriptions/captions
        self.text_bright = "#f8fafc"       # Crisp pure ice white

        self.root.configure(bg=self.bg_color)
        self.style.configure(".", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 9))
        self.style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        self.style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), background=self.card_bg, foreground=self.accent_cyan)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9), background=self.card_bg, foreground=self.text_dim)
        self.style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), background=self.bg_color, foreground=self.text_bright)
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 9, "bold"), background=self.bg_color, foreground=self.accent_green)
        self.style.configure("BannerCyan.TLabel", font=("Segoe UI", 9), background="#132433", foreground=self.accent_cyan, relief="flat", borderwidth=0, padding=6)
        self.style.configure("BannerGreen.TLabel", font=("Segoe UI", 8), background="#122c22", foreground=self.accent_green, relief="flat", borderwidth=0, padding=5)

        # Action Buttons
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), foreground="#0f1117", background=self.accent_green, borderwidth=0, relief="flat")
        self.style.map("Primary.TButton", background=[("active", "#34d399"), ("disabled", "#132e24")], foreground=[("disabled", "#2d6a54"), ("!disabled", "#0f1117")])

        self.style.configure("Preview.TButton", font=("Segoe UI", 10, "bold"), foreground=self.accent_cyan, background="#182738", bordercolor=self.accent_cyan, borderwidth=1, relief="solid")
        self.style.map("Preview.TButton", background=[("active", "#20354d")], foreground=[("disabled", "#1e3a52")])

        self.style.configure("Cancel.TButton", font=("Segoe UI", 10, "bold"), foreground=self.accent_pink, background="#2e141a", bordercolor=self.accent_pink, borderwidth=1, relief="solid")
        self.style.map("Cancel.TButton", background=[("active", "#451c25")], foreground=[("disabled", "#4a1e27")])

        self.style.configure("TButton", font=("Segoe UI", 9), foreground=self.text_color, background=self.panel_bg, bordercolor=self.border_color, borderwidth=1, relief="solid")
        self.style.map("TButton", background=[("active", "#2d3447")], foreground=[("active", self.accent_cyan)])

        # Notebook & Sleek Padded Tabs
        self.style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[20, 8], background=self.card_bg, foreground=self.text_dim, borderwidth=1)
        self.style.map("TNotebook.Tab", background=[("selected", self.panel_bg)], foreground=[("selected", self.accent_cyan)])

        # Recessed Obsidian Input Fields
        self.style.configure("TEntry", fieldbackground=self.input_bg, foreground=self.text_bright, insertcolor=self.accent_cyan, bordercolor=self.border_color, lightcolor=self.border_color, darkcolor=self.border_color)
        self.style.configure("TCombobox", fieldbackground=self.input_bg, background=self.panel_bg, foreground=self.text_bright, selectbackground=self.panel_bg, selectforeground=self.accent_cyan, arrowcolor=self.accent_cyan, bordercolor=self.border_color)
        self.style.map("TCombobox", fieldbackground=[("readonly", self.input_bg)], selectbackground=[("readonly", self.input_bg)], selectforeground=[("readonly", self.text_bright)])

        # Checkbuttons
        self.style.configure("TCheckbutton", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9))
        self.style.map("TCheckbutton", background=[("active", self.card_bg)], foreground=[("active", self.accent_cyan)])

        # Progressbar & Scrollbar
        self.style.configure("Horizontal.TProgressbar", troughcolor=self.input_bg, background=self.accent_green, bordercolor=self.border_color)
        self.style.configure("Vertical.TScrollbar", troughcolor=self.bg_color, background=self.panel_bg, bordercolor=self.border_color, arrowcolor=self.accent_cyan)


    def _build_ui(self):
        header_frame = ttk.Frame(self.root, padding="15 10 15 4")
        header_frame.pack(fill="x")
        
        top_row = ttk.Frame(header_frame)
        top_row.pack(fill="x")

        title_lbl = ttk.Label(top_row, text="🎙️ Multitrack Audio & Transcription Studio v4.0", style="Title.TLabel")
        title_lbl.pack(side="left", anchor="w")

        # Campaign Switcher Toggle Buttons (Right side of header)
        campaign_box = ttk.Frame(top_row)
        campaign_box.pack(side="right", anchor="e")

        ttk.Label(campaign_box, text="Campaign Mode:", font=("Segoe UI", 9, "bold"), foreground=self.text_dim).pack(side="left", padx=(0, 6))

        self.btn_mode_sw5e = tk.Button(
            campaign_box,
            text="🌌 Star Wars 5e",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
            command=lambda: self._set_campaign_mode("sw5e")
        )
        self.btn_mode_sw5e.pack(side="left", padx=(0, 4))

        self.btn_mode_red = tk.Button(
            campaign_box,
            text="🦾 Cyberpunk RED",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
            command=lambda: self._set_campaign_mode("red")
        )
        self.btn_mode_red.pack(side="left")

        subtitle_lbl = ttk.Label(
            header_frame,
            textvariable=self.subtitle_lbl_var,
            style="Subtitle.TLabel"
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        main_frame = ttk.Frame(self.root, padding="15 4 15 10")
        main_frame.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="x", pady=(0, 4))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # TAB 1: 🎛️ Automated Audio Chain Processing (1-Click Session Master)
        self.tab_auto = ttk.Frame(self.notebook, padding="6")
        self.notebook.add(self.tab_auto, text="  🎛️ 1. Automated Audio Chain Processing  ")
        self._build_tab_auto()

        # TAB 2: 📝 Multitrack Transcriber & Script Merger
        self.tab_transcribe = ttk.Frame(self.notebook, padding="6")
        self.notebook.add(self.tab_transcribe, text="  📝 2. Multitrack Transcriber & Script Merger  ")
        self._build_tab_transcribe()

        # Bottom Shared Action & Progress Panel
        self._build_exec_panel(main_frame)
        self._update_campaign_toggle_ui()

    def _set_campaign_mode(self, mode: str):
        if self.campaign_mode_var.get() == mode:
            return
        self.campaign_mode_var.set(mode)
        cfg = get_campaign_config(mode)

        # Update header subtitle
        self.subtitle_lbl_var.set(
            f"🎛️ Automated Audio Chain ({cfg['name']}) & 📝 Speech-to-Text Transcriber"
        )

        # Update toggle buttons visually
        self._update_campaign_toggle_ui()

        # Update auto-master slots
        for s in range(1, 7):
            s_cfg = cfg["slots"].get(s, {})
            self.auto_slots[s]["label_var"].set(s_cfg.get("label", f"Slot {s}:"))
            if not self.auto_slots[s]["path_var"].get():
                def_prof = s_cfg.get("profile_id", "t3_reference")
                self.auto_slots[s]["prof_var"].set(profile_id_to_str(def_prof))

        # Update transcriber slots
        for s in range(1, 7):
            s_cfg = cfg["slots"].get(s, {})
            self.transcribe_slots[s]["label_var"].set(s_cfg.get("label", f"Slot {s}:"))
            if not self.transcribe_slots[s]["path_var"].get():
                self.transcribe_slots[s]["name_var"].set(s_cfg.get("speaker", f"Speaker {s}"))
                self.transcribe_slots[s]["active_var"].set(s_cfg.get("active", True))
            else:
                if not s_cfg.get("active", True):
                    self.transcribe_slots[s]["active_var"].set(False)

        # Update transcription context prompt
        self.transcribe_prompt_var.set(cfg["prompt"])

        self._log(f"[⚡] Switched to Campaign Profile: {cfg['icon']} {cfg['name']}")

    def _update_campaign_toggle_ui(self):
        mode = self.campaign_mode_var.get()
        if mode == "red":
            self.btn_mode_red.configure(bg=self.accent_pink, fg="#ffffff", activebackground="#fb7185", activeforeground="#ffffff")
            self.btn_mode_sw5e.configure(bg="#181b24", fg=self.text_dim, activebackground="#222736", activeforeground=self.text_color)
        else:
            self.btn_mode_sw5e.configure(bg=self.accent_cyan, fg="#0f1117", activebackground="#7dd3fc", activeforeground="#0f1117")
            self.btn_mode_red.configure(bg="#181b24", fg=self.text_dim, activebackground="#222736", activeforeground=self.text_color)


    def _build_tab_auto(self):
        # Card 1: Session Source
        card_src = ttk.Frame(self.tab_auto, style="Card.TFrame", padding="10")
        card_src.pack(fill="x", pady=3)

        c1_head = ttk.Frame(card_src, style="Card.TFrame")
        c1_head.pack(fill="x", pady=(0, 4))
        ttk.Label(c1_head, text="1. Session Audio Source", style="Header.TLabel").pack(side="left")
        ttk.Label(c1_head, text=" (Drag & Drop session folder or 6 audio tracks)", font=("Segoe UI", 8, "italic"), background=self.card_bg, foreground=self.text_dim).pack(side="left", padx=6)

        btn_box = ttk.Frame(card_src, style="Card.TFrame")
        btn_box.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_box, text="📁 Select Session Folder...", command=self._select_auto_folder).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="📁 Select Track Files...", command=self._select_auto_files).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="🗑️ Clear All Slots", command=self._clear_auto_slots).pack(side="right")

        self.auto_src_info_var = tk.StringVar(value="💡 Drop a session folder (e.g. SW5E SESSION 7 TRACKS) or 6 audio files here to auto-map all slots.")
        ttk.Label(card_src, textvariable=self.auto_src_info_var, style="BannerCyan.TLabel").pack(fill="x")

        # Card 2: 6-Track Session Slot Matrix
        card_slots = ttk.Frame(self.tab_auto, style="Card.TFrame", padding="10")
        card_slots.pack(fill="x", pady=3)

        c2_head = ttk.Frame(card_slots, style="Card.TFrame")
        c2_head.pack(fill="x", pady=(0, 6))
        ttk.Label(c2_head, text="2. Multitrack Session Mapping (Auto-Detected Slots 1 to 6)", style="Header.TLabel").pack(side="left")
        for slot_num in range(1, 7):
            row_frame = ttk.Frame(card_slots, style="Card.TFrame")
            row_frame.pack(fill="x", pady=2)

            ttk.Label(row_frame, textvariable=self.auto_slots[slot_num]["label_var"], font=("Segoe UI", 9, "bold"), width=18, background=self.card_bg, foreground=self.text_color).pack(side="left")

            entry = ttk.Entry(row_frame, textvariable=self.auto_slots[slot_num]["path_var"], font=("Segoe UI", 8))
            entry.pack(side="left", fill="x", expand=True, padx=(2, 6))

            combo = ttk.Combobox(
                row_frame,
                textvariable=self.auto_slots[slot_num]["prof_var"],
                values=PROFILE_CHOICES,
                state="readonly",
                width=38
            )
            combo.pack(side="left", padx=(0, 6))

            btn = ttk.Button(row_frame, text="Browse...", width=9, command=lambda s=slot_num: self._select_single_slot_file(s))
            btn.pack(side="right")

        # Card 3: Session Output Destination & Master Settings
        card_out = ttk.Frame(self.tab_auto, style="Card.TFrame", padding="10")
        card_out.pack(fill="x", pady=3)

        c3_head = ttk.Frame(card_out, style="Card.TFrame")
        c3_head.pack(fill="x", pady=(0, 4))
        ttk.Label(c3_head, text="3. Mastering Settings & Output Destination", style="Header.TLabel").pack(side="left")

        out_grid = ttk.Frame(card_out, style="Card.TFrame")
        out_grid.pack(fill="x")

        ttk.Label(out_grid, text="Master Output Folder:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=3)
        self.auto_out_entry = ttk.Entry(out_grid, textvariable=self.auto_out_dir_var, font=("Segoe UI", 9))
        self.auto_out_entry.grid(row=0, column=1, sticky="we", padx=6, pady=3)
        ttk.Button(out_grid, text="Browse...", command=self._select_auto_out_dir).grid(row=0, column=2, sticky="e", pady=3)
        out_grid.columnconfigure(1, weight=1)

        opt_row = ttk.Frame(card_out, style="Card.TFrame")
        opt_row.pack(fill="x", pady=(4, 0))

        ttk.Label(opt_row, text="Export Format:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        fmt_combo = ttk.Combobox(
            opt_row,
            textvariable=self.auto_export_fmt_var,
            values=["MP3 (320 kbps Broadcast)", "WAV (24-bit PCM)"],
            state="readonly",
            width=24
        )
        fmt_combo.pack(side="left", padx=(0, 16))

        chk_silence = ttk.Checkbutton(
            opt_row,
            text="✅ Dialogue-Safe Digital Silence Gate (Pure 0.000000 in pauses, 30ms lookahead, zero vocal cutoff)",
            variable=self.auto_silence_gate_var
        )
        chk_silence.pack(side="left")

        ai_row = ttk.Frame(card_out, style="Card.TFrame")
        ai_row.pack(fill="x", pady=(6, 2))

        chk_ai = ttk.Checkbutton(
            ai_row,
            text="🧠 Neural AI Noise Suppression (RNNoise Deep Learning Voice Isolator across all speech)",
            variable=self.auto_ai_denoise_var
        )
        chk_ai.pack(side="left", padx=(0, 12))

        ttk.Label(ai_row, text="AI Strength:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 4))
        ai_combo = ttk.Combobox(
            ai_row,
            textvariable=self.auto_ai_strength_var,
            values=[
                "100% (Maximum Elimination)",
                "85% (Strong Suppression)",
                "70% (Balanced Natural)",
                "50% (Gentle Hiss Removal)"
            ],
            state="readonly",
            width=26
        )
        ai_combo.pack(side="left")

        norm_row = ttk.Frame(card_out, style="Card.TFrame")
        norm_row.pack(fill="x", pady=(6, 2))

        chk_norm = ttk.Checkbutton(
            norm_row,
            text="⚖️ Match Vocal Loudness Across Tracks (ITU-R BS.1770 Speech Normalizer + Peak Limiter)",
            variable=self.auto_normalize_var
        )
        chk_norm.pack(side="left", padx=(0, 12))

        ttk.Label(norm_row, text="Target Dialogue Level:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 4))
        norm_combo = ttk.Combobox(
            norm_row,
            textvariable=self.auto_target_lufs_var,
            values=[
                "-18 LUFS (Default - Broadcast Dialogue / Equal Volume)",
                "-16 LUFS (Punchy & Upfront Podcast)",
                "-20 LUFS (Dynamic Natural / Film)",
                "-23 LUFS (Traditional EBU R128 Television)"
            ],
            state="readonly",
            width=42
        )
        norm_combo.pack(side="left")

        handoff_row = ttk.Frame(card_out, style="Card.TFrame")
        handoff_row.pack(fill="x", pady=(8, 2))
        ttk.Button(
            handoff_row,
            text="➡️ Send Mastered Stems to Transcriber Tab",
            style="Preview.TButton",
            command=self._send_mastered_to_transcriber
        ).pack(side="right")
        ttk.Label(
            handoff_row,
            text="💡 Seamless Handoff: Once mastering completes, click this button to automatically load your finished stems into the Transcriber tab.",
            font=("Segoe UI", 8, "italic"),
            background=self.card_bg,
            foreground=self.text_dim
        ).pack(side="left")

    def _build_tab_bleed(self):
        # Card 1: Audio Inputs
        card_inputs = ttk.Frame(self.tab_bleed, style="Card.TFrame", padding="10")
        card_inputs.pack(fill="x", pady=3)

        card1_head = ttk.Frame(card_inputs, style="Card.TFrame")
        card1_head.pack(fill="x", pady=(0, 4))
        ttk.Label(card1_head, text="1. Audio Source Tracks", style="Header.TLabel").pack(side="left")
        dnd_note = " (Drag & Drop files supported)" if HAS_WINDND else ""
        ttk.Label(card1_head, text=dnd_note, font=("Segoe UI", 8, "italic"), background=self.card_bg, foreground="#757575").pack(side="left", padx=6)

        btn_box = ttk.Frame(card_inputs, style="Card.TFrame")
        btn_box.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_box, text="📁 Select Track Files...", command=self._select_multi_files).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="📁 Select Multichannel WAV...", command=self._select_multichannel_file).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="🗑️ Clear Tracks", command=self._clear_tracks).pack(side="right")

        columns = ("track", "name", "duration", "sr", "channels", "status")
        self.tree = ttk.Treeview(card_inputs, columns=columns, show="headings", height=5, selectmode="browse")
        self.tree.heading("track", text="Track #")
        self.tree.heading("name", text="File Name")
        self.tree.heading("duration", text="Duration")
        self.tree.heading("sr", text="Sample Rate")
        self.tree.heading("channels", text="Channels")
        self.tree.heading("status", text="Role")

        self.tree.column("track", width=65, anchor="center")
        self.tree.column("name", width=380, anchor="w")
        self.tree.column("duration", width=80, anchor="center")
        self.tree.column("sr", width=90, anchor="center")
        self.tree.column("channels", width=65, anchor="center")
        self.tree.column("status", width=130, anchor="center")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Card 2: Bleed Parameters
        card_settings = ttk.Frame(self.tab_bleed, style="Card.TFrame", padding="10")
        card_settings.pack(fill="x", pady=3)
        ttk.Label(card_settings, text="2. Target Track & Bleed Parameters", style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        grid_frame = ttk.Frame(card_settings, style="Card.TFrame")
        grid_frame.pack(fill="x")

        ttk.Label(grid_frame, text="Target Bleed Track:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=3)
        self.target_var = tk.StringVar(value="Track 5")
        self.target_combo = ttk.Combobox(grid_frame, textvariable=self.target_var, state="readonly", width=14)
        self.target_combo.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        self.target_combo.bind("<<ComboboxSelected>>", self._on_target_changed)

        ttk.Label(grid_frame, text="Profile Preset:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(16, 0), pady=3)
        self.preset_var = tk.StringVar(value="Standard Studio (45% Match Cutoff, -60 dB Floor)")
        preset_combo = ttk.Combobox(
            grid_frame,
            textvariable=self.preset_var,
            state="readonly",
            values=[
                "Standard Studio (45% Match Cutoff, -60 dB Floor)",
                "Aggressive Bleed Cut (38% Match Cutoff, -inf dB Floor)",
                "Whisper / Sensitive Vocals (55% Match Cutoff, 1.35x Sens)",
                "Complete Digital Silence (-inf dB Floor)",
                "Subtle Room Ambience (-36 dB Floor)"
            ],
            width=36
        )
        preset_combo.grid(row=0, column=3, sticky="w", padx=6, pady=3)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        ttk.Label(grid_frame, text="Bleed Match Cutoff:", background=self.card_bg).grid(row=1, column=0, sticky="w", pady=3)
        self.cutoff_val_var = tk.StringVar(value="48%")
        self.cutoff_slider = ttk.Scale(grid_frame, from_=25, to=75, value=48, command=self._on_cutoff_slider)
        self.cutoff_slider.grid(row=1, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(grid_frame, textvariable=self.cutoff_val_var, background=self.card_bg, width=8).grid(row=1, column=2, sticky="w")

        ttk.Label(grid_frame, text="Vocal Sensitivity:", background=self.card_bg).grid(row=1, column=2, sticky="w", padx=(16, 0), pady=3)
        self.sens_val_var = tk.StringVar(value="1.00x")
        self.sens_slider = ttk.Scale(grid_frame, from_=0.5, to=2.0, value=1.0, command=self._on_sens_slider)
        self.sens_slider.grid(row=1, column=3, sticky="we", padx=6, pady=3)
        ttk.Label(grid_frame, textvariable=self.sens_val_var, background=self.card_bg, width=8).grid(row=1, column=4, sticky="w")

        ttk.Label(grid_frame, text="Mute Floor:", background=self.card_bg).grid(row=2, column=0, sticky="w", pady=3)
        self.floor_val_var = tk.StringVar(value="-60 dB")
        self.floor_slider = ttk.Scale(grid_frame, from_=-99, to=-24, value=-60, command=self._on_floor_slider)
        self.floor_slider.grid(row=2, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(grid_frame, textvariable=self.floor_val_var, background=self.card_bg, width=8).grid(row=2, column=2, sticky="w")

        ttk.Label(grid_frame, text="Hold Time:", background=self.card_bg).grid(row=2, column=2, sticky="w", padx=(16, 0), pady=3)
        self.hold_val_var = tk.StringVar(value="160 ms")
        self.hold_slider = ttk.Scale(grid_frame, from_=50, to=500, value=160, command=self._on_hold_slider)
        self.hold_slider.grid(row=2, column=3, sticky="we", padx=6, pady=3)
        ttk.Label(grid_frame, textvariable=self.hold_val_var, background=self.card_bg, width=8).grid(row=2, column=4, sticky="w")

        ttk.Label(grid_frame, text="Release Time:", background=self.card_bg).grid(row=3, column=0, sticky="w", pady=3)
        self.rel_val_var = tk.StringVar(value="85 ms")
        self.rel_slider = ttk.Scale(grid_frame, from_=50, to=400, value=85, command=self._on_rel_slider)
        self.rel_slider.grid(row=3, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(grid_frame, textvariable=self.rel_val_var, background=self.card_bg, width=8).grid(row=3, column=2, sticky="w")

        self.enhance_var = tk.BooleanVar(value=True)
        self.enhance_chk = ttk.Checkbutton(grid_frame, text="✨ Broadcast Vocal Tuning (EQ De-box, Warmth & Leveler)", variable=self.enhance_var)
        self.enhance_chk.grid(row=3, column=2, columnspan=3, sticky="w", padx=(16, 0), pady=3)

        # Card 3: Output Destination
        card_out = ttk.Frame(self.tab_bleed, style="Card.TFrame", padding="10")
        card_out.pack(fill="x", pady=3)
        ttk.Label(card_out, text="3. Output Cleaned File Destination", style="Header.TLabel").pack(anchor="w", pady=(0, 4))
        out_row = ttk.Frame(card_out, style="Card.TFrame")
        out_row.pack(fill="x")
        self.out_path_var = tk.StringVar(value="")
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_path_var, font=("Segoe UI", 9))
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(out_row, text="Browse...", command=self._select_output_file).pack(side="right")

    def _build_tab_laptop(self):
        # Card 1: Laptop Audio File
        card_l_input = ttk.Frame(self.tab_laptop, style="Card.TFrame", padding="10")
        card_l_input.pack(fill="x", pady=3)
        
        c1_head = ttk.Frame(card_l_input, style="Card.TFrame")
        c1_head.pack(fill="x", pady=(0, 4))
        ttk.Label(c1_head, text="1. Laptop Microphone Track", style="Header.TLabel").pack(side="left")
        ttk.Label(c1_head, text=" (Fan noise elimination & proximity acoustic EQ)", font=("Segoe UI", 8, "italic"), background=self.card_bg, foreground="#757575").pack(side="left", padx=6)

        l_btn_box = ttk.Frame(card_l_input, style="Card.TFrame")
        l_btn_box.pack(fill="x", pady=(0, 6))
        ttk.Button(l_btn_box, text="📁 Select Track File (MP3/WAV)...", command=self._select_laptop_file).pack(side="left", padx=(0, 6))
        ttk.Button(l_btn_box, text="📥 Import Track 6 from Bleed Tab", command=self._import_track6_from_tab1).pack(side="left", padx=(0, 6))

        self.laptop_info_var = tk.StringVar(value="No laptop track loaded. Select Track 6 above or drag & drop here.")
        self.laptop_info_lbl = ttk.Label(card_l_input, textvariable=self.laptop_info_var, font=("Segoe UI", 9, "bold"), background="#e8f5e9", foreground="#2e7d32", padding="6")
        self.laptop_info_lbl.pack(fill="x")

        # Card 2: Acoustic Restoration Controls
        card_l_settings = ttk.Frame(self.tab_laptop, style="Card.TFrame", padding="10")
        card_l_settings.pack(fill="x", pady=3)
        ttk.Label(card_l_settings, text="2. Acoustic Restoration & De-Rooming Parameters", style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        l_grid = ttk.Frame(card_l_settings, style="Card.TFrame")
        l_grid.pack(fill="x")

        ttk.Label(l_grid, text="Restoration Preset:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=3)
        self.laptop_preset_var = tk.StringVar(value="Studio Balanced (Recommended - 1.6x Denoise, -40 dB Floor)")
        l_preset_combo = ttk.Combobox(
            l_grid,
            textvariable=self.laptop_preset_var,
            state="readonly",
            values=[
                "Studio Balanced (Recommended - 1.6x Denoise, -40 dB Floor)",
                "Aggressive Fan Whine Removal (2.2x Denoise, -50 dB Floor)",
                "Subtle / Quiet Room (1.1x Denoise, -30 dB Floor)",
                "Broadcast Punch & Clarity (+6 dB Air, 1.8x Denoise, -45 dB Floor)"
            ],
            width=48
        )
        l_preset_combo.grid(row=0, column=1, columnspan=3, sticky="w", padx=6, pady=3)
        l_preset_combo.bind("<<ComboboxSelected>>", self._on_laptop_preset_changed)

        # Sliders
        ttk.Label(l_grid, text="Fan Denoise Intensity:", background=self.card_bg).grid(row=1, column=0, sticky="w", pady=3)
        self.l_denoise_var = tk.StringVar(value="1.60x")
        self.l_denoise_slider = ttk.Scale(l_grid, from_=0.5, to=3.0, value=1.6, command=lambda v: self.l_denoise_var.set(f"{float(v):.2f}x"))
        self.l_denoise_slider.grid(row=1, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(l_grid, textvariable=self.l_denoise_var, background=self.card_bg, width=8).grid(row=1, column=2, sticky="w")

        ttk.Label(l_grid, text="Silence Expander Floor:", background=self.card_bg).grid(row=1, column=2, sticky="w", padx=(16, 0), pady=3)
        self.l_floor_var = tk.StringVar(value="-40 dB")
        self.l_floor_slider = ttk.Scale(l_grid, from_=-60, to=-20, value=-40, command=lambda v: self.l_floor_var.set(f"{int(float(v))} dB"))
        self.l_floor_slider.grid(row=1, column=3, sticky="we", padx=6, pady=3)
        ttk.Label(l_grid, textvariable=self.l_floor_var, background=self.card_bg, width=8).grid(row=1, column=4, sticky="w")

        ttk.Label(l_grid, text="Proximity Warmth (160 Hz):", background=self.card_bg).grid(row=2, column=0, sticky="w", pady=3)
        self.l_warmth_var = tk.StringVar(value="+2.5 dB")
        self.l_warmth_slider = ttk.Scale(l_grid, from_=0.0, to=6.0, value=2.5, command=lambda v: self.l_warmth_var.set(f"+{float(v):.1f} dB"))
        self.l_warmth_slider.grid(row=2, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(l_grid, textvariable=self.l_warmth_var, background=self.card_bg, width=8).grid(row=2, column=2, sticky="w")

        ttk.Label(l_grid, text="Desk Bounce Cut (340 Hz):", background=self.card_bg).grid(row=2, column=2, sticky="w", padx=(16, 0), pady=3)
        self.l_desk_var = tk.StringVar(value="-5.0 dB")
        self.l_desk_slider = ttk.Scale(l_grid, from_=-10.0, to=0.0, value=-5.0, command=lambda v: self.l_desk_var.set(f"{float(v):.1f} dB"))
        self.l_desk_slider.grid(row=2, column=3, sticky="we", padx=6, pady=3)
        ttk.Label(l_grid, textvariable=self.l_desk_var, background=self.card_bg, width=8).grid(row=2, column=4, sticky="w")

        ttk.Label(l_grid, text="Hollow Chassis Cut (850 Hz):", background=self.card_bg).grid(row=3, column=0, sticky="w", pady=3)
        self.l_hollow_var = tk.StringVar(value="-3.5 dB")
        self.l_hollow_slider = ttk.Scale(l_grid, from_=-8.0, to=0.0, value=-3.5, command=lambda v: self.l_hollow_var.set(f"{float(v):.1f} dB"))
        self.l_hollow_slider.grid(row=3, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(l_grid, textvariable=self.l_hollow_var, background=self.card_bg, width=8).grid(row=3, column=2, sticky="w")

        ttk.Label(l_grid, text="Vocal Articulation (3.4 kHz):", background=self.card_bg).grid(row=3, column=2, sticky="w", padx=(16, 0), pady=3)
        self.l_presence_var = tk.StringVar(value="+4.5 dB")
        self.l_presence_slider = ttk.Scale(l_grid, from_=0.0, to=8.0, value=4.5, command=lambda v: self.l_presence_var.set(f"+{float(v):.1f} dB"))
        self.l_presence_slider.grid(row=3, column=3, sticky="we", padx=6, pady=3)
        ttk.Label(l_grid, textvariable=self.l_presence_var, background=self.card_bg, width=8).grid(row=3, column=4, sticky="w")

        # Row 4: AI Denoise
        chk_l_ai = ttk.Checkbutton(
            l_grid,
            text="🧠 Neural AI Noise Suppression (RNNoise Deep Learning Voice Isolator)",
            variable=self.laptop_ai_denoise_var
        )
        chk_l_ai.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 2))

        ttk.Label(l_grid, text="AI Strength:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=4, column=2, sticky="w", padx=(16, 0), pady=(6, 2))
        l_ai_combo = ttk.Combobox(
            l_grid,
            textvariable=self.laptop_ai_strength_var,
            values=[
                "100% (Maximum Elimination)",
                "85% (Strong Suppression)",
                "70% (Balanced Natural)",
                "50% (Gentle Hiss Removal)"
            ],
            state="readonly",
            width=26
        )
        l_ai_combo.grid(row=4, column=3, sticky="w", padx=6, pady=(6, 2))

        # Card 3: Output Destination
        card_l_out = ttk.Frame(self.tab_laptop, style="Card.TFrame", padding="10")
        card_l_out.pack(fill="x", pady=3)
        ttk.Label(card_l_out, text="3. Output Restored File Destination", style="Header.TLabel").pack(anchor="w", pady=(0, 4))
        l_out_row = ttk.Frame(card_l_out, style="Card.TFrame")
        l_out_row.pack(fill="x")
        self.laptop_out_entry = ttk.Entry(l_out_row, textvariable=self.laptop_out_path_var, font=("Segoe UI", 9))
        self.laptop_out_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(l_out_row, text="Browse...", command=self._select_laptop_output_file).pack(side="right")

    def _build_tab_studio(self):
        # Card 1: Input Audio File
        card_s_input = ttk.Frame(self.tab_studio, style="Card.TFrame", padding="10")
        card_s_input.pack(fill="x", pady=3)

        s_head = ttk.Frame(card_s_input, style="Card.TFrame")
        s_head.pack(fill="x", pady=(0, 4))
        ttk.Label(s_head, text="1. Select Dialogue Track to Restore", style="Header.TLabel").pack(side="left")
        ttk.Label(s_head, text=" (Supports Track 1, 2, 4, 5, or any vocal recording)", font=("Segoe UI", 8, "italic"), background=self.card_bg, foreground="#757575").pack(side="left", padx=6)

        s_row = ttk.Frame(card_s_input, style="Card.TFrame")
        s_row.pack(fill="x", pady=(0, 4))
        ttk.Button(s_row, text="📁 Select Audio File...", command=self._select_studio_file).pack(side="left", padx=(0, 6))
        ttk.Button(s_row, text="📥 Import Target Track from Bleed Tab", command=self._import_target_to_studio).pack(side="left", padx=(0, 6))
        ttk.Button(s_row, text="🗑️ Clear", command=self._clear_studio_file).pack(side="right")

        self.studio_file_lbl = ttk.Label(card_s_input, text="No file selected.", background=self.card_bg, font=("Segoe UI", 9, "bold"))
        self.studio_file_lbl.pack(anchor="w", pady=(2, 0))

        # Card 2: Restoration Profile & Digital Silence Parameters
        card_s_settings = ttk.Frame(self.tab_studio, style="Card.TFrame", padding="10")
        card_s_settings.pack(fill="x", pady=3)
        ttk.Label(card_s_settings, text="2. Restoration Profile & Voice Activity Silence Gate", style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        s_grid = ttk.Frame(card_s_settings, style="Card.TFrame")
        s_grid.pack(fill="x")

        ttk.Label(s_grid, text="Restoration Profile:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=4)
        profile_combo = ttk.Combobox(
            s_grid,
            textvariable=self.studio_profile_var,
            state="readonly",
            values=[
                "Track 1: Room Echo & Slapback Suppressor (Warmth + Dry Studio)",
                "Track 2: Heavy Blanket De-Muffler (Consonants & Air)",
                "Track 4: Megaphone & Sibilance De-Resonator (Plosive Cut)",
                "Track 5: Headset Mic De-Boxer (Proximity Warmth)",
                "Universal Clean Voice (Subtle Warmth & Leveler)",
                "🧠 Pure AI Voice Isolator (RNNoise Neural Network)",
                "Noise Suppressor Only (Pure Digital Silence Gate)"
            ],
            width=58
        )
        profile_combo.grid(row=0, column=1, columnspan=3, sticky="w", padx=6, pady=4)

        # Digital Silence Checkbox
        chk_silence = ttk.Checkbutton(
            s_grid,
            text="🔇 Apply Voice Activity Digital Silence Gate (Pure 0.000000 in Pauses, Zero Cutoff)",
            variable=self.studio_silence_gate_var
        )
        chk_silence.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 2))

        # Row 2: AI Denoise
        chk_ai = ttk.Checkbutton(
            s_grid,
            text="🧠 Neural AI Noise Suppression (RNNoise Deep Learning Voice Isolator)",
            variable=self.studio_ai_denoise_var
        )
        chk_ai.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 6))

        ttk.Label(s_grid, text="AI Strength:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=2, column=2, sticky="w", padx=(16, 0), pady=(2, 6))
        s_ai_combo = ttk.Combobox(
            s_grid,
            textvariable=self.studio_ai_strength_var,
            values=[
                "100% (Maximum Elimination)",
                "85% (Strong Suppression)",
                "70% (Balanced Natural)",
                "50% (Gentle Hiss Removal)"
            ],
            state="readonly",
            width=26
        )
        s_ai_combo.grid(row=2, column=3, sticky="w", padx=6, pady=(2, 6))

        ttk.Label(s_grid, text="Gate Sensitivity (Threshold):", background=self.card_bg).grid(row=3, column=0, sticky="w", pady=3)
        self.studio_thresh_slider = ttk.Scale(s_grid, from_=-45, to=-25, value=-34, command=lambda v: self.studio_thresh_var.set(f"{int(float(v))} dB"))
        self.studio_thresh_slider.grid(row=3, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(s_grid, textvariable=self.studio_thresh_var, background=self.card_bg, width=8).grid(row=3, column=2, sticky="w")

        ttk.Label(s_grid, text="Speech Hold Time:", background=self.card_bg).grid(row=3, column=2, sticky="w", padx=(16, 0), pady=3)
        self.studio_hold_slider = ttk.Scale(s_grid, from_=100, to=500, value=280, command=lambda v: self.studio_hold_var.set(f"{int(float(v))} ms"))
        self.studio_hold_slider.grid(row=3, column=3, sticky="we", padx=6, pady=3)
        ttk.Label(s_grid, textvariable=self.studio_hold_var, background=self.card_bg, width=8).grid(row=3, column=4, sticky="w")

        ttk.Label(s_grid, text="Silence Floor:", background=self.card_bg).grid(row=4, column=0, sticky="w", pady=3)
        floor_combo = ttk.Combobox(
            s_grid,
            textvariable=self.studio_floor_var,
            state="readonly",
            values=[
                "Pure Digital Silence (-inf dB)",
                "Subtle Room Tone (-60 dBFS)",
                "Gentle Floor (-50 dBFS)"
            ],
            width=28
        )
        floor_combo.grid(row=4, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(s_grid, text="Export Format:", background=self.card_bg, font=("Segoe UI", 9, "bold")).grid(row=4, column=2, sticky="w", padx=(16, 0), pady=3)
        fmt_combo = ttk.Combobox(
            s_grid,
            textvariable=self.studio_export_fmt_var,
            state="readonly",
            values=[
                "MP3 (320 kbps Broadcast)",
                "WAV (24-bit PCM)",
                "MP3 (192 kbps Standard)"
            ],
            width=24
        )
        fmt_combo.grid(row=4, column=3, sticky="w", padx=6, pady=3)
        fmt_combo.bind("<<ComboboxSelected>>", self._on_studio_format_changed)

        # Card 3: Output Destination
        card_s_out = ttk.Frame(self.tab_studio, style="Card.TFrame", padding="10")
        card_s_out.pack(fill="x", pady=3)
        ttk.Label(card_s_out, text="3. Output File Destination", style="Header.TLabel").pack(anchor="w", pady=(0, 4))
        s_out_row = ttk.Frame(card_s_out, style="Card.TFrame")
        s_out_row.pack(fill="x")
        self.studio_out_entry = ttk.Entry(s_out_row, textvariable=self.studio_out_path_var, font=("Segoe UI", 9))
        self.studio_out_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(s_out_row, text="Browse...", command=self._select_studio_output_file).pack(side="right")

    def _build_tab_transcribe(self):
        # Card 1: 6-Track Speaker Assignment Matrix
        card_matrix = ttk.Frame(self.tab_transcribe, style="Card.TFrame", padding="10")
        card_matrix.pack(fill="x", pady=3)

        c1_head = ttk.Frame(card_matrix, style="Card.TFrame")
        c1_head.pack(fill="x", pady=(0, 4))
        ttk.Label(c1_head, text="1. Multitrack Speaker Assignment Matrix", style="Header.TLabel").pack(side="left")
        ttk.Label(c1_head, text=" (Assign each track to its speaker for 100% accurate quotes)", font=("Segoe UI", 8, "italic"), background=self.card_bg, foreground=self.text_dim).pack(side="left", padx=6)

        btn_box = ttk.Frame(card_matrix, style="Card.TFrame")
        btn_box.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_box, text="📥 Import Tracks from Session Master", command=self._import_master_slots_to_transcriber).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="📁 Select Track Files...", command=self._select_transcribe_files).pack(side="left", padx=(0, 6))
        ttk.Button(btn_box, text="🗑️ Clear Matrix", command=self._clear_transcribe_slots).pack(side="right")

        self.transcribe_info_lbl = ttk.Label(
            card_matrix,
            text="💡 Tip: Import your Mastered or Session tracks, give each player their name, then click Transcribe below.",
            style="BannerGreen.TLabel"
        )
        self.transcribe_info_lbl.pack(fill="x", pady=(0, 6))

        for slot_num in range(1, 7):
            row_frame = ttk.Frame(card_matrix, style="Card.TFrame")
            row_frame.pack(fill="x", pady=2)

            chk = ttk.Checkbutton(row_frame, variable=self.transcribe_slots[slot_num]["active_var"])
            chk.pack(side="left", padx=(0, 4))

            ttk.Label(row_frame, textvariable=self.transcribe_slots[slot_num]["label_var"], font=("Segoe UI", 9, "bold"), width=18, background=self.card_bg, foreground=self.text_color).pack(side="left")

            entry = ttk.Entry(row_frame, textvariable=self.transcribe_slots[slot_num]["path_var"], font=("Segoe UI", 8))
            entry.pack(side="left", fill="x", expand=True, padx=(2, 6))

            ttk.Button(row_frame, text="Browse...", width=8, command=lambda s=slot_num: self._select_transcribe_single_file(s)).pack(side="left", padx=(0, 8))

            ttk.Label(row_frame, text="Speaker:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(4, 2))
            spk_entry = ttk.Entry(row_frame, textvariable=self.transcribe_slots[slot_num]["name_var"], width=18, font=("Segoe UI", 9))
            spk_entry.pack(side="left")

        # Card 2: AI Whisper Model & Context Settings
        card_model = ttk.Frame(self.tab_transcribe, style="Card.TFrame", padding="10")
        card_model.pack(fill="x", pady=3)

        c2_head = ttk.Frame(card_model, style="Card.TFrame")
        c2_head.pack(fill="x", pady=(0, 4))
        ttk.Label(c2_head, text="2. Speech-to-Text Model & Conversational Alignment", style="Header.TLabel").pack(side="left")

        m_grid = ttk.Frame(card_model, style="Card.TFrame")
        m_grid.pack(fill="x")

        ttk.Label(m_grid, text="Whisper Model:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=3)
        model_combo = ttk.Combobox(
            m_grid,
            textvariable=self.transcribe_model_var,
            state="readonly",
            values=[
                "large-v3 (Default - Highest Accuracy Benchmark)",
                "medium.en (High Precision - English Dedicated)",
                "small.en (Fast & Balanced)",
                "base.en (Quick Draft)",
                "tiny.en (Ultra-Fast Preview)",
                "large-v3-turbo (High Accuracy & Accelerated)"
            ],
            width=42
        )
        model_combo.grid(row=0, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(m_grid, text="Speaker Gap Merge:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(16, 0), pady=3)
        gap_combo = ttk.Combobox(
            m_grid,
            textvariable=self.transcribe_gap_var,
            state="readonly",
            values=[
                "0.8s (Rapid Interjections)",
                "1.2s (Conversational Flow - Snappy Turns)",
                "1.5s (Balanced)",
                "2.0s (Natural Flow)",
                "3.0s (Longer Paragraphs)"
            ],
            width=32
        )
        gap_combo.grid(row=0, column=3, sticky="w", padx=6, pady=3)

        # Row 1: Compute Engine
        ttk.Label(m_grid, text="Inference Engine:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", pady=3)
        device_combo = ttk.Combobox(
            m_grid,
            textvariable=self.transcribe_device_var,
            state="readonly",
            values=[
                "⚡ Auto (NVIDIA GPU CUDA -> Multi-Core CPU Fallback)",
                "🚀 Force GPU (NVIDIA CUDA float16)",
                "💻 Force CPU (Multi-core int8 - Guaranteed Compatibility)",
            ],
            width=42
        )
        device_combo.grid(row=1, column=1, sticky="w", padx=6, pady=3)

        # Row 2: Keyword prompt
        ttk.Label(m_grid, text="Campaign Vocabulary:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=4)
        prompt_entry = ttk.Entry(m_grid, textvariable=self.transcribe_prompt_var, font=("Segoe UI", 8))
        prompt_entry.grid(row=2, column=1, columnspan=3, sticky="we", padx=6, pady=4)
        m_grid.columnconfigure(1, weight=1)

        # Output format options
        fmt_row = ttk.Frame(card_model, style="Card.TFrame")
        fmt_row.pack(fill="x", pady=(4, 0))
        ttk.Label(fmt_row, text="Output Exports:", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        ttk.Label(fmt_row, text="[x] Formatted Dialogue Script (.txt)", background=self.card_bg, foreground=self.accent_cyan, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(fmt_row, text="🎬 SubRip Subtitles (.srt)", variable=self.transcribe_export_srt_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(fmt_row, text="📊 JSON Metadata (.json)", variable=self.transcribe_export_json_var).pack(side="left")

        # YouTube Content Safety & Slur Redaction
        safe_row = ttk.Frame(card_model, style="Card.TFrame")
        safe_row.pack(fill="x", pady=(6, 0))
        ttk.Label(safe_row, text="🛡️ YouTube Safety:", background=self.card_bg, foreground=self.accent_pink, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(safe_row, text="Flag Slurs & Generate Timeline Markers (DaVinci / Premiere)", variable=self.transcribe_moderation_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(safe_row, text="🔇 Auto-Mute Flagged Words in Audio Stems", variable=self.transcribe_automute_var).pack(side="left")

        safe_terms_row = ttk.Frame(card_model, style="Card.TFrame")
        safe_terms_row.pack(fill="x", pady=(4, 0))
        ttk.Label(safe_terms_row, text="Flagged Words:", background=self.card_bg, foreground=self.text_dim, font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))
        self.safe_terms_entry = ttk.Entry(safe_terms_row, textvariable=self.transcribe_flagged_terms_var, font=("Segoe UI", 8))
        self.safe_terms_entry.pack(side="left", fill="x", expand=True)
        ttk.Label(safe_terms_row, text="(comma-separated)", background=self.card_bg, foreground=self.text_dim, font=("Segoe UI", 8, "italic")).pack(side="left", padx=(6, 0))

        # Card 3: Output Destination
        card_t_out = ttk.Frame(self.tab_transcribe, style="Card.TFrame", padding="10")
        card_t_out.pack(fill="x", pady=3)
        c3_head = ttk.Frame(card_t_out, style="Card.TFrame")
        c3_head.pack(fill="x", pady=(0, 4))
        ttk.Label(c3_head, text="3. Output Transcript Destination", style="Header.TLabel").pack(side="left")
        ttk.Button(c3_head, text="✨ Clean Existing Transcript Names (.txt / .srt)...", command=self._clean_existing_transcript_file).pack(side="right")

        t_out_row = ttk.Frame(card_t_out, style="Card.TFrame")
        t_out_row.pack(fill="x")
        self.transcribe_out_entry = ttk.Entry(t_out_row, textvariable=self.transcribe_out_file_var, font=("Segoe UI", 9))
        self.transcribe_out_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(t_out_row, text="Browse...", command=self._select_transcribe_output_file).pack(side="right")

    def _build_exec_panel(self, parent):
        card_exec = ttk.Frame(parent, style="Card.TFrame", padding="10")
        card_exec.pack(fill="both", expand=True, pady=3)

        action_box = ttk.Frame(card_exec, style="Card.TFrame")
        action_box.pack(fill="x", pady=(0, 4))

        self.btn_preview = ttk.Button(
            action_box,
            text="⚡ Run 2-Minute Preview (Bleed Gate)",
            style="Preview.TButton",
            command=lambda: self._start_processing(preview_sec=120.0)
        )
        self.btn_preview.pack(side="left", padx=(0, 8))

        self.btn_process = ttk.Button(
            action_box,
            text="🚀 Process Full Session (Bleed Gate)",
            style="Primary.TButton",
            command=lambda: self._start_processing(preview_sec=None)
        )
        self.btn_process.pack(side="left", padx=(0, 8))

        self.btn_cancel = ttk.Button(
            action_box,
            text="⏹️ Cancel",
            style="Cancel.TButton",
            state="disabled",
            command=self._cancel_processing
        )
        self.btn_cancel.pack(side="left")

        self.btn_open_folder = ttk.Button(
            action_box,
            text="📂 Reveal in Explorer",
            command=self._open_output_folder
        )
        self.btn_open_folder.pack(side="right", padx=(6, 0))
        self.btn_open_folder.pack_forget()

        self.btn_open_transcript = ttk.Button(
            action_box,
            text="📄 Open Transcript (Notepad)",
            command=self._open_transcript_file
        )
        self.btn_open_transcript.pack(side="right", padx=(6, 0))
        self.btn_open_transcript.pack_forget()

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(card_exec, variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill="x", pady=(4, 2))

        self.status_lbl_var = tk.StringVar(value="Ready. Load your audio tracks or drop a session folder above.")
        ttk.Label(card_exec, textvariable=self.status_lbl_var, font=("Segoe UI", 9, "bold"), background=self.card_bg, foreground=self.accent_cyan).pack(anchor="w", pady=(0, 4))

        log_frame = ttk.Frame(card_exec, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame,
            height=6,
            wrap="word",
            bg="#0c0e14",
            fg=self.text_color,
            insertbackground=self.accent_cyan,
            selectbackground="#1e3a5f",
            selectforeground="#ffffff",
            font=("Consolas", 9),
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.accent_cyan,
            padx=8,
            pady=8
        )
        self.log_text.tag_configure("ok", foreground="#10b981")
        self.log_text.tag_configure("info", foreground="#38bdf8")
        self.log_text.tag_configure("warn", foreground="#f59e0b")
        self.log_text.tag_configure("err", foreground="#f43f5e")
        self.log_text.tag_configure("dim", foreground="#94a3b8")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _on_tab_changed(self, event=None):
        tab_idx = self.notebook.index(self.notebook.select())
        if tab_idx == 0:
            self.btn_preview.configure(text="⚡ Run 2-Minute Preview (All Tracks)")
            self.btn_process.configure(text="🚀 Master Entire Session (All Tracks)")
            if not self.is_processing:
                self.status_lbl_var.set("Ready for Automated Audio Chain Processing. Drop session folder or tracks above.")
        else:
            self.btn_preview.configure(text="⚡ Transcribe 2-Minute Preview")
            self.btn_process.configure(text="📝 Transcribe & Generate Complete Script")
            if not self.is_processing:
                self.status_lbl_var.set("Ready for Multitrack Transcription. Assign speaker names and click 'Transcribe & Generate Complete Script'.")

    def _check_queue(self):
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == "log":
                    try:
                        tag = None
                        stripped = payload.lstrip()
                        if stripped.startswith(("[+]", "[OK]", "SUCCESS", "✓")):
                            tag = "ok"
                        elif stripped.startswith(("[-]", "[ERROR]", "[ERR]", "FAILED", "✗")):
                            tag = "err"
                        elif stripped.startswith(("[!]", "[WARN]", "WARNING")):
                            tag = "warn"
                        elif stripped.startswith(("[*]", "[⚡]", "===", "---", ">>>", "🚀", "🧠", "⚖️")):
                            tag = "info"

                        
                        if tag:
                            self.log_text.insert("end", payload + "\n", tag)
                        else:
                            self.log_text.insert("end", payload + "\n")
                        self.log_text.see("end")
                    except Exception:
                        pass
                elif msg_type == "auto_progress":
                    track_idx, num_tracks, track_name, track_pct, total_pct, speed, eta_str, status_msg = payload
                    self.progress_var.set(total_pct)
                    self.status_lbl_var.set(f"[{track_idx}/{num_tracks}] {track_name} ({track_pct:.1f}%) | Total: {total_pct:.1f}% | Speed: {speed:.1f}x | ETA: {eta_str}")
                elif msg_type == "transcribe_progress":
                    track_idx, total_tracks, spk, curr_sec, tot_sec, pct, seg_text = payload
                    overall_pct = ((track_idx - 1) * 100.0 + pct) / max(1, total_tracks)
                    self.progress_var.set(overall_pct)
                    self.status_lbl_var.set(f"[{track_idx}/{total_tracks}] {spk} ({pct:.1f}%) | Overall: {overall_pct:.1f}% | Time: {format_time(curr_sec)}")
                elif msg_type == "progress":
                    pct, curr_sec, total_sec, speed, eta_sec = payload
                    self._update_progress_ui(pct, curr_sec, total_sec, speed, eta_sec)
                elif msg_type == "transcribe_finish":
                    success, msg, out_file = payload
                    self._finish_transcription(success, msg, out_file)
                elif msg_type == "finish":
                    success, msg = payload
                    self._finish_processing(success, msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self._check_queue)

    def _log(self, text: str):
        self.msg_queue.put(("log", text))

    # Drag and Drop
    def _on_files_dropped(self, files):
        file_list = []
        raw_paths = []
        for f in files:
            p = f.decode("utf-8", errors="replace") if isinstance(f, bytes) else str(f)
            raw_paths.append(p)
            if os.path.isfile(p):
                file_list.append(p)
            elif os.path.isdir(p):
                for sub_f in sorted(os.listdir(p)):
                    if sub_f.lower().endswith((".wav", ".flac", ".aiff", ".mp3", ".ogg", ".m4a", ".aac")):
                        file_list.append(os.path.join(p, sub_f))

        if not file_list and not any(os.path.isdir(p) for p in raw_paths):
            return

        current_tab = self.notebook.index(self.notebook.select())

        # If on Tab 0 (Auto Session Master), or if folder dropped on tab 0
        if current_tab == 0:
            if len(raw_paths) == 1 and os.path.isdir(raw_paths[0]):
                self._load_auto_session_source(raw_paths[0])
            else:
                self._load_auto_session_source(file_list if file_list else raw_paths)
            return

        # If on Tab 1 (Multitrack Transcriber)
        audio_files = [f for f in file_list if f.lower().endswith((".wav", ".flac", ".aiff", ".mp3", ".ogg", ".m4a", ".aac"))]
        if audio_files:
            self._load_transcribe_dropped_files(audio_files)
        else:
            messagebox.showwarning("No Audio Files", "None of the dropped files are supported audio formats.")

    # Tab 1 Handlers
    def _select_multi_files(self):
        files = filedialog.askopenfilenames(
            title="Select Audio Tracks (e.g. 6 track files)",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff"), ("All Files", "*.*")]
        )
        if files:
            self._load_track_files(list(files))

    def _select_multichannel_file(self):
        f = filedialog.askopenfilename(
            title="Select Multichannel WAV File",
            filetypes=[("Audio Files (*.wav, *.flac)", "*.wav *.flac"), ("All Files", "*.*")]
        )
        if f:
            self._load_multichannel_file(f)

    def _load_track_files(self, file_paths: List[str]):
        file_paths.sort(key=natural_sort_key)
        self.track_files = file_paths
        self.is_multichannel = False
        self._refresh_track_list()

    def _load_multichannel_file(self, file_path: str):
        self.track_files = [file_path]
        self.is_multichannel = True
        self._refresh_track_list()

    def _clear_tracks(self):
        self.track_files = []
        self.is_multichannel = False
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.target_combo["values"] = ["Track 5"]
        self.target_var.set("Track 5")
        self.status_lbl_var.set("Ready. Load your audio tracks above.")
        self.out_path_var.set("")

    def _refresh_track_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.track_files:
            return

        total_tracks = 0
        try:
            src = AudioSource(self.track_files)
            total_tracks = src.n_tracks
            sr = src.samplerate
            dur_str = format_time(src.duration_sec)

            if src.is_multichannel_file:
                for t in range(total_tracks):
                    self.tree.insert("", "end", values=(
                        f"Track {t+1}",
                        f"{os.path.basename(self.track_files[0])} [Ch {t+1}]",
                        dur_str,
                        f"{sr} Hz",
                        "1",
                        "Reference Mic"
                    ))
            else:
                for t, path in enumerate(self.track_files):
                    fname = os.path.basename(path)
                    try:
                        info = sf.info(path)
                        ch_str = str(info.channels)
                    except Exception:
                        ch_str = "1"
                    self.tree.insert("", "end", values=(
                        f"Track {t+1}",
                        fname,
                        dur_str,
                        f"{sr} Hz",
                        ch_str,
                        "Reference Mic"
                    ))
            src.close()
        except Exception as e:
            messagebox.showerror("Error Loading Tracks", f"Failed to load audio tracks:\n{e}")
            return

        track_opts = [f"Track {i+1}" for i in range(total_tracks)]
        self.target_combo["values"] = track_opts
        if total_tracks >= 5:
            self.target_combo.current(4)
            self.target_var.set("Track 5")
        elif total_tracks > 0:
            self.target_combo.current(0)
            self.target_var.set("Track 1")

        self._on_target_changed()
        self._auto_fill_output_path()
        self.status_lbl_var.set(f"Loaded {total_tracks} tracks ({dur_str} total duration). Ready.")

    def _get_target_idx(self) -> int:
        val = self.target_var.get()
        m = re.search(r'\d+', val)
        return int(m.group(0)) - 1 if m else 4

    def _auto_fill_output_path(self):
        if not self.track_files:
            return
        t_num = self._get_target_idx() + 1
        first_file = self.track_files[0]
        dir_name = os.path.dirname(os.path.abspath(first_file))
        base = os.path.splitext(os.path.basename(first_file))[0]
        clean_name = f"{base}_track{t_num}_cleaned.wav"
        self.out_path_var.set(os.path.join(dir_name, clean_name))

    def _on_target_changed(self, event=None):
        t_idx = self._get_target_idx()
        children = self.tree.get_children()
        for i, item in enumerate(children):
            vals = list(self.tree.item(item, "values"))
            vals[5] = "🎯 TARGET BLEED TRACK" if i == t_idx else f"Reference Mic {i+1}"
            self.tree.item(item, values=vals)
        self._auto_fill_output_path()

    def _on_tree_select(self, event=None):
        selected = self.tree.selection()
        if selected:
            idx = self.tree.index(selected[0])
            values = self.target_combo["values"]
            if 0 <= idx < len(values):
                self.target_combo.current(idx)
                self.target_var.set(values[idx])
                self._on_target_changed()

    def _select_output_file(self):
        initial = self.out_path_var.get()
        init_dir = os.path.dirname(initial) if initial else os.path.expanduser("~")
        init_file = os.path.basename(initial) if initial else "cleaned_track5.wav"
        out_f = filedialog.asksaveasfilename(
            title="Select Output Audio Destination",
            initialdir=init_dir,
            initialfile=init_file,
            defaultextension=".wav",
            filetypes=[("WAV Audio (*.wav)", "*.wav"), ("All Files", "*.*")]
        )
        if out_f:
            self.out_path_var.set(out_f)

    # Slider handlers for Tab 1
    def _on_cutoff_slider(self, val):
        self.cutoff_val_var.set(f"{int(float(val))}%")

    def _on_floor_slider(self, val):
        v = float(val)
        self.floor_val_var.set("-inf dB" if v <= -98 else f"{int(v)} dB")

    def _on_sens_slider(self, val):
        self.sens_val_var.set(f"{float(val):.2f}x")

    def _on_hold_slider(self, val):
        self.hold_val_var.set(f"{int(float(val))} ms")

    def _on_rel_slider(self, val):
        self.rel_val_var.set(f"{int(float(val))} ms")

    def _on_preset_changed(self, event=None):
        preset = self.preset_var.get()
        if "Standard Studio" in preset:
            self.cutoff_slider.set(48)
            self.floor_slider.set(-60)
            self.sens_slider.set(1.0)
            self.hold_slider.set(160)
            self.rel_slider.set(85)
        elif "Aggressive Bleed Cut" in preset:
            self.cutoff_slider.set(38)
            self.floor_slider.set(-99)
            self.sens_slider.set(0.95)
            self.hold_slider.set(200)
            self.rel_slider.set(100)
        elif "Whisper / Sensitive Vocals" in preset:
            self.cutoff_slider.set(55)
            self.floor_slider.set(-54)
            self.sens_slider.set(1.35)
            self.hold_slider.set(280)
            self.rel_slider.set(160)
        elif "Complete Digital Silence" in preset:
            self.cutoff_slider.set(45)
            self.floor_slider.set(-99)
            self.sens_slider.set(1.0)
            self.hold_slider.set(220)
            self.rel_slider.set(120)
        elif "Subtle Room Ambience" in preset:
            self.cutoff_slider.set(48)
            self.floor_slider.set(-36)
            self.sens_slider.set(1.0)
            self.hold_slider.set(220)
            self.rel_slider.set(120)
        self._on_cutoff_slider(self.cutoff_slider.get())
        self._on_floor_slider(self.floor_slider.get())
        self._on_sens_slider(self.sens_slider.get())
        self._on_hold_slider(self.hold_slider.get())
        self._on_rel_slider(self.rel_slider.get())

    # Tab 2 Handlers (Laptop Mic Restorer)
    def _select_laptop_file(self):
        f = filedialog.askopenfilename(
            title="Select Laptop Track Audio File",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff *.ogg"), ("All Files", "*.*")]
        )
        if f:
            self._load_laptop_file(f)

    def _import_track6_from_tab1(self):
        if len(self.track_files) >= 6:
            self._load_laptop_file(self.track_files[5])
        elif len(self.track_files) > 0:
            t6 = None
            for p in self.track_files:
                if '06' in os.path.basename(p) or '_6' in os.path.basename(p):
                    t6 = p
                    break
            if t6:
                self._load_laptop_file(t6)
            else:
                self._load_laptop_file(self.track_files[-1])
        else:
            messagebox.showinfo("No Tracks in Tab 1", "Please load track files in Tab 1 first or click 'Select Track File...'")

    def _load_laptop_file(self, file_path: str):
        if not os.path.isfile(file_path):
            return
        self.laptop_file = file_path
        try:
            info = sf.info(file_path)
            dur_str = format_time(info.duration)
            fname = os.path.basename(file_path)
            self.laptop_info_var.set(f"Loaded: {fname} | Duration: {dur_str} | {info.samplerate} Hz | {info.channels}ch | Ready")
            self.laptop_info_lbl.configure(background="#e8f5e9", foreground="#2e7d32")
        except Exception as e:
            self.laptop_info_var.set(f"Loaded: {os.path.basename(file_path)} (Unable to read header: {e})")

        dir_name = os.path.dirname(os.path.abspath(file_path))
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_f = os.path.join(dir_name, f"{base}_cleaned_enhanced.wav")
        self.laptop_out_path_var.set(out_f)

    def _select_laptop_output_file(self):
        initial = self.laptop_out_path_var.get()
        init_dir = os.path.dirname(initial) if initial else os.path.expanduser("~")
        init_file = os.path.basename(initial) if initial else "laptop_cleaned_enhanced.wav"
        out_f = filedialog.asksaveasfilename(
            title="Select Restored Output Audio Destination",
            initialdir=init_dir,
            initialfile=init_file,
            defaultextension=".wav",
            filetypes=[("WAV Audio (*.wav)", "*.wav"), ("All Files", "*.*")]
        )
        if out_f:
            self.laptop_out_path_var.set(out_f)

    def _on_laptop_preset_changed(self, event=None):
        preset = self.laptop_preset_var.get()
        if "Studio Balanced" in preset:
            self.l_denoise_slider.set(1.6)
            self.l_floor_slider.set(-40)
            self.l_warmth_slider.set(2.5)
            self.l_desk_slider.set(-5.0)
            self.l_hollow_slider.set(-3.5)
            self.l_presence_slider.set(4.5)
        elif "Aggressive Fan Whine Removal" in preset:
            self.l_denoise_slider.set(2.2)
            self.l_floor_slider.set(-50)
            self.l_warmth_slider.set(3.0)
            self.l_desk_slider.set(-6.0)
            self.l_hollow_slider.set(-4.5)
            self.l_presence_slider.set(5.0)
        elif "Subtle / Quiet Room" in preset:
            self.l_denoise_slider.set(1.1)
            self.l_floor_slider.set(-30)
            self.l_warmth_slider.set(1.5)
            self.l_desk_slider.set(-3.0)
            self.l_hollow_slider.set(-2.0)
            self.l_presence_slider.set(3.0)
        elif "Broadcast Punch" in preset:
            self.l_denoise_slider.set(1.8)
            self.l_floor_slider.set(-45)
            self.l_warmth_slider.set(3.5)
            self.l_desk_slider.set(-5.5)
            self.l_hollow_slider.set(-4.0)
            self.l_presence_slider.set(6.0)

        self.l_denoise_var.set(f"{float(self.l_denoise_slider.get()):.2f}x")
        self.l_floor_var.set(f"{int(float(self.l_floor_slider.get()))} dB")
        self.l_warmth_var.set(f"+{float(self.l_warmth_slider.get()):.1f} dB")
        self.l_desk_var.set(f"{float(self.l_desk_slider.get()):.1f} dB")
        self.l_hollow_var.set(f"{float(self.l_hollow_slider.get()):.1f} dB")
        self.l_presence_var.set(f"+{float(self.l_presence_slider.get()):.1f} dB")

    # Tab 3 Handlers (Studio Voice Restorer & Digital Silence)
    def _select_studio_file(self):
        f = filedialog.askopenfilename(
            title="Select Dialogue Track Audio File",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff *.ogg"), ("All Files", "*.*")]
        )
        if f:
            self._load_studio_file(f)

    def _import_target_to_studio(self):
        if not self.track_files:
            messagebox.showinfo("No Tracks Loaded", "No tracks currently loaded in Tab 1.")
            return
        target_0idx = self._get_target_idx()
        if 0 <= target_0idx < len(self.track_files):
            self._load_studio_file(self.track_files[target_0idx])
        else:
            self._load_studio_file(self.track_files[0])

    def _clear_studio_file(self):
        self.studio_file = None
        self.studio_file_lbl.configure(text="No file selected.")
        self.studio_out_path_var.set("")

    def _load_studio_file(self, file_path: str):
        if not os.path.isfile(file_path):
            return
        self.studio_file = file_path
        file_name = os.path.basename(file_path)
        dir_name = os.path.dirname(os.path.abspath(file_path))
        try:
            info = sf.info(file_path)
            dur_str = format_time(info.duration)
            self.studio_file_lbl.configure(
                text=f"Loaded: {file_name} | Duration: {dur_str} | {info.samplerate} Hz | {info.channels}ch | Ready"
            )
        except Exception as e:
            self.studio_file_lbl.configure(text=f"Loaded: {file_name}")

        base = os.path.splitext(file_name)[0]
        ext = ".mp3" if "MP3" in self.studio_export_fmt_var.get() else ".wav"
        out_f = os.path.join(dir_name, f"{base}_restored{ext}")
        self.studio_out_path_var.set(out_f)

    def _select_studio_output_file(self):
        initial = self.studio_out_path_var.get()
        init_dir = os.path.dirname(initial) if initial else os.path.expanduser("~")
        init_file = os.path.basename(initial) if initial else "dialogue_restored.mp3"
        is_mp3 = "MP3" in self.studio_export_fmt_var.get()
        def_ext = ".mp3" if is_mp3 else ".wav"
        filetypes = [("MP3 Audio (*.mp3)", "*.mp3"), ("WAV Audio (*.wav)", "*.wav"), ("All Files", "*.*")] if is_mp3 else [("WAV Audio (*.wav)", "*.wav"), ("MP3 Audio (*.mp3)", "*.mp3"), ("All Files", "*.*")]
        out_f = filedialog.asksaveasfilename(
            title="Select Restored Output Audio Destination",
            initialdir=init_dir,
            initialfile=init_file,
            defaultextension=def_ext,
            filetypes=filetypes
        )
        if out_f:
            self.studio_out_path_var.set(out_f)

    def _on_studio_format_changed(self, event=None):
        current_out = self.studio_out_path_var.get().strip()
        if not current_out:
            return
        base, _ = os.path.splitext(current_out)
        if "MP3" in self.studio_export_fmt_var.get():
            self.studio_out_path_var.set(f"{base}.mp3")
        else:
            self.studio_out_path_var.set(f"{base}.wav")

    # Tab 0 Handlers (1-Click Session Auto-Master)
    def _select_auto_folder(self):
        folder = filedialog.askdirectory(title="Select Multitrack Session Folder")
        if folder:
            self._load_auto_session_source(folder)

    def _select_auto_files(self):
        files = filedialog.askopenfilenames(
            title="Select Session Audio Tracks (e.g. 6 track files)",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff *.ogg *.m4a *.aac"), ("All Files", "*.*")]
        )
        if files:
            self._load_auto_session_source(list(files))

    def _select_single_slot_file(self, slot_num: int):
        f = filedialog.askopenfilename(
            title=f"Select Audio File for Slot {slot_num}",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff *.ogg *.m4a *.aac"), ("All Files", "*.*")]
        )
        if f:
            self.auto_slots[slot_num]["path_var"].set(f)
            if not self.auto_out_dir_var.get().strip():
                dir_name = os.path.dirname(os.path.abspath(f))
                self.auto_out_dir_var.set(os.path.join(dir_name, "Mastered"))

    def _select_auto_out_dir(self):
        folder = filedialog.askdirectory(title="Select Mastered Output Directory")
        if folder:
            self.auto_out_dir_var.set(folder)

    def _clear_auto_slots(self):
        for slot_num in range(1, 7):
            self.auto_slots[slot_num]["path_var"].set("")
        self.auto_out_dir_var.set("")
        self.auto_src_info_var.set("Slots cleared. Drop a session folder or select track files above.")

    def _load_auto_session_source(self, source):
        mode = self.campaign_mode_var.get()
        detected = auto_detect_session_tracks(source, mode=mode)
        if not detected:
            messagebox.showwarning("No Tracks Detected", f"No audio tracks (1-6) were automatically identified in the selected folder/files for mode '{mode}'.")
            return

        cfg = get_campaign_config(mode)
        for slot_num in range(1, 7):
            if slot_num in detected:
                self.auto_slots[slot_num]["path_var"].set(detected[slot_num])
                def_prof = cfg["slots"].get(slot_num, {}).get("profile_id", "t3_reference")
                self.auto_slots[slot_num]["prof_var"].set(profile_id_to_str(def_prof))

        first_path = next(iter(detected.values()))
        session_dir = os.path.dirname(os.path.abspath(first_path))
        self.auto_session_dir = session_dir
        self.auto_out_dir_var.set(os.path.join(session_dir, "Mastered"))

        self.auto_src_info_var.set(f"✅ Loaded {len(detected)} tracks from {os.path.basename(session_dir)} | Ready to Master")
        self._log(f"[+] Auto-detected {len(detected)} tracks from {os.path.basename(session_dir)}:")
        for s_idx, p in sorted(detected.items()):
            self._log(f"    Slot {s_idx}: {os.path.basename(p)} -> {self.auto_slots[s_idx]['prof_var'].get()}")
        self.status_lbl_var.set(f"Ready: {len(detected)} session tracks loaded and assigned. Click 'Process Entire Session' to master.")

    # Processing Dispatch
    def _start_processing(self, preview_sec: Optional[float] = None):
        tab_idx = self.notebook.index(self.notebook.select())
        if tab_idx == 0:
            self._start_auto_session_processing(preview_sec)
        else:
            self._start_transcription_processing(preview_sec)

    def _start_auto_session_processing(self, preview_sec: Optional[float] = None):
        slots_data = {}
        for slot_num in range(1, 7):
            p = self.auto_slots[slot_num]["path_var"].get().strip()
            prof_str = self.auto_slots[slot_num]["prof_var"].get().strip()
            prof_id = profile_str_to_id(prof_str)
            if p and os.path.isfile(p) and prof_id != "skip":
                slots_data[slot_num] = {"path": p, "profile": prof_id}

        if not slots_data:
            messagebox.showwarning("No Tracks to Process", "Please load session tracks first into the slots above.")
            return

        out_dir = self.auto_out_dir_var.get().strip()
        if not out_dir:
            first_p = next(iter(slots_data.values()))["path"]
            out_dir = os.path.join(os.path.dirname(os.path.abspath(first_p)), "Mastered")
            self.auto_out_dir_var.set(out_dir)

        export_fmt = "mp3" if "MP3" in self.auto_export_fmt_var.get() else "wav"
        apply_silence = bool(self.auto_silence_gate_var.get())
        apply_ai_denoise = bool(self.auto_ai_denoise_var.get())
        strength_str = self.auto_ai_strength_var.get()
        m = re.search(r'(\d+)%', strength_str)
        ai_strength = float(m.group(1)) / 100.0 if m else 1.0
        apply_norm = bool(self.auto_normalize_var.get())
        t_lufs_str = self.auto_target_lufs_var.get().split()[0]
        try:
            target_lufs = float(t_lufs_str)
        except ValueError:
            target_lufs = -18.0

        self.is_processing = True
        self.cancel_requested = False
        self.last_output_file = out_dir

        self.btn_preview.configure(state="disabled")
        self.btn_process.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_open_folder.pack_forget()
        self.progress_var.set(0.0)

        self.worker_thread = threading.Thread(
            target=self._worker_run_auto_session,
            args=(slots_data, out_dir, export_fmt, apply_silence, apply_ai_denoise, ai_strength, apply_norm, target_lufs, preview_sec),
            daemon=True
        )
        self.worker_thread.start()

    def _worker_run_auto_session(self, slots_data, out_dir, export_fmt, apply_silence, apply_ai_denoise, ai_strength, apply_norm, target_lufs, preview_sec):
        try:
            def cb(track_idx, num_tracks, track_name, track_pct, total_pct, speed, eta_str, status_msg):
                self.msg_queue.put(("auto_progress", (track_idx, num_tracks, track_name, track_pct, total_pct, speed, eta_str, status_msg)))

            results = process_automated_session(
                slots=slots_data,
                output_dir=out_dir,
                export_format=export_fmt,
                apply_silence_gate=apply_silence,
                apply_ai_denoise=apply_ai_denoise,
                ai_denoise_strength=ai_strength,
                apply_speech_normalization=apply_norm,
                target_lufs=target_lufs,
                preview_sec=preview_sec,
                progress_callback=cb,
                cancel_check=lambda: self.cancel_requested,
                log_func=self._log
            )
            if self.cancel_requested:
                self.msg_queue.put(("finish", (False, "Session processing cancelled.")))
            else:
                mode_str = "Preview" if preview_sec else "Session Mastering"
                self.msg_queue.put(("finish", (True, f"✅ 1-Click {mode_str} Complete!\nAll {len(results)} tracks saved in:\n{out_dir}")))
        except Exception as e:
            self._log(f"[-] FATAL ERROR in session mastering: {e}")
            self.msg_queue.put(("finish", (False, f"Session mastering failed: {e}")))

    def _start_multitrack_processing(self, preview_sec: Optional[float] = None):
        if not self.track_files:
            messagebox.showwarning("No Tracks Loaded", "Please load your audio track files first in Tab 1.")
            return

        out_path = self.out_path_var.get().strip()
        if not out_path:
            messagebox.showwarning("No Output Path", "Please specify an output file path.")
            return

        target_0idx = self._get_target_idx()
        match_cutoff = float(self.cutoff_slider.get()) / 100.0
        floor_db = float(self.floor_slider.get())
        sensitivity = float(self.sens_slider.get())
        hold_ms = float(self.hold_slider.get())
        rel_ms = float(self.rel_slider.get())
        track_files = list(self.track_files)
        enhance = bool(self.enhance_var.get())

        self.is_processing = True
        self.cancel_requested = False
        self.last_output_file = out_path

        self.btn_preview.configure(state="disabled")
        self.btn_process.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_open_folder.pack_forget()
        self.progress_var.set(0.0)

        self.worker_thread = threading.Thread(
            target=self._worker_run_multitrack,
            args=(preview_sec, target_0idx, out_path, match_cutoff, floor_db, sensitivity, hold_ms, rel_ms, track_files, enhance),
            daemon=True
        )
        self.worker_thread.start()

    def _start_laptop_processing(self, preview_sec: Optional[float] = None):
        if not self.laptop_file or not os.path.isfile(self.laptop_file):
            messagebox.showwarning("No File Selected", "Please select a laptop audio file first in Tab 2.")
            return

        out_path = self.laptop_out_path_var.get().strip()
        if not out_path:
            messagebox.showwarning("No Output Path", "Please specify an output file path for the restored track.")
            return

        alpha = float(self.l_denoise_slider.get())
        floor_db = float(self.l_floor_slider.get())
        warmth_db = float(self.l_warmth_slider.get())
        desk_cut_db = float(self.l_desk_slider.get())
        hollow_cut_db = float(self.l_hollow_slider.get())
        presence_db = float(self.l_presence_slider.get())
        apply_ai_denoise = bool(self.laptop_ai_denoise_var.get())
        strength_str = self.laptop_ai_strength_var.get()
        m = re.search(r'(\d+)%', strength_str)
        ai_strength = float(m.group(1)) / 100.0 if m else 1.0
        laptop_file = self.laptop_file

        self.is_processing = True
        self.cancel_requested = False
        self.last_output_file = out_path

        self.btn_preview.configure(state="disabled")
        self.btn_process.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_open_folder.pack_forget()
        self.progress_var.set(0.0)

        self.worker_thread = threading.Thread(
            target=self._worker_run_laptop,
            args=(laptop_file, out_path, preview_sec, alpha, floor_db, desk_cut_db, hollow_cut_db, warmth_db, presence_db, apply_ai_denoise, ai_strength),
            daemon=True
        )
        self.worker_thread.start()

    def _start_studio_processing(self, preview_sec: Optional[float] = None):
        if not self.studio_file or not os.path.isfile(self.studio_file):
            messagebox.showwarning("No File Selected", "Please select a dialogue audio file first in Tab 3.")
            return

        out_path = self.studio_out_path_var.get().strip()
        if not out_path:
            messagebox.showwarning("No Output Path", "Please specify an output file path.")
            return

        profile_str = self.studio_profile_var.get()
        if "Track 1" in profile_str:
            profile = "t1_room_echo"
        elif "Track 2" in profile_str:
            profile = "t2_muffled"
        elif "Track 4" in profile_str:
            profile = "t4_megaphone"
        elif "Track 5" in profile_str:
            profile = "t5_headset"
        elif "ai" in profile_str.lower() or "rnnoise" in profile_str.lower():
            profile = "ai_rnnoise"
        else:
            profile = "universal_clean"

        apply_silence = bool(self.studio_silence_gate_var.get())
        apply_ai_denoise = bool(self.studio_ai_denoise_var.get())
        strength_str = self.studio_ai_strength_var.get()
        m = re.search(r'(\d+)%', strength_str)
        ai_strength = float(m.group(1)) / 100.0 if m else 1.0

        open_thresh_db = float(self.studio_thresh_slider.get())
        hold_ms = float(self.studio_hold_slider.get())
        
        floor_str = self.studio_floor_var.get()
        if "Digital Silence" in floor_str:
            floor_db = -120.0
        elif "-60" in floor_str:
            floor_db = -60.0
        else:
            floor_db = -50.0

        export_fmt = self.studio_export_fmt_var.get()
        studio_file = self.studio_file

        self.is_processing = True
        self.cancel_requested = False
        self.last_output_file = out_path

        self.btn_preview.configure(state="disabled")
        self.btn_process.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_open_folder.pack_forget()
        self.progress_var.set(0.0)

        self.worker_thread = threading.Thread(
            target=self._worker_run_studio,
            args=(studio_file, out_path, profile, apply_silence, apply_ai_denoise, ai_strength, floor_db, open_thresh_db, hold_ms, preview_sec, export_fmt),
            daemon=True
        )
        self.worker_thread.start()

    def _cancel_processing(self):
        if self.is_processing:
            self.cancel_requested = True
            self.status_lbl_var.set("Cancelling processing...")
            self.btn_cancel.configure(state="disabled")

    # Worker: Multitrack Bleed Gate
    def _worker_run_multitrack(
        self,
        preview_sec: Optional[float],
        target_0idx: int,
        out_path: str,
        match_cutoff: float,
        floor_db: float,
        sensitivity: float,
        hold_ms: float,
        rel_ms: float,
        track_files: List[str],
        enhance: bool = False
    ):
        attack_ms = 10.0
        mode_label = f"PREVIEW ({int(preview_sec)}s)" if preview_sec else "FULL SESSION"
        self._log(f"\n{'='*60}\nSTARTING MULTITRACK BLEED GATE ({mode_label})...\n{'='*60}")
        
        try:
            src = AudioSource(track_files)
            total_sec = src.duration_sec
            if preview_sec and preview_sec > 0:
                total_sec = min(total_sec, preview_sec)

            self._log(f"[*] Total duration: {format_time(src.duration_sec)} | Sample rate: {src.samplerate} Hz | Tracks: {src.n_tracks}")
            self._log(f"[*] Target bleed track: Track {target_0idx + 1}")
            self._log(f"[*] Broadcast Vocal Tuning: {'ENABLED' if enhance else 'DISABLED'}")
            self._log(f"[*] Calibrating dynamic speech thresholds & noise floors...")

            calibrator = BleedCalibrator(target_idx=target_0idx, n_tracks=src.n_tracks, sr=src.samplerate)
            ref_thresh, tgt_thresh, noise_floor, speech_level = calibrator.calibrate(
                src,
                max_scan_sec=min(120.0, total_sec),
                sensitivity=sensitivity
            )

            self._log(f"    - Target Noise Floor:       {20*math.log10(max(1e-9, noise_floor)):.1f} dBFS")
            self._log(f"    - Target Speech Level:      {20*math.log10(max(1e-9, speech_level)):.1f} dBFS")
            self._log(f"    - Target Speech Threshold:  {20*math.log10(max(1e-9, tgt_thresh)):.1f} dBFS")
            self._log(f"[*] Bleed Match Cutoff: {int(match_cutoff*100)}% | Mute Floor: {floor_db:.0f} dB")

            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                
            out_handle = sf.SoundFile(
                out_path,
                mode='w',
                samplerate=src.samplerate,
                channels=1,
                subtype='PCM_24' if src.samplerate <= 48000 else 'FLOAT'
            )

            gate = StreamingMultitrackBleedGate(
                sr=src.samplerate,
                target_idx=target_0idx,
                n_tracks=src.n_tracks,
                match_cutoff=match_cutoff,
                sensitivity=sensitivity,
                floor_db=floor_db,
                attack_ms=attack_ms,
                hold_ms=hold_ms,
                release_ms=rel_ms,
                ref_speech_thresh=ref_thresh,
                tgt_speech_thresh=tgt_thresh,
                enhance=enhance,
            )

            chunk_samples = int(30.0 * src.samplerate)
            total_processed_samples = 0
            max_process_samples = int(total_sec * src.samplerate)
            total_unmuted_samples = 0
            t_start = time.time()

            while total_processed_samples < max_process_samples and not self.cancel_requested:
                to_read = min(chunk_samples, max_process_samples - total_processed_samples)
                raw_chunk = src.read_chunk(to_read)
                if raw_chunk is None or raw_chunk.shape[1] == 0:
                    break

                processed_chunk, stats = gate.process_chunk(raw_chunk)
                out_handle.write(processed_chunk)

                n_chunk = len(processed_chunk)
                total_processed_samples += n_chunk
                total_unmuted_samples += int(stats["unmuted_ratio"] * n_chunk)

                curr_sec = total_processed_samples / src.samplerate
                pct = (total_processed_samples / max_process_samples) * 100.0
                elapsed = time.time() - t_start
                speed = curr_sec / max(0.001, elapsed)
                eta_sec = (max_process_samples - total_processed_samples) / (src.samplerate * max(0.1, speed))

                self.msg_queue.put(("progress", (pct, curr_sec, total_sec, speed, eta_sec)))

            out_handle.close()
            src.close()

            if self.cancel_requested:
                self._log("[!] Bleed gate processing was cancelled by user.")
                self.msg_queue.put(("finish", (False, "Processing cancelled.")))
            else:
                total_elapsed = time.time() - t_start
                proc_dur = total_processed_samples / src.samplerate
                unmuted_dur = total_unmuted_samples / src.samplerate
                muted_dur = proc_dur - unmuted_dur
                muted_pct = (muted_dur / max(0.1, proc_dur)) * 100.0

                self._log("\n" + "="*60)
                self._log("🎉 BLEED GATE PROCESSING FINISHED SUCCESSFULLY!")
                self._log(f"  • Processed Audio:  {format_time(proc_dur)} ({proc_dur:.1f}s)")
                self._log(f"  • Target Active:    {format_time(unmuted_dur)} ({100.0 - muted_pct:.1f}%)")
                self._log(f"  • Target Muted:     {format_time(muted_dur)} ({muted_pct:.1f}%)")
                self._log(f"  • Processing Speed: {proc_dur / max(0.001, total_elapsed):.1f}x real-time (Took {total_elapsed:.1f}s)")
                self._log(f"  • Cleaned Output:   {out_path}")
                self._log("="*60)
                self.msg_queue.put(("finish", (True, f"Done in {total_elapsed:.1f}s! Muted {muted_pct:.1f}% bleed.")))

        except Exception as e:
            self._log(f"[-] ERROR occurred during processing: {e}")
            self.msg_queue.put(("finish", (False, f"Error: {e}")))

    # Worker: Laptop Mic Restorer
    def _worker_run_laptop(
        self,
        input_path: str,
        output_path: str,
        preview_sec: Optional[float],
        alpha: float,
        floor_db: float,
        desk_cut_db: float,
        hollow_cut_db: float,
        warmth_db: float,
        presence_db: float,
        apply_ai_denoise: bool,
        ai_strength: float
    ):
        mode_label = f"PREVIEW ({int(preview_sec)}s)" if preview_sec else "FULL RECORDING"
        self._log(f"\n{'='*60}\nSTARTING LAPTOP MIC RESTORATION ({mode_label})...\n{'='*60}")

        def progress_cb(pct, curr_sec, total_sec, speed, eta_sec):
            self.msg_queue.put(("progress", (pct, curr_sec, total_sec, speed, eta_sec)))

        def cancel_check():
            return self.cancel_requested

        t_start = time.time()
        try:
            success = process_laptop_mic_file(
                input_path=input_path,
                output_path=output_path,
                preview_sec=preview_sec,
                alpha=alpha,
                floor_db=floor_db,
                desk_cut_db=desk_cut_db,
                hollow_cut_db=hollow_cut_db,
                warmth_db=warmth_db,
                presence_db=presence_db,
                apply_ai_denoise=apply_ai_denoise,
                ai_denoise_strength=ai_strength,
                progress_callback=progress_cb,
                cancel_check=cancel_check,
                log_func=self._log
            )

            total_elapsed = time.time() - t_start
            if success:
                self._log("\n" + "="*60)
                self._log("🎉 LAPTOP MIC RESTORATION FINISHED SUCCESSFULLY!")
                self._log(f"  • Wall Clock Time:  {total_elapsed:.1f}s")
                self._log(f"  • Restored Output:  {output_path}")
                self._log("="*60)
                self.msg_queue.put(("finish", (True, f"Laptop Mic Restoration complete in {total_elapsed:.1f}s!")))
            else:
                self._log("[!] Laptop mic processing was cancelled by user.")
                self.msg_queue.put(("finish", (False, "Processing cancelled.")))

        except Exception as e:
            self._log(f"[-] ERROR occurred during laptop restoration: {e}")
            self.msg_queue.put(("finish", (False, f"Error: {e}")))

    # Worker: Studio Voice Restorer & Digital Silence Gate
    def _worker_run_studio(
        self,
        in_file: str,
        out_path: str,
        profile: str,
        apply_silence: bool,
        apply_ai_denoise: bool,
        ai_strength: float,
        floor_db: float,
        open_thresh_db: float,
        hold_ms: float,
        preview_sec: Optional[float],
        export_fmt: str
    ):
        mode_label = f"PREVIEW ({int(preview_sec)}s)" if preview_sec else "FULL RECORDING"
        self._log(f"\n{'='*60}\nSTARTING STUDIO DIALOGUE PROCESSING ({mode_label})...\n{'='*60}")

        def progress_cb(pct, speed, eta, elapsed):
            self.msg_queue.put(("progress", (pct, elapsed, 0.0, speed, eta)))

        def cancel_check():
            return self.cancel_requested

        t_start = time.time()
        try:
            mp3_bitrate = "192k" if "192" in export_fmt else "320k"
            success = process_vocal_restoration_file(
                input_path=in_file,
                output_path=out_path,
                profile=profile,
                apply_silence_gate=apply_silence,
                apply_ai_denoise=apply_ai_denoise,
                ai_denoise_strength=ai_strength,
                silence_floor_db=floor_db,
                open_thresh_db=open_thresh_db,
                hold_ms=hold_ms,
                preview_sec=preview_sec,
                export_format=export_fmt,
                mp3_bitrate=mp3_bitrate,
                progress_callback=progress_cb,
                cancel_check=cancel_check,
                log_func=self._log
            )

            total_elapsed = time.time() - t_start
            if success:
                self._log("\n" + "="*60)
                self._log("🎉 DIALOGUE RESTORATION & SILENCE GATE COMPLETED!")
                self._log(f"  • Wall Clock Time:  {total_elapsed:.1f}s")
                self._log(f"  • Restored Output:  {out_path}")
                self._log("="*60)
                self.msg_queue.put(("finish", (True, f"Dialogue processing complete in {total_elapsed:.1f}s!")))
            else:
                self._log("[!] Dialogue processing was cancelled by user.")
                self.msg_queue.put(("finish", (False, "Processing cancelled.")))

        except Exception as e:
            self._log(f"[-] ERROR occurred during dialogue processing: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.msg_queue.put(("finish", (False, f"Error: {e}")))

    # =========================================================================
    # Tab 4 Handlers & Worker (Multitrack Transcriber & Script Merger)
    # =========================================================================

    def _import_master_slots_to_transcriber(self):
        master_dir = self.auto_out_dir_var.get().strip()
        imported = 0
        mode = self.campaign_mode_var.get()
        cfg = get_campaign_config(mode)

        for slot_num in range(1, 7):
            s_cfg = cfg["slots"].get(slot_num, {})
            chosen_path = ""
            # First check if Mastered directory exists with output files
            if master_dir and os.path.isdir(master_dir):
                aliases = s_cfg.get("aliases", ())
                for fname in sorted(os.listdir(master_dir)):
                    fl = fname.lower()
                    if fl.endswith((".mp3", ".wav")) and (
                        f"track_{slot_num}" in fl or
                        f"track_a0{slot_num}" in fl or
                        f"track_{slot_num:02d}" in fl or
                        f"slot_{slot_num}" in fl or
                        f"track{slot_num}" in fl or
                        any(alias in fl for alias in aliases)
                    ):
                        chosen_path = os.path.join(master_dir, fname)
                        break

            # If not found in Mastered folder, fall back to input file in slot
            if not chosen_path:
                in_p = self.auto_slots[slot_num]["path_var"].get().strip()
                if in_p and os.path.isfile(in_p):
                    chosen_path = in_p

            if chosen_path and os.path.isfile(chosen_path):
                self.transcribe_slots[slot_num]["path_var"].set(chosen_path)
                self.transcribe_slots[slot_num]["active_var"].set(s_cfg.get("active", True))
                imported += 1

        if imported > 0:
            self._update_transcribe_output_path()
            self._log(f"[+] Imported {imported} tracks into Transcriber Matrix ({cfg['name']}).")
            self.transcribe_info_lbl.configure(text=f"✅ Imported {imported} tracks from Session Master! Ready to transcribe.")
        else:
            messagebox.showinfo("No Session Tracks Found", "No audio tracks were found in Tab 1 (Auto-Master) to import.")

    def _send_mastered_to_transcriber(self):
        self._import_master_slots_to_transcriber()
        self.notebook.select(self.tab_transcribe)

    def _select_transcribe_files(self):
        files = filedialog.askopenfilenames(
            title="Select Track Audio Files (e.g. 6 speaker tracks)",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff *.ogg *.m4a *.aac"), ("All Files", "*.*")]
        )
        if files:
            mode = self.campaign_mode_var.get()
            cfg = get_campaign_config(mode)
            detected = auto_detect_session_tracks(list(files), mode=mode)
            if detected:
                for idx in range(1, 7):
                    s_cfg = cfg["slots"].get(idx, {})
                    if idx in detected:
                        self.transcribe_slots[idx]["path_var"].set(detected[idx])
                        self.transcribe_slots[idx]["active_var"].set(s_cfg.get("active", True))
                    else:
                        self.transcribe_slots[idx]["path_var"].set("")
                        self.transcribe_slots[idx]["active_var"].set(False)
            else:
                sorted_files = sorted(files, key=natural_sort_key)
                for idx in range(1, 7):
                    s_cfg = cfg["slots"].get(idx, {})
                    if idx <= len(sorted_files):
                        self.transcribe_slots[idx]["path_var"].set(sorted_files[idx - 1])
                        self.transcribe_slots[idx]["active_var"].set(s_cfg.get("active", True))
                    else:
                        self.transcribe_slots[idx]["path_var"].set("")
                        self.transcribe_slots[idx]["active_var"].set(False)
            self._update_transcribe_output_path()

    def _select_transcribe_single_file(self, slot_num: int):
        f = filedialog.askopenfilename(
            title=f"Select Audio Track for Slot {slot_num}",
            filetypes=[("Audio Files (*.mp3, *.wav, *.flac)", "*.mp3 *.wav *.flac *.aiff *.ogg *.m4a *.aac"), ("All Files", "*.*")]
        )
        if f:
            self.transcribe_slots[slot_num]["path_var"].set(f)
            self.transcribe_slots[slot_num]["active_var"].set(True)
            self._update_transcribe_output_path()

    def _clear_transcribe_slots(self):
        for slot_num in range(1, 7):
            self.transcribe_slots[slot_num]["path_var"].set("")
        self.transcribe_out_file_var.set("")
        self.transcribe_info_lbl.configure(text="Slots cleared. Import session tracks or select audio files above.")

    def _load_transcribe_dropped_files(self, files: List[str]):
        mode = self.campaign_mode_var.get()
        cfg = get_campaign_config(mode)
        detected = auto_detect_session_tracks(files, mode=mode)
        if detected:
            for idx in range(1, 7):
                s_cfg = cfg["slots"].get(idx, {})
                if idx in detected:
                    self.transcribe_slots[idx]["path_var"].set(detected[idx])
                    self.transcribe_slots[idx]["active_var"].set(s_cfg.get("active", True))
                else:
                    self.transcribe_slots[idx]["path_var"].set("")
                    self.transcribe_slots[idx]["active_var"].set(False)
        else:
            sorted_files = sorted(files, key=natural_sort_key)
            for idx in range(1, 7):
                s_cfg = cfg["slots"].get(idx, {})
                if idx <= len(sorted_files):
                    self.transcribe_slots[idx]["path_var"].set(sorted_files[idx - 1])
                    self.transcribe_slots[idx]["active_var"].set(s_cfg.get("active", True))
                else:
                    self.transcribe_slots[idx]["path_var"].set("")
                    self.transcribe_slots[idx]["active_var"].set(False)
        self._update_transcribe_output_path()

    def _update_transcribe_output_path(self):
        if not self.transcribe_out_file_var.get().strip():
            for slot_num in range(1, 7):
                p = self.transcribe_slots[slot_num]["path_var"].get().strip()
                if p and os.path.isfile(p):
                    parent = os.path.dirname(os.path.abspath(p))
                    self.transcribe_out_file_var.set(os.path.join(parent, "Session_Transcript.txt"))
                    break

    def _select_transcribe_output_file(self):
        init_p = self.transcribe_out_file_var.get().strip()
        init_dir = os.path.dirname(init_p) if init_p else os.path.expanduser("~")
        init_f = os.path.basename(init_p) if init_p else "Session_Transcript.txt"
        f = filedialog.asksaveasfilename(
            title="Select Output Transcript Script Destination",
            initialdir=init_dir,
            initialfile=init_f,
            defaultextension=".txt",
            filetypes=[("Text Transcript (*.txt)", "*.txt"), ("All Files", "*.*")]
        )
        if f:
            self.transcribe_out_file_var.set(f)

    def _open_transcript_file(self):
        target = self.last_transcript_file or self.transcribe_out_file_var.get().strip()
        if target and os.path.isfile(target):
            try:
                os.startfile(target)
            except Exception:
                subprocess.Popen(["notepad.exe", target])

    def _clean_existing_transcript_file(self):
        f = filedialog.askopenfilename(
            title="Select Existing Transcript to Correct Campaign Vocabulary",
            filetypes=[("Text Transcript (*.txt)", "*.txt"), ("SubRip Subtitle (*.srt)", "*.srt"), ("All Files", "*.*")]
        )
        if not f:
            return
        try:
            out_p = correct_transcript_file(f)
            self._log(f"[+] Successfully corrected campaign names and Star Wars lore vocabulary in: {out_p}")
            messagebox.showinfo(
                "Vocabulary Cleaned",
                f"Transcript successfully cleaned and corrected!\n\nSaved to:\n{out_p}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to correct transcript: {e}")

    def _start_transcription_processing(self, preview_sec: Optional[float] = None):
        if not HAS_TRANSCRIBER:
            messagebox.showerror(
                "Engine Missing",
                "faster-whisper is not installed or failed to import.\n\nPlease run:\npip install faster-whisper"
            )
            return

        tracks_config = []
        for slot_num in range(1, 7):
            if not self.transcribe_slots[slot_num]["active_var"].get():
                continue
            p = self.transcribe_slots[slot_num]["path_var"].get().strip()
            spk = self.transcribe_slots[slot_num]["name_var"].get().strip() or f"Speaker {slot_num}"
            if p and os.path.isfile(p):
                tracks_config.append({"path": p, "speaker": spk, "active": True})

        if not tracks_config:
            messagebox.showwarning("No Tracks Loaded", "Please select or import at least one audio track to transcribe.")
            return

        out_path = self.transcribe_out_file_var.get().strip()
        if not out_path:
            first_p = tracks_config[0]["path"]
            out_path = os.path.join(os.path.dirname(os.path.abspath(first_p)), "Session_Transcript.txt")
            self.transcribe_out_file_var.set(out_path)

        # Parse model size
        model_raw = self.transcribe_model_var.get().strip().split()[0]
        # Parse gap safely with regex
        gap_match = re.search(r"(\d+(?:\.\d+)?)", self.transcribe_gap_var.get())
        merge_gap = float(gap_match.group(1)) if gap_match else 1.2

        prompt = self.transcribe_prompt_var.get().strip() or None
        export_srt_flag = bool(self.transcribe_export_srt_var.get())
        export_json_flag = bool(self.transcribe_export_json_var.get())
        enable_moderation = bool(self.transcribe_moderation_var.get())
        auto_mute_flags = bool(self.transcribe_automute_var.get())
        flagged_terms = self.transcribe_flagged_terms_var.get().strip() or None

        self.is_processing = True
        self.cancel_requested = False
        self.last_output_file = out_path

        self.btn_preview.configure(state="disabled")
        self.btn_process.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_open_folder.pack_forget()
        self.btn_open_transcript.pack_forget()
        self.progress_var.set(0.0)

        device_str = self.transcribe_device_var.get()

        self.worker_thread = threading.Thread(
            target=self._worker_run_transcription,
            args=(tracks_config, out_path, model_raw, prompt, merge_gap, export_srt_flag, export_json_flag, preview_sec, enable_moderation, auto_mute_flags, flagged_terms, device_str),
            daemon=True
        )
        self.worker_thread.start()

    def _worker_run_transcription(
        self,
        tracks_config: List[Dict[str, Any]],
        out_path: str,
        model_name: str,
        prompt: Optional[str],
        merge_gap: float,
        export_srt_flag: bool,
        export_json_flag: bool,
        preview_sec: Optional[float],
        enable_moderation: bool = True,
        auto_mute_flags: bool = False,
        flagged_terms: Optional[str] = None,
        device_str: Optional[str] = None
    ):
        mode_label = f"PREVIEW ({int(preview_sec)}s)" if preview_sec else "FULL MULTITRACK SESSION"
        self._log(f"\n{'='*60}\nSTARTING MULTITRACK TRANSCRIPTION ({mode_label})...\n{'='*60}")
        self._log(f"  • Whisper Model:    {model_name}")
        self._log(f"  • Compute Engine:   {device_str or 'Auto Acceleration'}")
        self._log(f"  • Speaker Tracks:   {len(tracks_config)}")
        for idx, tr in enumerate(tracks_config, start=1):
            self._log(f"    [{idx}] {tr['speaker']:18s} -> {os.path.basename(tr['path'])}")
        if prompt:
            self._log(f"  • Context Prompt:   \"{prompt}\"")
        self._log(f"  • Merge Gap:        {merge_gap:.1f}s")
        if enable_moderation:
            self._log(f"  • YouTube Safety:   ACTIVE (Flagging sensitive terms; Auto-Mute: {'YES' if auto_mute_flags else 'NO'})")
        self._log(f"  • Target Output:    {out_path}")

        t_start = time.time()
        temp_files_to_clean = []

        try:
            # Handle preview by creating sliced audio snippets
            active_inputs = []
            if preview_sec:
                self._log(f"[+] Slicing first {int(preview_sec)}s for fast preview...")
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), ".preview_snippets")
                os.makedirs(temp_dir, exist_ok=True)
                for tr in tracks_config:
                    orig_p = tr["path"]
                    spk = tr["speaker"]
                    snip_p = os.path.join(temp_dir, f"prev_{os.path.splitext(os.path.basename(orig_p))[0]}.wav")
                    try:
                        info = sf.info(orig_p)
                        stop_samp = min(int(preview_sec * info.samplerate), info.frames)
                        data, rate = sf.read(orig_p, stop=stop_samp)
                        sf.write(snip_p, data, rate)
                        active_inputs.append({"path": snip_p, "speaker": spk, "active": True})
                        temp_files_to_clean.append(snip_p)
                    except Exception as e:
                        self._log(f"[-] Warning: Failed to slice preview for {spk}: {e}. Using original.")
                        active_inputs.append(tr)
            else:
                active_inputs = tracks_config

            dev_clean = (device_str or "").lower()
            if "cpu" in dev_clean and "auto" not in dev_clean:
                device_choice = "cpu"
                compute_choice = "int8"
            elif "gpu" in dev_clean and "auto" not in dev_clean:
                device_choice = "cuda"
                compute_choice = "float16"
            else:
                device_choice = "auto"
                compute_choice = "default"

            transcriber = MultitrackTranscriber(
                model_size=model_name,
                device=device_choice,
                compute_type=compute_choice,
            )

            def on_progress(pdata):
                if pdata.get("type") == "status_note":
                    self.msg_queue.put(("log", f">> {pdata['msg']}"))
                    return
                if pdata.get("type") == "moderation_flag":
                    flag = pdata["flag"]
                    self.msg_queue.put(("log", f"  🚨 [POLICY FLAG] {flag['speaker']} at {flag['timecode']}: \"{flag['context']}\" (Term: '{flag['word']}')"))
                    return
                self.msg_queue.put((
                    "transcribe_progress",
                    (
                        pdata["track_idx"],
                        pdata["total_tracks"],
                        pdata["speaker"],
                        pdata["current_sec"],
                        pdata["total_sec"],
                        pdata["progress_pct"],
                        pdata["latest_segment"]["text"],
                    )
                ))
                seg = pdata["latest_segment"]
                ts = format_ts_transcribe(seg["start"], "txt")
                flag_warn = " ⚠️ [FLAGGED]" if seg.get("flagged") else ""
                self.msg_queue.put(("log", f"  {ts} [{seg['speaker']}]{flag_warn}: \"{seg['text']}\""))

            def on_status(msg):
                self.msg_queue.put(("log", f">> {msg}"))

            result = transcriber.transcribe_multitrack_session(
                tracks_config=active_inputs,
                initial_prompt=prompt,
                pause_merge_sec=merge_gap,
                enable_moderation=enable_moderation,
                flagged_terms=flagged_terms,
                auto_mute_flags=auto_mute_flags,
                progress_callback=on_progress,
                status_callback=on_status,
                cancel_check=lambda: self.cancel_requested
            )

            if self.cancel_requested:
                self.msg_queue.put(("transcribe_finish", (False, "Transcription cancelled by user.", None)))
                return

            # Save .txt script
            if not out_path.lower().endswith(".txt"):
                out_path += ".txt"
            session_title = os.path.splitext(os.path.basename(out_path))[0].replace("_", " ")
            export_txt(result, out_path, title=session_title)
            self._log(f"[+] Exported formatted dialogue script to: {out_path}")

            base_no_ext = os.path.splitext(out_path)[0]
            out_dir = os.path.dirname(os.path.abspath(out_path))

            if export_srt_flag:
                out_srt = base_no_ext + ".srt"
                export_srt(result, out_srt)
                self._log(f"[+] Exported SubRip subtitles to: {out_srt}")

            if export_json_flag:
                out_json = base_no_ext + ".json"
                export_json(result, out_json)
                self._log(f"[+] Exported JSON metadata to: {out_json}")

            # Export YouTube Safety Report and DaVinci/Premiere Timeline Markers
            mod_flags = result.get("moderation_flags", [])
            if enable_moderation:
                rep_path = os.path.join(out_dir, "Session_YouTube_Safety_Report.txt")
                export_moderation_report(mod_flags, rep_path)
                self._log(f"[+] Exported YouTube Content Safety Report to: {rep_path}")

                if mod_flags:
                    davinci_csv = os.path.join(out_dir, "Session_Timeline_Markers_DaVinci.csv")
                    export_timeline_markers_davinci(mod_flags, davinci_csv)
                    self._log(f"[+] Exported DaVinci Resolve Timeline Markers to: {davinci_csv}")

                    premiere_csv = os.path.join(out_dir, "Session_Timeline_Markers_Premiere.csv")
                    export_timeline_markers_premiere(mod_flags, premiere_csv)
                    self._log(f"[+] Exported Premiere Pro Timeline Markers to: {premiere_csv}")

            total_elapsed = time.time() - t_start
            total_blocks = result.get("total_dialogue_blocks", 0)
            stats = result.get("speaker_stats", {})
            total_words = sum(s.get("word_count", 0) for s in stats.values())

            self._log("\n" + "=" * 60)
            self._log("🎉 MULTITRACK TRANSCRIPTION COMPLETED!")
            self._log(f"  • Processing Time:  {total_elapsed:.1f}s")
            self._log(f"  • Dialogue Turns:   {total_blocks}")
            self._log(f"  • Total Words:      {total_words}")
            self._log(f"  • Script Document:  {out_path}")
            self._log("=" * 60)

            msg = f"Done in {total_elapsed:.1f}s! Decoded {total_words} words across {total_blocks} turns."
            self.msg_queue.put(("transcribe_finish", (True, msg, out_path)))

        except Exception as e:
            self._log(f"[-] FATAL ERROR in multitrack transcription: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.msg_queue.put(("transcribe_finish", (False, f"Transcription error: {e}", None)))

        finally:
            for tf in temp_files_to_clean:
                try:
                    if os.path.isfile(tf):
                        os.remove(tf)
                except Exception:
                    pass

    def _finish_transcription(self, success: bool, msg: str, out_file: Optional[str]):
        self.is_processing = False
        self.btn_preview.configure(state="normal")
        self.btn_process.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.status_lbl_var.set(msg)

        if success and out_file:
            self.progress_var.set(100.0)
            self.last_transcript_file = out_file
            self.last_output_file = out_file
            self.btn_open_transcript.pack(side="right", padx=(6, 0))
            self.btn_open_folder.pack(side="right", padx=(6, 0))
            messagebox.showinfo(
                "Transcription Complete",
                f"Multitrack transcript successfully created!\n\n"
                f"File: {os.path.basename(out_file)}\n\n"
                f"Path: {out_file}\n\n"
                f"Click 'Open Transcript (Notepad)' to review the script."
            )

    def _update_progress_ui(self, pct: float, curr_sec: float, total_sec: float, speed: float, eta_sec: float):
        self.progress_var.set(pct)
        status = (
            f"Processing: {pct:5.1f}% | {format_time(curr_sec)} / {format_time(total_sec)} | "
            f"Speed: {speed:4.1f}x | ETA: {format_time(eta_sec)}"
        )
        self.status_lbl_var.set(status)

    def _finish_processing(self, success: bool, msg: str):
        self.is_processing = False
        self.btn_preview.configure(state="normal")
        self.btn_process.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.status_lbl_var.set(msg)

        if success:
            self.progress_var.set(100.0)
            self.btn_open_folder.pack(side="right", padx=(6, 0))
            messagebox.showinfo("Processing Complete", f"Audio processing finished successfully!\n\nOutput saved to:\n{self.last_output_file}")

    def _open_output_folder(self):
        if self.last_output_file and os.path.exists(self.last_output_file):
            try:
                subprocess.run(["explorer", f"/select,{os.path.abspath(self.last_output_file)}"])
            except Exception:
                folder = os.path.dirname(os.path.abspath(self.last_output_file))
                os.startfile(folder)


def run_studio_webview(start_port: int = 8765):
    """
    Launches the modern hardware-accelerated Studio Workstation in Edge App Mode
    backed by the local FastAPI & WebSocket streaming server.
    """
    import socket
    import tempfile
    import webbrowser
    import urllib.request
    import uvicorn
    from web_server import app

    def find_free_port(base_port: int = 8765) -> int:
        for p in range(base_port, base_port + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
        return base_port

    port = find_free_port(start_port)
    server_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to become ready
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{server_url}/api/state", timeout=0.5) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.1)

    opened_native = False
    try:
        import webview
        window = webview.create_window(
            title="Multitrack Audio Studio & Transcriber",
            url=server_url,
            width=1240,
            height=920,
            min_size=(960, 680),
            background_color="#090b10",
            easy_drag=False
        )
        webview.start(debug=False)
        opened_native = True
    except Exception as e:
        print(f"[-] Native webview window failed ({e}), checking fallback...")

    if not opened_native:
        candidate_browsers = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        browser_exe = next((p for p in candidate_browsers if os.path.isfile(p)), None)

        if browser_exe:
            temp_profile = os.path.join(tempfile.gettempdir(), "multitrack_studio_webview_profile")
            cmd = [
                browser_exe,
                f"--app={server_url}",
                f"--user-data-dir={temp_profile}",
                "--window-size=1200,920",
                "--disable-extensions",
                "--disable-features=Translate",
                "--no-first-run",
                "--no-default-browser-check"
            ]
            try:
                proc = subprocess.Popen(cmd)
                proc.wait()
            except Exception as e:
                print(f"[-] App mode window failed ({e}), opening default browser...")
                webbrowser.open(server_url)
                try:
                    while server_thread.is_alive():
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        else:
            webbrowser.open(server_url)
            try:
                while server_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

    server.should_exit = True
    server_thread.join(timeout=2.0)


def run_classic_gui():
    """Fallback Tkinter user interface."""
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    root = tk.Tk()
    app = BleedGateGUI(root)
    root.mainloop()


def main():
    if "--classic" in sys.argv:
        sys.argv.remove("--classic")
        run_classic_gui()
    else:
        try:
            run_studio_webview()
        except Exception as e:
            import traceback
            log_file = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "studio_error.log")
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[{time.ctime()}] Webview launcher error:\n" + traceback.format_exc())
            except Exception:
                pass
            print(f"[-] Webview launcher error ({e}). Launching classic GUI...")
            run_classic_gui()


if __name__ == "__main__":
    if len(sys.argv) > 1 and ("--inputs" in sys.argv or "--multichannel" in sys.argv or "--denoise-laptop" in sys.argv or "-h" in sys.argv or "--help" in sys.argv):
        from multitrack_bleed_gate import build_cli_parser, process_multitrack_session
        parser = build_cli_parser()
        args = parser.parse_args()
        if args.denoise_laptop or args.laptop_file:
            in_file = args.laptop_file or (args.inputs[0] if args.inputs else None)
            if not in_file:
                print("[-] Error: Specify input file.")
                sys.exit(1)
            out_file = args.output or f"{os.path.splitext(in_file)[0]}_cleaned_enhanced.wav"
            process_laptop_mic_file(in_file, out_file, preview_sec=args.preview, chunk_sec=args.chunk_sec)
        else:
            process_multitrack_session(args)
    else:
        main()
