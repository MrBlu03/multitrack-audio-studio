"""
Multitrack Speech-to-Text Transcription & Chronological Script Merger
====================================================================
Designed for Tabletop RPG sessions, roundtable podcasts, and multitrack voice recordings.

Key Features:
- 100% Ground-Truth Speaker Attribution (per isolated mic track)
- High-Performance CTranslate2 / faster-whisper engine (int8 CPU optimized)
- Silero VAD acceleration (automatically skips bleed-gated digital silence)
- Chronological alignment & conversational paragraph grouping
- Speaker speaking stats (duration, word counts, talk-time percentage)
- Formats: Clean Script (.txt), Subtitles (.srt), Structured Data (.json)
"""

import os
import sys
import time
import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
import numpy as np
import soundfile as sf

def setup_cuda_dll_paths() -> List[str]:
    """
    Ensure NVIDIA CUDA 12 / cuBLAS / cuDNN runtime DLLs can be located by CTranslate2 on Windows.
    Adds package directories, _MEIPASS, and executable directory to os.add_dll_directory and PATH.
    """
    candidates = []
    
    # 1. Executable and current directory (portable external DLLs next to MultitrackAudioStudio.exe)
    for base in [os.path.dirname(os.path.abspath(sys.argv[0])), os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
        candidates.append(base)
        candidates.append(os.path.join(base, "nvidia", "cublas", "bin"))
        candidates.append(os.path.join(base, "nvidia", "cudnn", "bin"))

    # 2. PyInstaller temporary extraction folder
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            candidates.append(meipass)
            candidates.append(os.path.join(meipass, "nvidia", "cublas", "bin"))
            candidates.append(os.path.join(meipass, "nvidia", "cudnn", "bin"))

    # 3. Python site-packages nvidia wheels
    try:
        import site
        for sp in site.getsitepackages():
            candidates.append(os.path.join(sp, "nvidia", "cublas", "bin"))
            candidates.append(os.path.join(sp, "nvidia", "cudnn", "bin"))
            candidates.append(os.path.join(sp, "nvidia", "cuda_nvrtc", "bin"))
    except Exception:
        pass

    candidates.append(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin"))
    candidates.append(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"))

    # 4. Standard CUDA toolkit paths
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))

    added = []
    for cand in candidates:
        if os.path.isdir(cand) and cand not in added:
            try:
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(cand)
                os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
                added.append(cand)
            except Exception:
                pass
    return added

# Automatically register CUDA DLL directories on module load
setup_cuda_dll_paths()

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Process-level model cache: avoids ~15-30s reload cost on repeated transcription runs.
# Key: (model_size_or_path, device, compute_type) → WhisperModel instance
_WHISPER_MODEL_CACHE: Dict[tuple, Any] = {}

def _get_cached_whisper_model(model_path: str, device: str, compute_type: str,
                               cpu_threads: int, download_root: Optional[str]) -> Any:
    """Return a cached WhisperModel, loading it only on first call per unique config."""
    cache_key = (model_path, device, compute_type)
    if cache_key not in _WHISPER_MODEL_CACHE:
        _WHISPER_MODEL_CACHE[cache_key] = WhisperModel(
            model_size_or_path=model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
        )
    return _WHISPER_MODEL_CACHE[cache_key]


def format_timestamp(seconds: float, format_type: str = "txt") -> str:
    """Format floating seconds into HH:MM:SS or HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999

    if format_type == "srt":
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"
    elif format_type == "txt":
        if hrs > 0:
            return f"[{hrs:02d}:{mins:02d}:{secs:02d}]"
        else:
            return f"[{mins:02d}:{secs:02d}]"
    else:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


DEFAULT_FLAGGED_TERMS = ["nigger", "nigga", "faggot", "retard"]


def compile_flagged_regex(terms_str_or_list: Union[str, List[str], None] = None) -> re.Pattern:
    """Compiles a whole-word case-insensitive regex for sensitive content and slur detection."""
    if terms_str_or_list is None:
        terms = DEFAULT_FLAGGED_TERMS
    elif isinstance(terms_str_or_list, str):
        terms = [t.strip().lower() for t in terms_str_or_list.replace(",", " ").split() if t.strip()]
        if not terms:
            terms = DEFAULT_FLAGGED_TERMS
    else:
        terms = [t.strip().lower() for t in terms_str_or_list if t.strip()]
        if not terms:
            terms = DEFAULT_FLAGGED_TERMS

    escaped = []
    for t in terms:
        if "nigg" in t:
            escaped.append(r"nigg(er|a|az|ers|as|ah)?")
        elif "fag" in t:
            escaped.append(r"fagg?ot(s)?")
        elif "retard" in t:
            escaped.append(r"retard(ed|s)?")
        else:
            escaped.append(re.escape(t))
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def seconds_to_timecode(seconds: float, fps: float = 30.0) -> str:
    """Converts seconds to standard SMPTE broadcast timecode (HH:MM:SS:FF)."""
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    frames = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def seconds_to_hms_ms(seconds: float) -> str:
    """Converts seconds to HH:MM:SS.mmm format."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    total_secs = int(seconds)
    secs = total_secs % 60
    mins = (total_secs // 60) % 60
    hours = total_secs // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


def correct_campaign_vocabulary(text: str) -> str:
    """
    Context-aware phonetics and spelling normalization for Star Wars 5e TTRPG campaign.
    Corrects character names, factions, languages, and gaming terms.
    """
    if not text:
        return text

    # Caelen (Blu's human ex-Sith apprentice)
    text = re.sub(r'\b(cailin|kaylin|kalen|caitlin|caitlyn|kaylee|kaylen|kaelen|caelan|kailen|cailen|kaelin|kalem)\b', 'Caelen', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(cailin|kaylin|kalen|caitlin|caitlyn)\'s\b', "Caelen's", text, flags=re.IGNORECASE)

    # Cratebreaker (Robin's Ewok)
    text = re.sub(r'\b(craybrooker|kramer|crepe\s*breaker|cray\s*picker|crate\s*breaker|crane\s*breaker|kraybrick|craybrick|kraidraker|cray\s*breaker)\b', 'Cratebreaker', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcrate\s+breaker\b', 'Cratebreaker', text, flags=re.IGNORECASE)

    # Belial (Timmy's Devaronian bounty hunter)
    text = re.sub(r'\b(baelios|bale\s*i|bavarius|bilal|biel|belia|billy\s*all|billion|biddle)\b', 'Belial', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(baelios|belial)\'s\b', "Belial's", text, flags=re.IGNORECASE)
    text = re.sub(r'\bbelial\b', 'Belial', text, flags=re.IGNORECASE)

    # It'Mir (Tino's Jawa)
    text = re.sub(r'\b(ydmir|idmir|ytmer|eat\s*mir|it\s*mir|it\'s\s*mir)\b', "It'Mir", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(it\s*me|it\'s\s*me|it\s*may)\b(?=\s+(?:that\'s|it\'s|your|take|turn|what|is|was|can|you|wake|follow|hide|sneak|jump|run|steal|shoot|knife|dagger|roll|did|didn\'t|got|trip|fell|give))', "It'Mir", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(with|to|and|ask|tell|grab|levitate|see|holding|carry|carrying|behind|follow|take|against)\s+(?:it\s*me|it\'s\s*me|it\s*may)\b', lambda m: m.group(1) + " It'Mir", text, flags=re.IGNORECASE)
    text = re.sub(r'\bIt\s*mir\b', "It'Mir", text, flags=re.IGNORECASE)

    # Salova (Mathew's Wookiee Jedi Master)
    text = re.sub(r'\b(slova|solova|the\s+lova|sulova)\b', 'Salova', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsalova\b', 'Salova', text, flags=re.IGNORECASE)

    # Kyrix & Antagonists
    text = re.sub(r'\b(kirix|cirrix|cirrus)\b', 'Kyrix', text, flags=re.IGNORECASE)
    text = re.sub(r'\blord\s+kyrix\b', 'Lord Kyrix', text, flags=re.IGNORECASE)

    # Languages & Species
    text = re.sub(r'\b(shriuk|shreewook|shitty\s*wook|shreewookies|shreewook)\b', 'Shyriiwook', text, flags=re.IGNORECASE)
    text = re.sub(r'\bspeaking\s+in\s+chili\b', 'speaking in Shyriiwook', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwookie\b', 'Wookiee', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwookies\b', 'Wookiees', text, flags=re.IGNORECASE)
    text = re.sub(r'\bjawar\b', 'Jawa', text, flags=re.IGNORECASE)
    text = re.sub(r'\bjawars\b', 'Jawas', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(new\s*walk|new\s*walk\s*is|iwaki\'s|iwakis)\b', 'Ewok', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcourse\s*sounds\b', 'Coruscant', text, flags=re.IGNORECASE)

    # Mechanics & Weapons (SW5E)
    text = re.sub(r'\bviper\s*dagger\b', 'vibrodagger', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(not|net)\s*20\b', 'Nat 20', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(not|net)\s*1\b', 'Nat 1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(roll\s+ad\s*20|ad\s*20|a\s+320)\b', 'a d20', text, flags=re.IGNORECASE)

    # =========================================================================
    # CYBERPUNK RED CAMPAIGN VOCABULARY
    # =========================================================================
    # QBall (Robin's character)
    text = re.sub(r'\b(q\s*ball|qball|cue\s*ball|cube\s*all)\b', 'QBall', text, flags=re.IGNORECASE)

    # Ryu (Marc's character)
    text = re.sub(r'\b(ryu|riyu|ryou|reyou)\b', 'Ryu', text, flags=re.IGNORECASE)

    # Umbra (Rati's character)
    text = re.sub(r'\b(umbra|ombre|umber)\b(?=\s+(?:you|is|was|can|what|take|roll|did|turn|shoot|hack|slice|run|move|check|said|says))', 'Umbra', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(with|to|and|ask|tell|grab|see|behind|follow|take|against|for)\s+(?:umbra|ombre|umber)\b', lambda m: m.group(1) + " Umbra", text, flags=re.IGNORECASE)
    text = re.sub(r'\bumbra\b', 'Umbra', text, flags=re.IGNORECASE)

    # Magnus (Timmy's character)
    text = re.sub(r'\b(magnus|magness|magnes)\b', 'Magnus', text, flags=re.IGNORECASE)

    # Factions, Mega-Corps & World Lore
    text = re.sub(r'\b(arasaca|ara\s*saka|araska)\b', 'Arasaka', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(mili\s*tech|milli\s*tech|miltech)\b', 'Militech', text, flags=re.IGNORECASE)
    text = re.sub(r'\bknight\s*city\b', 'Night City', text, flags=re.IGNORECASE)
    text = re.sub(r'\btrauma\s*team\b', 'Trauma Team', text, flags=re.IGNORECASE)

    # Tech, Slang & Mechanics
    text = re.sub(r'\bedge\s*runner\b', 'edgerunner', text, flags=re.IGNORECASE)
    text = re.sub(r'\bedge\s*runners\b', 'edgerunners', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(eddy\'s|eddys|eddie\'s)\b', 'eddies', text, flags=re.IGNORECASE)
    text = re.sub(r'\beuro\s*dollars\b', 'eurodollars', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcyber\s*ware\b', 'cyberware', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcyber\s*deck\b', 'cyberdeck', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbrain\s*dance\b', 'braindance', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(sandy\s*vistan|sandi\s*vistan|santa\s*vistan)\b', 'sandevistan', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(roll\s+ad\s*10|roll\s+a\s+d\s*10)\b', 'roll a d10', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ad\s*10|a\s+d\s*10)\b', 'a d10', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(crit\s*success|critical\s*success)\b', 'Critical Success', text, flags=re.IGNORECASE)

    return text


def split_segment_into_utterances(
    seg: Any,
    pause_thresh: float = 0.50,
    sent_pause_thresh: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Splits a coarse Faster-Whisper segment into natural conversational utterances
    based on word-level pause gaps, punctuation boundaries, and conversational pacing.
    Prevents large multi-sentence blocks from monopolizing the timeline and
    preserves authentic conversational back-and-forth between multiple speakers.
    """
    words = getattr(seg, "words", None)
    if not words:
        clean_text = correct_campaign_vocabulary(seg.text.strip())
        return [{
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": round(seg.end - seg.start, 2),
            "text": clean_text,
            "words": [],
        }] if clean_text else []

    utterances: List[Dict[str, Any]] = []
    curr_words = [words[0]]

    for i in range(1, len(words)):
        prev_w = words[i - 1]
        curr_w = words[i]
        gap = curr_w.start - prev_w.end
        curr_dur = prev_w.end - curr_words[0].start

        prev_text = prev_w.word.strip()
        ends_with_terminal_punct = bool(re.search(r'[.?!]+$', prev_text))
        ends_with_comma = bool(re.search(r'[,;:\-]+$', prev_text))

        is_split = False
        if ends_with_terminal_punct and gap >= sent_pause_thresh:
            is_split = True
        elif ends_with_comma and gap >= 0.30:
            is_split = True
        elif gap >= pause_thresh:
            is_split = True
        elif curr_dur >= 6.0 and (ends_with_terminal_punct or (ends_with_comma and gap >= 0.15) or gap >= 0.35):
            is_split = True
        elif curr_dur >= 9.0 and gap >= 0.20:
            is_split = True


        if is_split:
            raw_t = " ".join(w.word.strip() for w in curr_words).strip()
            norm_t = correct_campaign_vocabulary(raw_t)
            if norm_t:
                utterances.append({
                    "start": round(curr_words[0].start, 2),
                    "end": round(curr_words[-1].end, 2),
                    "duration": round(curr_words[-1].end - curr_words[0].start, 2),
                    "text": norm_t,
                    "words": curr_words,
                })
            curr_words = [curr_w]
        else:
            curr_words.append(curr_w)

    if curr_words:
        raw_t = " ".join(w.word.strip() for w in curr_words).strip()
        norm_t = correct_campaign_vocabulary(raw_t)
        if norm_t:
            utterances.append({
                "start": round(curr_words[0].start, 2),
                "end": round(curr_words[-1].end, 2),
                "duration": round(curr_words[-1].end - curr_words[0].start, 2),
                "text": norm_t,
                "words": curr_words,
            })

    return utterances


class MultitrackTranscriber:
    """
    Multitrack transcription engine powered by faster-whisper with CTranslate2.
    """

    SUPPORTED_MODELS = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v3", "large-v3-turbo"]

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: Optional[int] = None,
        download_root: Optional[str] = None,
    ):
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. Please run: pip install faster-whisper"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads or min(8, max(1, (os.cpu_count() or 4) - 1))
        self.download_root = download_root
        self._model: Optional[WhisperModel] = None

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Loads and caches the WhisperModel in memory."""
        if self._model is not None:
            return

        # Check if an offline 'models' folder exists next to the executable, in the working directory, or on Google Drive
        exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        search_dirs = [
            os.path.join(exe_dir, "models"),
            os.path.join(os.getcwd(), "models"),
            r"G:\My Drive\Multitrack Audio Studio\models",
        ]

        target_model: str = self.model_size
        effective_root = self.download_root
        is_local_found = False

        for m_dir in search_dirs:
            if not os.path.isdir(m_dir):
                continue

            # 1. Direct folder: models/<model_size>/ (e.g. models/small.en/model.bin)
            direct_cand = os.path.join(m_dir, self.model_size)
            if os.path.isdir(direct_cand) and os.path.isfile(os.path.join(direct_cand, "model.bin")):
                target_model = direct_cand
                is_local_found = True
                break

            # 2. Direct model.bin inside models/ itself
            if os.path.isfile(os.path.join(m_dir, "model.bin")):
                target_model = m_dir
                is_local_found = True
                break

            # 3. HuggingFace repo folder: models/models--Systran--faster-whisper-<model_size>
            repo_cand = os.path.join(m_dir, f"models--Systran--faster-whisper-{self.model_size}")
            if os.path.isdir(repo_cand):
                snaps = os.path.join(repo_cand, "snapshots")
                if os.path.isdir(snaps):
                    for sub in os.listdir(snaps):
                        sub_p = os.path.join(snaps, sub)
                        if os.path.isdir(sub_p) and os.path.isfile(os.path.join(sub_p, "model.bin")):
                            target_model = sub_p
                            is_local_found = True
                            break
                if is_local_found:
                    break

            if effective_root is None:
                effective_root = m_dir

        if progress_callback:
            if is_local_found:
                progress_callback(f"Loading Whisper model '{self.model_size}' from local models folder...")
            else:
                progress_callback(f"Loading Whisper model '{self.model_size}' (auto-downloading if running on this PC for the first time)...")

        # Auto-detect CUDA if device is "auto"
        effective_device = self.device
        effective_compute = self.compute_type

        if effective_device == "auto":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    effective_device = "cuda"
                    effective_compute = "float16" if effective_compute == "default" else effective_compute
                else:
                    effective_device = "cpu"
                    effective_compute = "int8" if effective_compute == "default" else effective_compute
            except Exception:
                effective_device = "cpu"
                effective_compute = "int8" if effective_compute == "default" else effective_compute
        elif effective_compute == "default":
            effective_compute = "float16" if effective_device == "cuda" else "int8"

        self._target_model = target_model
        self._effective_root = effective_root

        try:
            self._model = _get_cached_whisper_model(
                target_model, effective_device, effective_compute,
                self.cpu_threads, effective_root,
            )

            # If loaded on CUDA, execute immediate warmup to verify cuBLAS / cuDNN DLLs
            if effective_device == "cuda":
                try:
                    dummy_audio = np.zeros(1600, dtype=np.float32)
                    _ = list(self._model.transcribe(dummy_audio, beam_size=1)[0])
                    if progress_callback:
                        progress_callback(f"Whisper model '{self.model_size}' verified working on CUDA ({effective_compute}).")
                except Exception as cuda_err:
                    if progress_callback:
                        progress_callback(f"[-] CUDA execution unavailable ({cuda_err}). Automatically falling back to CPU (int8 multi-core)...")
                    effective_device = "cpu"
                    effective_compute = "int8"
                    self._model = _get_cached_whisper_model(
                        target_model, "cpu", "int8", self.cpu_threads, effective_root,
                    )
                    if progress_callback:
                        progress_callback(f"Whisper model '{self.model_size}' successfully loaded on CPU (int8 multi-core).")
            else:
                if progress_callback:
                    progress_callback(f"Whisper model '{self.model_size}' successfully loaded on {effective_device.upper()} ({effective_compute}).")
        except Exception as e:
            if effective_device == "cuda":
                if progress_callback:
                    progress_callback(f"[-] CUDA initialization failed ({e}). Falling back to CPU (int8 multi-core)...")
                self._model = _get_cached_whisper_model(
                    target_model, "cpu", "int8", self.cpu_threads, effective_root,
                )
                effective_device = "cpu"
                effective_compute = "int8"
                if progress_callback:
                    progress_callback(f"Whisper model '{self.model_size}' successfully loaded on CPU (int8 multi-core).")
            else:
                raise e

        self.device = effective_device
        self.compute_type = effective_compute

    def transcribe_track(
        self,
        audio_path: str,
        speaker_name: str,
        track_idx: int = 1,
        total_tracks: int = 1,
        initial_prompt: Optional[str] = None,
        language: str = "en",
        vad_filter: bool = True,
        min_silence_duration_ms: int = 400,
        enable_moderation: bool = True,
        flagged_terms: Union[str, List[str], None] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transcribes a single audio track, attributing every spoken segment to speaker_name.
        Uses Silero VAD to skip digital silence instantly.
        If enable_moderation is True, tracks word-level timestamps to detect and flag sensitive slurs.
        """
        if self._model is None:
            self.load_model()

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        vad_params = dict(
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=200,
        ) if vad_filter else None

        flagged_regex = compile_flagged_regex(flagged_terms) if enable_moderation else None
        use_word_ts = True

        def run_transcription():
            return self._model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                beam_size=5,
                word_timestamps=use_word_ts,
                condition_on_previous_text=False,  # Prevents repetitive hallucination loops
            )

        try:
            segments_gen, info = run_transcription()
            first_seg = None
            try:
                first_seg = next(segments_gen, None)
            except Exception as e:
                err_str = str(e).lower()
                if "cublas" in err_str or "cuda" in err_str or "cudnn" in err_str or "driver" in err_str:
                    if progress_callback:
                        progress_callback({
                            "type": "status_note",
                            "msg": f"CUDA execution unavailable ({e}). Automatically switching to multi-core CPU (int8)..."
                        })
                    self._model = WhisperModel(
                        model_size_or_path=getattr(self, "_target_model", self.model_size),
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cpu_threads,
                        download_root=getattr(self, "_effective_root", self.download_root),
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    segments_gen, info = run_transcription()
                    first_seg = next(segments_gen, None)
                else:
                    raise e
        except Exception as e:
            err_str = str(e).lower()
            if "cublas" in err_str or "cuda" in err_str or "cudnn" in err_str or "driver" in err_str:
                if progress_callback:
                    progress_callback({
                        "type": "status_note",
                        "msg": f"CUDA initialization failed ({e}). Automatically switching to multi-core CPU (int8)..."
                    })
                self._model = WhisperModel(
                    model_size_or_path=getattr(self, "_target_model", self.model_size),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                    download_root=getattr(self, "_effective_root", self.download_root),
                )
                self.device = "cpu"
                self.compute_type = "int8"
                segments_gen, info = run_transcription()
                first_seg = next(segments_gen, None)
            else:
                raise e

        duration = info.duration or 1.0
        segments: List[Dict[str, Any]] = []
        track_flags: List[Dict[str, Any]] = []

        def all_segments():
            if first_seg is not None:
                yield first_seg
            for s in segments_gen:
                yield s

        for seg in all_segments():
            if cancel_check and cancel_check():
                break

            # Split coarse Faster-Whisper segments (up to 30s) into fine-grained conversational utterances
            utterances = split_segment_into_utterances(seg)
            if not utterances:
                continue

            for utt in utterances:
                utt_text = utt["text"].strip()
                if not utt_text:
                    continue

                utt_flags = []
                if enable_moderation and flagged_regex:
                    words_list = utt.get("words", [])
                    for w in words_list:
                        w_clean = re.sub(r"[^\w]", "", w.word).lower()
                        if flagged_regex.search(w_clean):
                            flag_item = {
                                "track_idx": track_idx,
                                "speaker": speaker_name,
                                "audio_path": audio_path,
                                "word": w.word.strip(),
                                "start": round(w.start, 3),
                                "end": round(w.end, 3),
                                "timecode": seconds_to_timecode(w.start),
                                "timestamp_hms": seconds_to_hms_ms(w.start),
                                "context": utt_text,
                            }
                            utt_flags.append(flag_item)
                            track_flags.append(flag_item)
                            if progress_callback:
                                progress_callback({
                                    "type": "moderation_flag",
                                    "flag": flag_item,
                                    "speaker": speaker_name,
                                    "timecode": flag_item["timecode"],
                                    "word": flag_item["word"],
                                    "context": utt_text,
                                })

                seg_data = {
                    "track_idx": track_idx,
                    "speaker": speaker_name,
                    "start": utt["start"],
                    "end": utt["end"],
                    "duration": utt["duration"],
                    "text": utt_text,
                    "avg_logprob": round(getattr(seg, "avg_logprob", 0.0), 3),
                    "flagged": (len(utt_flags) > 0),
                    "flags": utt_flags,
                }
                segments.append(seg_data)

                if progress_callback:
                    progress_callback({
                        "track_idx": track_idx,
                        "total_tracks": total_tracks,
                        "speaker": speaker_name,
                        "current_sec": utt["end"],
                        "total_sec": duration,
                        "progress_pct": min(100.0, (utt["end"] / duration) * 100.0),
                        "latest_segment": seg_data,
                    })

        return segments, track_flags

    def transcribe_multitrack_session(
        self,
        tracks_config: List[Dict[str, Any]],
        initial_prompt: Optional[str] = None,
        language: str = "en",
        pause_merge_sec: float = 2.0,
        enable_moderation: bool = True,
        flagged_terms: Union[str, List[str], None] = None,
        auto_mute_flags: bool = False,
        censored_out_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes multiple audio stems and merges them into a chronological script.
        tracks_config: list of dicts: [{"path": "...", "speaker": "James (GM)", "active": True}, ...]
        """
        self.load_model(status_callback)

        active_tracks = [t for t in tracks_config if t.get("active", True) and os.path.isfile(t.get("path", ""))]
        if not active_tracks:
            raise ValueError("No active, valid audio tracks provided for transcription.")

        total_tracks = len(active_tracks)
        all_raw_segments: List[Dict[str, Any]] = []
        all_session_flags: List[Dict[str, Any]] = []
        flags_by_track: Dict[str, List[Dict[str, Any]]] = {}

        # Dynamically extract character names and players to prime Whisper's context prompt
        char_names = []
        player_names = []
        for tr in active_tracks:
            spk = tr.get("speaker", "")
            bracket_match = re.search(r'\(([^)]+)\)', spk)
            if bracket_match:
                cname = bracket_match.group(1).strip()
                if cname.upper() not in ("GM", "DM", "HOST", "ST", "NARRATOR") and cname not in char_names:
                    char_names.append(cname)
            pname = re.sub(r'\(.*?\)', '', spk).strip()
            if pname and pname not in player_names:
                player_names.append(pname)

        prompt_parts = []
        if char_names:
            prompt_parts.append(f"Characters: {', '.join(char_names)}.")
        if player_names:
            prompt_parts.append(f"Players/GM: {', '.join(player_names)}.")
        if initial_prompt:
            prompt_parts.append(initial_prompt)
        session_prompt = " ".join(prompt_parts) if prompt_parts else initial_prompt

        start_wall_time = time.time()

        for idx, tr in enumerate(active_tracks, start=1):
            if cancel_check and cancel_check():
                if status_callback:
                    status_callback("Transcription cancelled by user.")
                break

            spk = tr.get("speaker") or f"Speaker {idx}"
            p = tr["path"]

            if status_callback:
                status_callback(f"[{idx}/{total_tracks}] Transcribing {spk} ({os.path.basename(p)})...")

            track_segs, track_flags = self.transcribe_track(
                audio_path=p,
                speaker_name=spk,
                track_idx=idx,
                total_tracks=total_tracks,
                initial_prompt=session_prompt,
                language=language,
                enable_moderation=enable_moderation,
                flagged_terms=flagged_terms,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            all_raw_segments.extend(track_segs)
            all_session_flags.extend(track_flags)
            flags_by_track[p] = track_flags

        total_elapsed = time.time() - start_wall_time

        if status_callback:
            status_callback(f"Merging and ordering {len(all_raw_segments)} segments across {total_tracks} tracks...")

        # Chronological Alignment & Paragraph Consolidation
        merged_dialogue = merge_multitrack_segments(all_raw_segments, pause_merge_sec=pause_merge_sec)

        # Compute Speaker Statistics
        stats = compute_speaker_stats(merged_dialogue)

        # Auto-Mute flagged audio stems if requested
        muted_files: List[str] = []
        if auto_mute_flags and all_session_flags:
            if status_callback:
                status_callback(f"[+] Applying digital silence mutes to {len(all_session_flags)} flagged word(s) in audio stems...")
            for tr in active_tracks:
                p = tr["path"]
                t_flags = flags_by_track.get(p, [])
                if t_flags:
                    spk = tr.get("speaker") or "Speaker"
                    base_dir = censored_out_dir or os.path.join(os.path.dirname(os.path.abspath(p)), "Mastered_Safe")
                    out_censored = os.path.join(base_dir, os.path.basename(p))
                    cnt = mute_audio_flagged_segments(p, t_flags, out_censored)
                    if cnt > 0:
                        muted_files.append(out_censored)
                        if status_callback:
                            status_callback(f"  🔇 Muted {cnt} sensitive word(s) on {spk}'s track -> {out_censored}")

        return {
            "session_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_size": self.model_size,
            "total_tracks": total_tracks,
            "total_segments_raw": len(all_raw_segments),
            "total_dialogue_blocks": len(merged_dialogue),
            "processing_time_sec": round(total_elapsed, 1),
            "speaker_stats": stats,
            "moderation_flags": all_session_flags,
            "muted_audio_files": muted_files,
            "segments": merged_dialogue,
        }


def merge_multitrack_segments(
    raw_segments: List[Dict[str, Any]],
    pause_merge_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Takes segments from all speakers, sorts them strictly by start time,
    and merges consecutive segments from the SAME speaker into natural conversational blocks
    if the silence gap between them is <= pause_merge_sec and block duration is <= 12.0s.
    """
    if not raw_segments:
        return []

    # Sort strictly by start timestamp, then end timestamp
    sorted_segs = sorted(raw_segments, key=lambda s: (s["start"], s["end"]))

    merged: List[Dict[str, Any]] = []
    curr: Optional[Dict[str, Any]] = None

    for seg in sorted_segs:
        text = seg["text"].strip()
        if not text:
            continue

        if curr is None:
            curr = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "text": text,
                "track_idx": seg.get("track_idx", 1),
                "flagged": seg.get("flagged", False),
            }
        else:
            # Check if same speaker, small gap, and block does not exceed 12.0s
            gap = seg["start"] - curr["end"]
            combined_dur = seg["end"] - curr["start"]
            if curr["speaker"] == seg["speaker"] and -1.0 <= gap <= pause_merge_sec and combined_dur <= 12.0:

                curr["end"] = max(curr["end"], seg["end"])
                curr["duration"] = round(curr["end"] - curr["start"], 2)
                # Combine text cleanly
                curr["text"] = f"{curr['text']} {text}".strip()
                if seg.get("flagged"):
                    curr["flagged"] = True
            else:
                curr["text"] = correct_campaign_vocabulary(curr["text"])
                merged.append(curr)
                curr = {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["duration"],
                    "text": text,
                    "track_idx": seg.get("track_idx", 1),
                    "flagged": seg.get("flagged", False),
                }

    if curr is not None:
        curr["text"] = correct_campaign_vocabulary(curr["text"])
        merged.append(curr)

    return merged


def correct_transcript_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Reads an existing transcript script (.txt or .srt), applies Star Wars 5e campaign
    vocabulary and character name spelling corrections, and writes the corrected file.
    If output_path is None, saves as [base]_Corrected.[ext].
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Transcript file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    corrected = correct_campaign_vocabulary(content)

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_Corrected{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corrected)

    return output_path


def compute_speaker_stats(merged_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Computes talk-time duration, word counts, and lines per speaker."""
    stats: Dict[str, Dict[str, Any]] = {}
    total_talk_sec = 0.0
    total_words = 0

    for seg in merged_segments:
        spk = seg["speaker"]
        dur = seg["end"] - seg["start"]
        words = len(seg["text"].split())

        if spk not in stats:
            stats[spk] = {
                "dialogue_turns": 0,
                "total_seconds": 0.0,
                "word_count": 0,
            }

        stats[spk]["dialogue_turns"] += 1
        stats[spk]["total_seconds"] += dur
        stats[spk]["word_count"] += words

        total_talk_sec += dur
        total_words += words

    for spk, data in stats.items():
        data["total_seconds"] = round(data["total_seconds"], 1)
        data["talk_percentage"] = round((data["total_seconds"] / total_talk_sec * 100.0) if total_talk_sec > 0 else 0.0, 1)
        data["word_percentage"] = round((data["word_count"] / total_words * 100.0) if total_words > 0 else 0.0, 1)

    return stats


# ==============================================================================
# Exporters
# ==============================================================================

def export_txt(
    result_data: Dict[str, Any],
    output_path: str,
    title: str = "Multitrack Session Transcript",
) -> str:
    """Exports a clean, beautifully formatted human-readable script."""
    lines = [
        "=" * 80,
        f"  {title.upper()}",
        f"  Generated: {result_data.get('session_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
        f"  Model: {result_data.get('model_size', 'whisper')} | Engine: faster-whisper (CTranslate2)",
        "=" * 80,
        "",
        "SPEAKER PARTICIPATION SUMMARY:",
        "-" * 80,
    ]

    stats = result_data.get("speaker_stats", {})
    for spk, s in sorted(stats.items(), key=lambda item: item[1]["total_seconds"], reverse=True):
        mins = int(s["total_seconds"] // 60)
        secs = int(s["total_seconds"] % 60)
        lines.append(
            f"  - {spk:24s} | {mins:3d}m {secs:02d}s ({s.get('talk_percentage', 0.0):4.1f}%) | "
            f"{s['word_count']:5d} words | {s['dialogue_turns']:4d} turns"
        )
    lines.append("-" * 80)
    lines.append("")
    lines.append("TRANSCRIPT SCRIPT:")
    lines.append("=" * 80)
    lines.append("")

    for seg in result_data.get("segments", []):
        ts = format_timestamp(seg["start"], "txt")
        spk = seg["speaker"]
        txt = seg["text"]
        flag_tag = " ⚠️ [FLAGGED]" if seg.get("flagged") else ""
        lines.append(f"{ts} {spk}{flag_tag}:")
        lines.append(f"{txt}")
        lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def export_srt(result_data: Dict[str, Any], output_path: str) -> str:
    """Exports standard SubRip (.srt) subtitle file."""
    lines = []
    for idx, seg in enumerate(result_data.get("segments", []), start=1):
        start_ts = format_timestamp(seg["start"], "srt")
        end_ts = format_timestamp(seg["end"], "srt")
        spk = seg["speaker"]
        txt = seg["text"]

        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"[{spk}]: {txt}")
        lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def export_json(result_data: Dict[str, Any], output_path: str) -> str:
    """Exports structured JSON file with full segment metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    return output_path


def export_moderation_report(
    flags: List[Dict[str, Any]],
    out_path: str,
    session_date: Optional[str] = None,
) -> str:
    """Generates an executive YouTube Safety & Content Moderation Audit Report."""
    session_date = session_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 76,
        "YOUTUBE CONTENT SAFETY & MODERATION AUDIT REPORT",
        f"Session Date: {session_date}",
        f"Total Flagged Incidents: {len(flags)}",
        "=" * 76,
        ""
    ]
    if not flags:
        lines.append("✓ CLEAN AUDIT: No sensitive language or YouTube policy slurs detected.")
    else:
        for idx, f in enumerate(flags, start=1):
            dur = round(f['end'] - f['start'], 2)
            lines.extend([
                f"[INCIDENT {idx}]",
                f"  • Speaker:           {f['speaker']} (Track {f.get('track_idx', '?')})",
                f"  • Timecode (SMPTE):  {f['timecode']} (30 fps)",
                f"  • Timeline Window:   {f['timestamp_hms']} ({f['start']:.2f}s - {f['end']:.2f}s, dur: {dur:.2f}s)",
                f"  • Flagged Term:      \"{f['word']}\"",
                f"  • Spoken Sentence:   \"{f['context']}\"",
                ""
            ])
        lines.extend([
            "=" * 76,
            "TIMELINE EDITING QUICK-GUIDE:",
            "1. DaVinci Resolve: Right-click timeline -> 'Import' -> 'Timeline Markers from CSV'",
            "   Select 'Session_Timeline_Markers_DaVinci.csv' to display red incident flags on the timeline.",
            "2. Adobe Premiere Pro: File -> Import -> 'Session_Timeline_Markers_Premiere.csv'",
            "3. If Auto-Mute was enabled, clean audio stems with digital silence on these exact words",
            "   have been saved into the 'Mastered_Safe/' folder.",
            "=" * 76,
        ])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return out_path


def export_timeline_markers_davinci(
    flags: List[Dict[str, Any]],
    out_csv: str,
    fps: float = 30.0,
) -> str:
    """Exports DaVinci Resolve compatible Timeline Marker CSV."""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Timecode", "Duration", "Color", "Marker Name", "Note"])
        for idx, f in enumerate(flags, start=1):
            tc = seconds_to_timecode(f["start"], fps=fps)
            dur_frames = max(1, int(round((f["end"] - f["start"]) * fps)))
            dur_tc = seconds_to_timecode(dur_frames / fps, fps=fps)
            writer.writerow([
                tc,
                dur_tc,
                "Red",
                f"YouTube Slur #{idx} ({f['speaker']})",
                f"Term: '{f['word']}' in: \"{f['context']}\""
            ])
    return out_csv


def export_timeline_markers_premiere(
    flags: List[Dict[str, Any]],
    out_csv: str,
    fps: float = 30.0,
) -> str:
    """Exports Adobe Premiere Pro compatible Marker CSV."""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"])
        for idx, f in enumerate(flags, start=1):
            in_tc = seconds_to_timecode(f["start"], fps=fps)
            out_tc = seconds_to_timecode(f["end"], fps=fps)
            dur_tc = seconds_to_timecode(f["end"] - f["start"], fps=fps)
            writer.writerow([
                f"YouTube Flag #{idx} ({f['speaker']})",
                f"Word: '{f['word']}' in \"{f['context']}\"",
                in_tc,
                out_tc,
                dur_tc,
                "Comment"
            ])
    return out_csv


def mute_audio_flagged_segments(
    audio_path: str,
    flags_for_track: List[Dict[str, Any]],
    out_path: str,
    pad_sec: float = 0.05,
    fade_ms: float = 12.0,
) -> int:
    """
    Applies transparent, click-free digital mutes (zeroes) over flagged word intervals.
    Returns the number of muted intervals.
    """
    if not flags_for_track or not os.path.isfile(audio_path):
        return 0

    data, sr = sf.read(audio_path)
    is_1d = (data.ndim == 1)
    fade_samp = max(1, int((fade_ms / 1000.0) * sr))
    fade_in = np.linspace(0.0, 1.0, fade_samp, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samp, dtype=np.float32)

    intervals = [(f["start"], f["end"]) for f in flags_for_track]
    intervals.sort(key=lambda x: x[0])
    merged = []
    for s, e in intervals:
        if not merged:
            merged.append([s, e])
        else:
            prev = merged[-1]
            if s <= prev[1] + pad_sec:
                prev[1] = max(prev[1], e)
            else:
                merged.append([s, e])

    muted_count = 0
    for s_sec, e_sec in merged:
        s_idx = max(0, int((s_sec - pad_sec) * sr))
        e_idx = min(len(data), int((e_sec + pad_sec) * sr))
        if s_idx >= e_idx:
            continue

        r_start = max(0, s_idx - fade_samp)
        actual_fade_out = s_idx - r_start
        if actual_fade_out > 0:
            curve = fade_out[-actual_fade_out:]
            if is_1d:
                data[r_start:s_idx] *= curve
            else:
                data[r_start:s_idx, :] *= curve[:, np.newaxis]

        if is_1d:
            data[s_idx:e_idx] = 0.0
        else:
            data[s_idx:e_idx, :] = 0.0

        r_end = min(len(data), e_idx + fade_samp)
        actual_fade_in = r_end - e_idx
        if actual_fade_in > 0:
            curve_in = fade_in[:actual_fade_in]
            if is_1d:
                data[e_idx:r_end] *= curve_in
            else:
                data[e_idx:r_end, :] *= curve_in[:, np.newaxis]

        muted_count += 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, data, sr)
    return muted_count


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multitrack Speech-to-Text Transcription & Chronological Script Merger"
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        required=True,
        help="List of tracks in 'path=SpeakerName' format, e.g. 'track1.mp3=James' 'track2.mp3=Sarah'",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output base path (e.g. Session_6_Transcript.txt)",
    )
    parser.add_argument(
        "--model",
        default="small.en",
        choices=MultitrackTranscriber.SUPPORTED_MODELS,
        help="Whisper model size (default: small.en)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Context keyword prompt for fantasy/sci-fi terms (e.g. 'SW5E, blaster, lightsaber, d20')",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=2.0,
        help="Max silence seconds between same-speaker segments to merge into one block (default: 2.0)",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also export .srt subtitle file alongside .txt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also export .json structured data alongside .txt",
    )

    args = parser.parse_args()

    tracks_config = []
    for item in args.tracks:
        if "=" in item:
            p, spk = item.split("=", 1)
        else:
            p = item
            spk = os.path.splitext(os.path.basename(p))[0]
        tracks_config.append({"path": p, "speaker": spk, "active": True})

    print(f"Loaded {len(tracks_config)} tracks:")
    for tr in tracks_config:
        print(f"  - {tr['speaker']}: {tr['path']}")

    transcriber = MultitrackTranscriber(model_size=args.model)

    def on_progress(pdata):
        print(
            f"  [{pdata['speaker']}] {pdata['progress_pct']:5.1f}% | "
            f"{format_timestamp(pdata['current_sec'])} -> {pdata['latest_segment']['text']}"
        )

    def on_status(msg):
        print(f">> {msg}")

    result = transcriber.transcribe_multitrack_session(
        tracks_config=tracks_config,
        initial_prompt=args.prompt,
        pause_merge_sec=args.merge_gap,
        progress_callback=on_progress,
        status_callback=on_status,
    )

    # Save .txt
    out_txt = args.output
    if not out_txt.endswith(".txt"):
        out_txt += ".txt"
    export_txt(result, out_txt)
    print(f"\n[SUCCESS] Formatted script exported to: {out_txt}")

    base_no_ext = os.path.splitext(out_txt)[0]
    if args.srt:
        out_srt = base_no_ext + ".srt"
        export_srt(result, out_srt)
        print(f"[SUCCESS] Subtitles exported to: {out_srt}")

    if args.json:
        out_json = base_no_ext + ".json"
        export_json(result, out_json)
        print(f"[SUCCESS] JSON metadata exported to: {out_json}")


if __name__ == "__main__":
    main()
