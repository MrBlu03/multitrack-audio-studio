#!/usr/bin/env python3
"""
Multitrack Audio Bleed Suppressor and Adaptive Spectral Gate
============================================================
Eliminates extreme microphone bleed in multitrack audio sessions
(podcasts, roundtables, tabletop RPGs, panels) by performing activity-aware
spectral magnitude cosine analysis and adaptive envelope gating.

- Works on lossy formats (MP3/Discord VoIP) as well as uncompressed WAV/FLAC.
- Distinguishes pure bleed from active solo speech and crosstalk.
- Smooth attack, hold, and release envelopes eliminate gating clicks.
- Streams audio in 30-second chunks with carry-over overlap buffers,
  enabling multi-hour 6+ channel sessions with minimal RAM usage.

Author: Antigravity DSP Tools
"""

import argparse
import os
import sys
import time
import math
import re
import subprocess
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
import scipy.signal as signal
import soundfile as sf
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# GPU acceleration via PyTorch (already bundled as a Whisper/CTranslate2 dep)
# Falls back to CPU silently on machines without CUDA.
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
    if _CUDA_AVAILABLE:
        _CUDA_DEVICE = _torch.device("cuda")
        # Keep a float32 dtype alias for brevity
        _TORCH_F32 = _torch.float32
except Exception:
    _torch = None  # type: ignore
    _CUDA_AVAILABLE = False
    _CUDA_DEVICE = None
    _TORCH_F32 = None

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


class AudioSource:
    """
    Unified streaming reader supporting:
    - Multiple individual audio files (one per track: MP3, WAV, FLAC, AIFF)
    - A single multichannel audio file (e.g. 6-channel WAV)
    """
    def __init__(self, file_paths: List[str]):
        if not file_paths:
            raise ValueError("No audio files provided.")
        
        self.file_paths = file_paths
        self.is_multichannel_file = (len(file_paths) == 1)
        self.handles: List[sf.SoundFile] = []
        
        if self.is_multichannel_file:
            handle = sf.SoundFile(file_paths[0], mode='r')
            self.handles.append(handle)
            self.samplerate = handle.samplerate
            self.n_tracks = handle.channels
            self.total_samples = handle.frames
        else:
            first_handle = sf.SoundFile(file_paths[0], mode='r')
            self.handles.append(first_handle)
            self.samplerate = first_handle.samplerate
            self.total_samples = first_handle.frames
            
            for p in file_paths[1:]:
                h = sf.SoundFile(p, mode='r')
                if h.samplerate != self.samplerate:
                    raise ValueError(f"Sample rate mismatch: {p} has {h.samplerate} Hz, expected {self.samplerate} Hz.")
                self.total_samples = min(self.total_samples, h.frames)
                self.handles.append(h)
                
            self.n_tracks = len(self.handles)
            
        self.duration_sec = self.total_samples / self.samplerate

    def read_chunk(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Read up to n_samples across all tracks.
        Returns array of shape (n_tracks, samples_read) as float32, or None if EOF.
        """
        if self.is_multichannel_file:
            data = self.handles[0].read(n_samples, dtype='float32', always_2d=True)
            if len(data) == 0:
                return None
            return data.T  # shape: (n_tracks, n_samples)
        else:
            chunks = []
            for h in self.handles:
                c = h.read(n_samples, dtype='float32')
                if len(c) == 0:
                    return None
                if c.ndim > 1:
                    c = c[:, 0]  # Take first channel if file is stereo
                chunks.append(c)
            min_len = min(len(c) for c in chunks)
            if min_len == 0:
                return None
            return np.array([c[:min_len] for c in chunks], dtype=np.float32)

    def seek(self, sample_idx: int):
        for h in self.handles:
            h.seek(sample_idx)

    def close(self):
        for h in self.handles:
            try:
                h.close()
            except Exception:
                pass


class BleedCalibrator:
    """
    Calibrates dynamic target and reference speech/noise thresholds from the session audio.
    """
    def __init__(self, target_idx: int, n_tracks: int, sr: int):
        self.target_idx = target_idx
        self.n_tracks = n_tracks
        self.other_indices = [i for i in range(n_tracks) if i != target_idx]
        self.sr = sr

    def calibrate(self, audio_src: AudioSource, max_scan_sec: float = 120.0, sensitivity: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Scan initial duration of recording to compute dynamic threshold levels.
        Returns:
            ref_speech_thresh: Linear RMS threshold for reference speech activity.
            tgt_speech_thresh: Linear RMS threshold for target vocal activity.
            target_noise_floor: Measured noise floor of target track.
            target_speech_level: Measured typical active speech level of target track.
        """
        scan_samples = min(audio_src.total_samples, int(max_scan_sec * self.sr))
        audio_src.seek(0)
        
        raw = audio_src.read_chunk(scan_samples)
        audio_src.seek(0)
        
        if raw is None or raw.shape[1] < int(1.0 * self.sr):
            ref_thresh = 10.0 ** (-42.0 / 20.0) / max(0.1, sensitivity)
            tgt_thresh = 10.0 ** (-48.0 / 20.0) / max(0.1, sensitivity)
            return ref_thresh, tgt_thresh, 1e-4, 0.05
            
        frame_len = int(0.020 * self.sr)
        hop_len = int(0.010 * self.sr)
        n_frames = (raw.shape[1] - frame_len) // hop_len
        
        frame_rms = np.zeros((self.n_tracks, n_frames), dtype=np.float32)
        for i in range(self.n_tracks):
            shape = (n_frames, frame_len)
            strides = (raw[i].strides[0] * hop_len, raw[i].strides[0])
            f_view = np.lib.stride_tricks.as_strided(raw[i], shape=shape, strides=strides)
            frame_rms[i] = np.sqrt(np.mean(f_view**2, axis=1))
            
        # Target track profile
        t_rms = frame_rms[self.target_idx]
        active_t = t_rms[t_rms > 1e-4]
        if len(active_t) > 50:
            target_noise_floor = float(np.percentile(active_t, 15))
            target_speech_level = float(np.percentile(active_t, 90))
        else:
            target_noise_floor = 10.0 ** (-70.0 / 20.0)
            target_speech_level = 10.0 ** (-20.0 / 20.0)
            
        # Reference tracks profile
        ref_rms_all = frame_rms[self.other_indices]
        ref_max = np.max(ref_rms_all, axis=0)
        active_ref = ref_max[ref_max > 1e-4]
        if len(active_ref) > 50:
            ref_p90 = float(np.percentile(active_ref, 90))
            ref_floor = float(np.percentile(active_ref, 15))
        else:
            ref_p90 = 10.0 ** (-20.0 / 20.0)
            ref_floor = 10.0 ** (-70.0 / 20.0)
            
        # Dynamic threshold computation
        ref_speech_dB = 20.0 * math.log10(max(1e-9, ref_p90)) - 22.0
        ref_speech_dB = max(-48.0, min(-36.0, ref_speech_dB))
        ref_speech_thresh = (10.0 ** (ref_speech_dB / 20.0)) / max(0.1, sensitivity)
        
        tgt_speech_dB = 20.0 * math.log10(max(1e-9, target_speech_level)) - 28.0
        tgt_speech_dB = max(-54.0, min(-42.0, tgt_speech_dB))
        base_tgt = 10.0 ** (tgt_speech_dB / 20.0)
        tgt_speech_thresh = max(target_noise_floor * 1.5, base_tgt) / max(0.1, sensitivity)
        
        return ref_speech_thresh, tgt_speech_thresh, target_noise_floor, target_speech_level


class StreamingMultitrackBleedGate:
    """
    Precision Multi-Track Acoustic and VoIP Bleed Gate.
    
    Operates chunk-by-chunk with sample-accurate carry-over overlap buffers.
    - Employs calibrated VoIP delay windows per remote speaker.
    - Computes short-time Fourier transform (STFT) spectral magnitude cosine similarity.
    - Enforces acoustic attenuation bounds (bleed cannot exceed reference speaker energy).
    - Uses causal forward-only bridging to prevent syllable chatter without eating speech.
    - Employs smooth attack, hold, and release envelopes with sample-accurate interpolation.
    """
    def __init__(
        self,
        sr: int = 48000,
        target_idx: int = 4,
        n_tracks: int = 6,
        match_cutoff: float = 0.48,
        sensitivity: float = 1.0,
        floor_db: float = -60.0,
        attack_ms: float = 10.0,
        hold_ms: float = 160.0,
        release_ms: float = 85.0,
        delay_windows: Optional[Dict[int, np.ndarray]] = None,
        ref_speech_thresh: Optional[float] = None,
        tgt_speech_thresh: Optional[float] = None,
        block_ms: float = 20.0,
        hop_ms: float = 10.0,
        nperseg: int = 1024,
        noverlap: int = 512,
        enhance: bool = False,
        **kwargs
    ):
        self.sr = sr
        self.target_idx = target_idx
        self.n_tracks = n_tracks
        self.other_indices = [i for i in range(n_tracks) if i != target_idx]
        self.match_cutoff = match_cutoff
        self.sensitivity = max(0.1, sensitivity)
        self.floor_gain = 0.0 if floor_db <= -98.0 else 10.0 ** (floor_db / 20.0)
        self.enhance = enhance
        self.enhancer = VoiceEnhancer(sr) if enhance else None
        
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.hop_stft = nperseg - noverlap
        self.win_samp = int((block_ms / 1000.0) * sr)
        
        self.attack_frames = max(1, int((attack_ms / 1000.0) / (self.hop_stft / sr)))
        self.hold_frames = max(1, int((hold_ms / 1000.0) / (self.hop_stft / sr)))
        self.rel_frames = max(1, int((release_ms / 1000.0) / (self.hop_stft / sr)))
        self.current_hold = 0
        self.current_gain = 0.0
        
        self.max_delay_sec = 1.0
        self.history_samp = int(self.max_delay_sec * sr) + self.nperseg
        
        if delay_windows is not None:
            self.delay_windows = delay_windows
        else:
            # Broad VoIP dynamic delay search (15ms to 850ms in 20ms steps)
            # Immune to WebRTC jitter spikes, ping fluctuations, and packet buffering
            broad_win = np.arange(0.015, 0.850, 0.020)
            self.delay_windows = {k: broad_win for k in self.other_indices}
            
        self.t_history = np.zeros(self.history_samp, dtype=np.float32)
        self.history_buffers = {k: np.zeros(self.history_samp, dtype=np.float32) for k in self.other_indices}

    def process_chunk(self, chunk: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a chunk of shape (n_tracks, n_samples).
        Returns:
            processed_target: 1D array of cleaned audio.
            stats: Dict with 'unmuted_ratio' and 'avg_gain'.
        """
        n_samples = chunk.shape[1]
        t_chunk = chunk[self.target_idx]
        
        t_full = np.concatenate([self.t_history, t_chunk])
        self.t_history = t_full[-self.history_samp:].copy()
        
        _, _, z5 = signal.stft(t_full, fs=self.sr, nperseg=self.nperseg, noverlap=self.noverlap)
        m5 = np.abs(z5)
        rms5 = np.sqrt(uniform_filter1d(t_full**2, size=self.win_samp)[::self.hop_stft])
        n_stft = min(m5.shape[1], len(rms5))
        
        m5 = m5[:, :n_stft]
        norm5 = np.sqrt(np.sum(m5**2, axis=0)) + 1e-12
        db5 = 20.0 * np.log10(rms5[:n_stft] + 1e-9)
        
        max_bleed_sim = np.zeros(n_stft, dtype=np.float32)
        max_ref_rms_global = np.zeros(n_stft, dtype=np.float32)
        
        default_win = np.concatenate([np.arange(0.000, 0.121, 0.015), np.arange(0.150, 0.501, 0.020), np.arange(0.550, 0.901, 0.030)])
        
        for k in self.other_indices:
            ref_chunk = chunk[k]
            ref_full = np.concatenate([self.history_buffers[k], ref_chunk])
            self.history_buffers[k] = ref_full[-self.history_samp:].copy()

            _, _, zk = signal.stft(ref_full, fs=self.sr, nperseg=self.nperseg, noverlap=self.noverlap)
            mk = np.abs(zk)[:, :n_stft]
            norm_k = np.sqrt(np.sum(mk**2, axis=0)) + 1e-12

            rms_k = np.sqrt(uniform_filter1d(ref_full**2, size=self.win_samp)[::self.hop_stft])[:n_stft]
            db_k = 20.0 * np.log10(rms_k + 1e-9)
            max_ref_rms_global = np.maximum(max_ref_rms_global, rms_k)

            d_list = self.delay_windows.get(k, default_win)
            # Convert delay seconds → STFT-frame integer shifts (vectorised, no Python loop)
            shifts = np.unique(np.round(
                np.asarray(d_list) * self.sr / self.hop_stft
            ).astype(np.int32))

            # GPU-accelerated cosine similarity across all shifts for this reference track
            track_sim, _ = _gpu_bleed_cosine_batch(
                m5, norm5, db5, mk, norm_k, db_k, rms_k, shifts, n_stft
            )
            max_bleed_sim = np.maximum(max_bleed_sim, track_sim)
                    
        # 1. Causal reference envelope and bleed extension
        max_ref_win_rms = maximum_filter1d(max_ref_rms_global, size=101, origin=-40)
        max_ref_win_db = 20.0 * np.log10(max_ref_win_rms + 1e-9)
        bleed_bridge = maximum_filter1d(max_bleed_sim, size=17, origin=-5)
        
        silence_floor = -40.0 - 6.0 * (self.sensitivity - 1.0)
        loud_speech_floor = -18.0 - 4.0 * (self.sensitivity - 1.0)
        
        # 2. Frame-level gate decisions with Dugan Sidechain Speech Rule
        raw_gate = np.zeros(n_stft, dtype=np.float32)
        for f in range(n_stft):
            e5 = db5[f]
            sim = bleed_bridge[f]
            ref_win = max_ref_win_db[f]
            
            # Is another speaker actively talking right now?
            other_active = (ref_win > -36.0)
            
            # Local speech test: Mathew speaking into a 1-inch boom mic hits -22 dB or higher.
            # Headphone bleed hits -32 dB or lower.
            mathew_dominant = (e5 >= -22.0) or (e5 >= ref_win - 3.0 and e5 >= -26.0)
            
            if e5 < silence_floor:
                raw_gate[f] = 0.0
            elif other_active and not mathew_dominant:
                # Other speaker is active, and Mathew is not overpowering them:
                # If there is even mild spectral similarity (muffled earcup bleed >= 0.30)
                # OR his energy is within the classic acoustic earcup attenuation range (e5 <= ref_win - 6.0 dB):
                # Lock the gate shut to prevent breakthrough!
                if sim >= 0.30 or (e5 <= ref_win - 6.0):
                    raw_gate[f] = 0.0
                elif sim >= self.match_cutoff:
                    raw_gate[f] = 0.0
                else:
                    # Borderline: only pass if high confidence vocal level
                    raw_gate[f] = 1.0 if e5 >= -24.0 else 0.0
            elif sim >= self.match_cutoff:
                # Direct spectral bleed match (even if other_active envelope was borderline)
                raw_gate[f] = 0.0
            else:
                # Mathew speaking solo with clean vocal energy
                raw_gate[f] = 1.0
                
        # 3. Smooth envelope over the current output chunk
        chunk_stft_start = int(self.history_samp / self.hop_stft)
        n_out_frames = (n_samples + self.hop_stft - 1) // self.hop_stft
        smoothed_chunk = np.zeros(n_out_frames, dtype=np.float32)
        
        for f_idx in range(n_out_frames):
            src_f = chunk_stft_start + f_idx
            dec = raw_gate[src_f] if src_f < n_stft else 0.0
            if dec > 0.5:
                self.current_hold = self.hold_frames
                self.current_gain = min(1.0, self.current_gain + (1.0 / self.attack_frames))
            elif self.current_hold > 0:
                self.current_hold -= 1
                self.current_gain = 1.0
            else:
                self.current_gain = max(0.0, self.current_gain - (1.0 / self.rel_frames))
            smoothed_chunk[f_idx] = self.current_gain
            
        smoothed_chunk = self.floor_gain + (1.0 - self.floor_gain) * smoothed_chunk
        
        # 4. Sample-accurate linear interpolation
        frame_centers = np.arange(n_out_frames) * self.hop_stft + (self.win_samp // 2)
        sample_gain = np.interp(
            np.arange(n_samples),
            frame_centers,
            smoothed_chunk,
            left=smoothed_chunk[0],
            right=smoothed_chunk[-1]
        ).astype(np.float32)
        
        processed_target = t_chunk * sample_gain
        if self.enhance and self.enhancer is not None:
            processed_target = self.enhancer.process_chunk(processed_target)
            
        unmuted_frames = np.sum(sample_gain > 0.1)
        stats = {
            "unmuted_ratio": float(unmuted_frames / max(1, n_samples)),
            "avg_gain": float(np.mean(sample_gain))
        }
        return processed_target, stats


# ---------------------------------------------------------------------------
# GPU-accelerated DSP helpers (RTX 3060 / any CUDA device)
# Falls back to CPU/scipy transparently when CUDA is unavailable.
# ---------------------------------------------------------------------------

def _gpu_spectral_subtract(
    audio: np.ndarray,
    noise_mag: np.ndarray,
    alpha: float,
    nperseg: int,
    noverlap: int,
    sr: int,
) -> np.ndarray:
    """
    STFT-domain spectral noise subtraction on GPU (torch.stft / torch.istft).
    Equivalent to the scipy STFT path in LaptopMicEnhancer but runs on CUDA,
    giving ~8-15x speedup for the 30-second chunks used here.
    Falls back to scipy if torch/CUDA is unavailable.
    """
    if not _CUDA_AVAILABLE or _torch is None:
        # CPU scipy fallback (unchanged behaviour)
        _, _, z = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag   = np.abs(z)
        phase = np.angle(z)
        sub   = np.maximum(mag - alpha * noise_mag, 0.10 * mag)
        _, y  = signal.istft(sub * np.exp(1j * phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
        return y.astype(np.float32)

    hop = nperseg - noverlap
    x_t  = _torch.from_numpy(audio).to(_CUDA_DEVICE, dtype=_TORCH_F32)
    nm_t = _torch.from_numpy(noise_mag[:, 0]).to(_CUDA_DEVICE, dtype=_TORCH_F32)

    win  = _torch.hann_window(nperseg, device=_CUDA_DEVICE)
    # torch.stft → complex (freq_bins, frames)
    Z       = _torch.stft(x_t, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                          window=win, return_complex=True)
    mag_t   = Z.abs()
    phase_t = Z.angle()

    # Spectral subtraction — 10% noise floor prevents musical noise artefacts.
    # At 4% the spectrum was nearly zeroed in quiet bins, leaving tonal spikes.
    sub_t   = (mag_t - alpha * nm_t.unsqueeze(1)).clamp_min(0.10 * mag_t)
    Z_clean = _torch.polar(sub_t, phase_t)

    y_t = _torch.istft(Z_clean, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                       window=win, length=len(audio))
    return y_t.cpu().numpy().astype(np.float32)


def _gpu_bleed_cosine_batch(
    m5: np.ndarray,
    norm5: np.ndarray,
    db5: np.ndarray,
    mk: np.ndarray,
    norm_k: np.ndarray,
    db_k: np.ndarray,
    rms_k: np.ndarray,
    shifts: np.ndarray,
    n_stft: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batches all delay-shift cosine similarity comparisons on GPU for one reference track.
    Returns (max_bleed_sim_update, max_ref_rms_update) as CPU numpy arrays.
    Falls back to vectorised numpy on CPU when CUDA unavailable.
    """
    max_sim = np.zeros(n_stft, dtype=np.float32)

    if _CUDA_AVAILABLE and _torch is not None:
        M5  = _torch.from_numpy(m5).to(_CUDA_DEVICE)
        N5  = _torch.from_numpy(norm5).to(_CUDA_DEVICE)
        D5  = _torch.from_numpy(db5).to(_CUDA_DEVICE)
        MK  = _torch.from_numpy(mk).to(_CUDA_DEVICE)
        NK  = _torch.from_numpy(norm_k).to(_CUDA_DEVICE)
        DK  = _torch.from_numpy(db_k).to(_CUDA_DEVICE)
        sim_acc = _torch.zeros(n_stft, device=_CUDA_DEVICE, dtype=_torch.float32)

        for shift in shifts:
            shift = int(shift)
            if shift >= n_stft:
                continue
            L    = n_stft - shift
            dot  = (M5[:, shift:] * MK[:, :L]).sum(dim=0)
            sim  = dot / (N5[shift:] * NK[:L] + 1e-12)
            valid = (DK[:L] > -43.0) & (D5[shift:] <= DK[:L] + 4.0) & (N5[shift:] > 0.003)
            sim_acc[shift:] = _torch.maximum(sim_acc[shift:],
                                             _torch.where(valid, sim, _torch.zeros(L, device=_CUDA_DEVICE)))

        max_sim = sim_acc.cpu().numpy()
    else:
        # Vectorised CPU path
        for shift in shifts:
            shift = int(shift)
            if shift >= n_stft:
                continue
            L    = n_stft - shift
            dot  = np.sum(m5[:, shift:] * mk[:, :L], axis=0)
            sim  = dot / (norm5[shift:] * norm_k[:L] + 1e-12)
            valid = (db_k[:L] > -43.0) & (db5[shift:] <= db_k[:L] + 4.0) & (norm5[shift:] > 0.003)
            max_sim[shift:] = np.maximum(max_sim[shift:], np.where(valid, sim, 0.0))

    return max_sim, rms_k


def biquad_peaking(gain_db: float, f0: float, q: float, fs: int):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    return np.array([b0/a0, b1/a0, b2/a0], dtype=np.float32), np.array([1.0, a1/a0, a2/a0], dtype=np.float32)


def biquad_shelf(gain_db: float, f0: float, is_low: bool, fs: int):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
    cos_w0 = np.cos(w0)
    if is_low:
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
    else:
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
    return np.array([b0/a0, b1/a0, b2/a0], dtype=np.float32), np.array([1.0, a1/a0, a2/a0], dtype=np.float32)


class VoiceEnhancer:
    """
    Broadcast Vocal Enhancement & Tuning Engine.
    Transforms muffled, boxy headset audio into a rich, clear broadcast voice with deep chest authority.
    - 75 Hz rumble & plosive high-pass
    - 145 Hz chest warmth & proximity restoration (+4.5 dB low shelf)
    - 185 Hz vocal fundamental body reinforcement (+2.0 dB peaking, Q=1.2)
    - 450 Hz plastic cavity de-boxing (-4.5 dB peaking, Q=1.5)
    - 3.3 kHz harshness de-esser notch (-2.5 dB peaking, Q=2.0)
    - 6.5 kHz broadcast air & presence (+2.0 dB high shelf)
    - Soft harmonic saturation drive (1.15x tanh)
    - Broadcast speech leveling compressor (2.5:1 ratio, soft knee)
    - Streaming stateful execution (zero boundary clicks across chunks)
    """
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.b_hp, self.a_hp = signal.butter(2, 75.0 / (sr / 2), btype='highpass')
        self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
        
        # 145 Hz low-shelf (+4.5 dB) for rich chest depth
        self.b_ls, self.a_ls = biquad_shelf(+4.5, 145.0, True, sr)
        self.zi_ls = signal.lfilter_zi(self.b_ls, self.a_ls)
        
        # 185 Hz body peak (+2.0 dB, Q=1.2) for fundamental vocal weight
        self.b_body, self.a_body = biquad_peaking(+2.0, 185.0, 1.2, sr)
        self.zi_body = signal.lfilter_zi(self.b_body, self.a_body)
        
        # 450 Hz notch (-4.5 dB, Q=1.5) for headset plastic cup boxiness
        self.b_nb, self.a_nb = biquad_peaking(-4.5, 450.0, 1.5, sr)
        self.zi_nb = signal.lfilter_zi(self.b_nb, self.a_nb)
        
        # 3.3 kHz notch (-2.5 dB, Q=2.0) for tinny sibilance / harshness
        self.b_nh, self.a_nh = biquad_peaking(-2.5, 3300.0, 2.0, sr)
        self.zi_nh = signal.lfilter_zi(self.b_nh, self.a_nh)
        
        # 6.5 kHz high-shelf (+2.0 dB) for smooth broadcast air
        self.b_hs, self.a_hs = biquad_shelf(+2.0, 6500.0, False, sr)
        self.zi_hs = signal.lfilter_zi(self.b_hs, self.a_hs)
        
        self.b_sm, self.a_sm = signal.butter(1, 15.0 / (sr / 2), btype='lowpass')
        self.zi_sm = signal.lfilter_zi(self.b_sm, self.a_sm)
        
        self.rms_win = int(0.020 * sr)
        self.rms_hist = np.zeros(self.rms_win, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return chunk
            
        # 1. High-pass filter
        y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
        
        # 2. Parametric EQ chain: Chest Depth + Fundamental Body + De-box + De-harsh + Air
        y, self.zi_ls = signal.lfilter(self.b_ls, self.a_ls, y, zi=self.zi_ls)
        y, self.zi_body = signal.lfilter(self.b_body, self.a_body, y, zi=self.zi_body)
        y, self.zi_nb = signal.lfilter(self.b_nb, self.a_nb, y, zi=self.zi_nb)
        y, self.zi_nh = signal.lfilter(self.b_nh, self.a_nh, y, zi=self.zi_nh)
        y, self.zi_hs = signal.lfilter(self.b_hs, self.a_hs, y, zi=self.zi_hs)
        
        # 3. Soft harmonic saturation (chest warmth & analog tape body)
        drive = 1.15
        y = np.tanh(y * drive) / drive
        
        # 4. Broadcast dynamic leveler
        y_full = np.concatenate([self.rms_hist, y])
        self.rms_hist = y[-self.rms_win:].copy() if len(y) >= self.rms_win else y_full[-self.rms_win:].copy()
        
        rms_env = np.sqrt(np.maximum(0.0, uniform_filter1d(y_full**2, size=self.rms_win)[self.rms_win:]))
        rms_db = 20.0 * np.log10(rms_env + 1e-9)
        
        thresh_db = -24.0
        ratio = 2.5
        gain_comp_db = np.where(rms_db > thresh_db, (thresh_db - rms_db) * (1.0 - 1.0/ratio), 0.0)
        gain_linear = 10.0 ** (gain_comp_db / 20.0)
        
        gain_smooth, self.zi_sm = signal.lfilter(self.b_sm, self.a_sm, gain_linear, zi=self.zi_sm)
        y_comp = y * gain_smooth
        
        target_peak = 10.0 ** (-1.0 / 20.0)
        y_lim = np.clip(y_comp, -target_peak, target_peak)
        return y_lim.astype(np.float32)


class LaptopMicEnhancer:
    """
    Acoustic restoration engine for built-in laptop microphones:
    - 85 Hz Butterworth high-pass: removes chassis vibration & fan motor fundamental
    - Spectral noise subtraction: profiles and surgically removes stationary fan motor whine
      and background room hiss without phase warble
    - 4-stage Parametric EQ:
      * +2.5 dB low shelf @ 160 Hz (proximity warmth restoration)
      * -5.0 dB notch @ 340 Hz, Q=1.2 (desk reflection comb filter cut)
      * -3.5 dB notch @ 850 Hz, Q=1.6 (hollow chassis resonance cut)
      * +4.5 dB high shelf @ 3400 Hz (vocal articulation and air)
    - Adaptive speech downward expander: pushes pause noise down by 35-40 dB
    - Broadcast leveler & peak limiter (-1.0 dBFS ceiling)
    """
    def __init__(
        self,
        sr: int = 48000,
        noise_mag: Optional[np.ndarray] = None,
        alpha: float = 1.2,
        floor_db: float = -24.0,
        desk_cut_db: float = -5.0,
        hollow_cut_db: float = -3.5,
        warmth_db: float = 2.5,
        presence_db: float = 4.5
    ):
        self.sr = sr
        self.alpha = float(alpha)
        self.noise_mag = noise_mag
        self.floor_db = float(floor_db)
        self.nperseg = 1024
        self.noverlap = 768
        self.margin = 4096

        # Expander state tracking across chunks (prevents muting on chunk boundaries)
        self.h_c = 0
        self.cur_g = 0.0

        # 1. 85 Hz High-Pass
        self.b_hp, self.a_hp = signal.butter(3, 85.0 / (sr / 2), btype='highpass')
        self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)

        # 2. Parametric EQ
        self.b_ls, self.a_ls = biquad_shelf(warmth_db, 160.0, True, sr)
        self.zi_ls = signal.lfilter_zi(self.b_ls, self.a_ls)

        self.b_n1, self.a_n1 = biquad_peaking(desk_cut_db, 340.0, 1.2, sr)
        self.zi_n1 = signal.lfilter_zi(self.b_n1, self.a_n1)

        self.b_n2, self.a_n2 = biquad_peaking(hollow_cut_db, 850.0, 1.6, sr)
        self.zi_n2 = signal.lfilter_zi(self.b_n2, self.a_n2)

        self.b_hs, self.a_hs = biquad_shelf(presence_db, 3400.0, False, sr)
        self.zi_hs = signal.lfilter_zi(self.b_hs, self.a_hs)

        # STFT guard margin buffer
        self.prev_tail = np.zeros(self.margin, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray, is_last: bool = False) -> np.ndarray:
        if len(chunk) == 0:
            return chunk

        N = len(chunk)
        # 1. High-pass filter
        y_hp, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)

        # Build buffer with guard margins for STFT
        in_buf = np.concatenate([self.prev_tail, y_hp])
        if not is_last:
            self.prev_tail = y_hp[-self.margin:].copy()

        # 2. Spectral noise suppression — GPU-accelerated on CUDA, scipy fallback on CPU
        if self.noise_mag is not None:
            y_clean_full = _gpu_spectral_subtract(
                in_buf, self.noise_mag, self.alpha,
                self.nperseg, self.noverlap, self.sr
            )
            y_clean = y_clean_full[self.margin : self.margin + N]
            if len(y_clean) < N:
                y_clean = np.pad(y_clean, (0, N - len(y_clean)))
        else:
            y_clean = y_hp

        # 3. Parametric EQ
        y_eq, self.zi_ls = signal.lfilter(self.b_ls, self.a_ls, y_clean, zi=self.zi_ls)
        y_eq, self.zi_n1 = signal.lfilter(self.b_n1, self.a_n1, y_eq, zi=self.zi_n1)
        y_eq, self.zi_n2 = signal.lfilter(self.b_n2, self.a_n2, y_eq, zi=self.zi_n2)
        y_eq, self.zi_hs = signal.lfilter(self.b_hs, self.a_hs, y_eq, zi=self.zi_hs)

        # 4. Adaptive Speech Expander
        hop_g = int(0.010 * self.sr)
        win_g = int(0.020 * self.sr)
        n_fr = (len(y_eq) - win_g) // hop_g
        if n_fr > 0:
            shape = (n_fr, win_g)
            strides = (y_eq.strides[0]*hop_g, y_eq.strides[0])
            rms_fr = np.sqrt(np.mean(np.lib.stride_tricks.as_strided(y_eq, shape=shape, strides=strides)**2, axis=1))
            rms_fr_db = 20.0 * np.log10(rms_fr + 1e-9)

            gate_dec = rms_fr_db > -38.0
            hold_fr = int(0.450 / 0.010)
            rel_fr = int(0.140 / 0.010)

            smooth_g = np.zeros(n_fr, dtype=np.float32)
            for i in range(n_fr):
                if gate_dec[i]:
                    self.h_c = hold_fr
                    self.cur_g = 1.0
                elif self.h_c > 0:
                    self.h_c -= 1
                    self.cur_g = 1.0
                else:
                    self.cur_g = max(0.0, self.cur_g - (1.0 / rel_fr))
                smooth_g[i] = self.cur_g


            floor_gain = 10.0 ** (self.floor_db / 20.0)
            smooth_g = floor_gain + (1.0 - floor_gain) * smooth_g

            sample_g = np.interp(
                np.arange(len(y_eq)),
                np.arange(n_fr)*hop_g + win_g//2,
                smooth_g,
                left=smooth_g[0],
                right=smooth_g[-1]
            ).astype(np.float32)
            y_gated = y_eq * sample_g
        else:
            y_gated = y_eq

        # 5. Peak limiter
        target_peak = 10.0 ** (-1.0 / 20.0)
        peak = np.max(np.abs(y_gated))
        if peak > 0.01:
            scale = min(1.8, target_peak / peak)
            y_out = y_gated * scale
        else:
            y_out = y_gated
            
        y_out = np.clip(y_out, -target_peak, target_peak)
        return y_out.astype(np.float32)


class DialogueSafeSilenceGate:
    """
    Dialogue-Safe Voice Activity Downward Expander & Digital Silence Gate.
    Ensures zero speech interference:
    - Bandpass speech formants detector (100 - 6000 Hz)
    - Dual-threshold hysteresis: opens at open_thresh_db, holds until close_thresh_db
    - Lookahead pre-roll buffer (30 ms) so consonant attacks (p, t, k, s) are preserved at 100% unity gain
    - Natural speech hold window (280 ms) bridging intra-sentence pauses
    - Smooth cosine S-curve release (120 ms) fading into true digital silence (0.000000)
    - Stateful streaming across chunk boundaries
    """
    def __init__(
        self,
        sr: int = 48000,
        open_thresh_db: float = -34.0,
        close_thresh_db: float = -44.0,
        hold_ms: float = 280.0,
        rel_ms: float = 120.0,
        lookahead_ms: float = 30.0,
        floor_db: float = -120.0
    ):
        self.sr = sr
        self.open_thresh_db = float(open_thresh_db)
        self.close_thresh_db = float(close_thresh_db)
        self.hold_ms = float(hold_ms)
        self.rel_ms = float(rel_ms)
        self.lookahead_ms = float(lookahead_ms)
        self.floor_db = float(floor_db)
        self.is_digital_silence = (self.floor_db <= -90.0)
        self.floor_linear = 0.0 if self.is_digital_silence else (10.0 ** (self.floor_db / 20.0))

        self.hop = int(0.005 * sr)
        self.win = int(0.020 * sr)
        self.lookahead_fr = max(1, int((self.lookahead_ms / 1000.0) / (self.hop / sr)))
        self.hold_fr = max(1, int((self.hold_ms / 1000.0) / (self.hop / sr)))
        self.rel_fr = max(1, int((self.rel_ms / 1000.0) / (self.hop / sr)))

        self.b_det, self.a_det = signal.butter(2, [100.0 / (sr / 2), min(0.999, 6000.0 / (sr / 2))], btype='bandpass')
        self.zi_det = signal.lfilter_zi(self.b_det, self.a_det)

        self.is_open = False
        self.h_c = 0
        self.cur_g = 0.0
        self.margin = self.win + self.lookahead_fr * self.hop
        self.prev_chunk = np.zeros(self.margin, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray, is_last: bool = False) -> np.ndarray:
        if len(chunk) == 0:
            return chunk

        N = len(chunk)
        extended = np.concatenate([self.prev_chunk, chunk])
        if not is_last:
            self.prev_chunk = chunk[-self.margin:].copy()

        audio_det, self.zi_det = signal.lfilter(self.b_det, self.a_det, extended, zi=self.zi_det)

        n_fr = (len(extended) - self.win) // self.hop
        if n_fr <= 0:
            return chunk

        # Vectorised RMS: stride trick (no Python loop — same approach as LaptopMicEnhancer)
        shape   = (n_fr, self.win)
        strides = (audio_det.strides[0] * self.hop, audio_det.strides[0])
        frames  = np.lib.stride_tricks.as_strided(audio_det, shape=shape, strides=strides)
        rms     = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1)).astype(np.float32)
        rms_db  = 20.0 * np.log10(rms + 1e-9)

        # Hysteresis trigger — stateful, must stay as a loop
        raw_trigger = np.zeros(n_fr, dtype=bool)
        for i in range(n_fr):
            if not self.is_open:
                if rms_db[i] > self.open_thresh_db:
                    self.is_open = True
                    raw_trigger[i] = True
            else:
                if rms_db[i] < self.close_thresh_db:
                    self.is_open = False
                else:
                    raw_trigger[i] = True

        # Lookahead: vectorised with maximum_filter1d (no Python loop)
        # Equivalent to: for each frame, True if any raw_trigger in [i, i+lookahead_fr)
        is_speech_lookahead = maximum_filter1d(
            raw_trigger.astype(np.uint8),
            size=self.lookahead_fr,
            origin=-(self.lookahead_fr // 2)
        ).astype(bool)

        # Gate state envelope — stateful hold+release, must stay as a loop
        gate_state = np.zeros(n_fr, dtype=np.float32)
        for i in range(n_fr):
            if is_speech_lookahead[i]:
                self.h_c = self.hold_fr
                self.cur_g = 1.0
            elif self.h_c > 0:
                self.h_c -= 1
                self.cur_g = 1.0
            else:
                self.cur_g = max(0.0, self.cur_g - (1.0 / self.rel_fr))
            gate_state[i] = self.cur_g

        gate_smooth = 0.5 * (1.0 - np.cos(np.pi * gate_state))
        sample_g = np.interp(np.arange(len(extended)), np.arange(n_fr)*self.hop + self.win//2, gate_smooth)

        chunk_g = sample_g[self.margin : self.margin + N]
        if len(chunk_g) < N:
            chunk_g = np.pad(chunk_g, (0, N - len(chunk_g)), mode='edge')

        if self.is_digital_silence:
            chunk_g = np.where(chunk_g < 0.002, 0.0, chunk_g)
        else:
            chunk_g = self.floor_linear + (1.0 - self.floor_linear) * chunk_g

        return (chunk * chunk_g).astype(np.float32)


class BroadcastSpeechNormalizer:
    """
    Active Dialogue Loudness Normalizer & True Peak Limiter (ITU-R BS.1770-4 / EBU R128).

    Calibrates all multitrack dialogue stems to an identical target speech loudness
    (default: -18.0 LUFS) so that every participant in a multi-speaker recording is
    equally audible, intelligible, and balanced.

    Features:
    - K-Weighting Pre-Filter (Acoustic head model + high-pass filter)
    - Dual-stage Gating: Absolute (-70 LKFS) & Relative (-10 dB below ungated mean)
      to measure ONLY active vocal speech and ignore pauses, silence, and room noise.
    - Safety Gain Clamps: max boost +18.0 dB, max cut -15.0 dB. Bypasses if empty.
    - Transparent Lookahead Peak Limiter: 5ms lookahead, 50ms smooth exponential release,
      and strict brickwall ceiling at -1.0 dBFS to prevent any digital clipping during laughter/shouts.
    """
    def __init__(
        self,
        sr: int = 48000,
        target_lufs: float = -18.0,
        peak_ceiling_db: float = -1.0,
        max_gain_db: float = 18.0,
        min_gain_db: float = -15.0,
    ):
        self.sr = sr
        self.target_lufs = float(target_lufs)
        self.peak_ceiling_db = float(peak_ceiling_db)
        self.max_gain_db = float(max_gain_db)
        self.min_gain_db = float(min_gain_db)
        self.ceiling_linear = 10.0 ** (self.peak_ceiling_db / 20.0)
        self.b_hs, self.a_hs, self.b_hp, self.a_hp = self._compute_k_filters(sr)

    @staticmethod
    def _compute_k_filters(sr: int):
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196
        K = np.tan(np.pi * f0 / sr)
        Vh = 10.0 ** (G / 20.0)
        Vb = Vh ** 0.4996667741545416
        a0 = 1.0 + K / Q + K * K
        b_hs = np.array([(Vh + Vb * K / Q + K * K)/a0, (2.0 * (K * K - Vh))/a0, (Vh - Vb * K / Q + K * K)/a0])
        a_hs = np.array([1.0, (2.0 * (K * K - 1.0))/a0, (1.0 - K / Q + K * K)/a0])

        f0_hp = 38.13547087602444
        Q_hp = 0.5003270373238773
        K_hp = np.tan(np.pi * f0_hp / sr)
        a0_hp = 1.0 + K_hp / Q_hp + K_hp * K_hp
        b_hp = np.array([1.0/a0_hp, -2.0/a0_hp, 1.0/a0_hp])
        a_hp = np.array([1.0, (2.0 * (K_hp * K_hp - 1.0))/a0_hp, (1.0 - K_hp / Q_hp + K_hp * K_hp)/a0_hp])
        return b_hs, a_hs, b_hp, a_hp

    def measure_active_speech_lufs(self, audio: np.ndarray) -> float:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if len(audio) < self.sr * 0.4:
            return -70.0
        y_k = signal.lfilter(self.b_hp, self.a_hp, signal.lfilter(self.b_hs, self.a_hs, audio))
        blk_size = int(0.400 * self.sr)
        hop_size = int(0.100 * self.sr)
        n_blocks = (len(y_k) - blk_size) // hop_size
        if n_blocks <= 0:
            return -70.0

        # Cumulative sum approach — O(N) memory, O(N) time, no large temporaries.
        # stride_tricks created a zero-copy VIEW but .astype(float64) materialised the
        # full (n_blocks × blk_size) matrix — 11.9 GiB for a 2.3-hour session. OOM fix.
        y_k_sq = y_k.astype(np.float64) ** 2       # (N,)  — same length as input, safe
        cumsum = np.cumsum(y_k_sq)                  # (N,)  — same length
        blk_ends   = np.arange(n_blocks) * hop_size + blk_size      # last sample (excl)
        blk_starts = blk_ends - blk_size                             # first sample
        sum_end    = cumsum[blk_ends - 1]
        sum_start  = np.concatenate([[0.0], cumsum])[blk_starts]    # 0 when start==0
        energies   = ((sum_end - sum_start) / blk_size).astype(np.float32)

        surviving = energies[energies > 10.0 ** (-70.0 / 10.0)]
        if len(surviving) == 0:
            return -70.0
        mean_u = np.mean(surviving)
        active = surviving[surviving > mean_u * 0.1]
        if len(active) == 0:
            return -70.0
        return float(-0.691 + 10.0 * np.log10(np.mean(active)))

    def limit_peaks(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak <= self.ceiling_linear:
            return audio
        lookahead   = int(0.005 * self.sr)
        rel_samples = int(0.050 * self.sr)
        overshoot = np.maximum(1.0, np.abs(audio) / self.ceiling_linear)
        gain_red  = 1.0 / overshoot

        # Exponential release smoother as a 1-pole IIR run in C via lfilter.
        # Only release (smoothing up) — downward snaps are instant.
        alpha_rel = np.exp(-1.0 / max(1, rel_samples))
        # Forward pass: instant attack, exponential release
        smooth_red = np.empty_like(gain_red)
        curr = 1.0
        for i in range(len(gain_red)):
            tgt = gain_red[i]
            curr = tgt if tgt < curr else curr * alpha_rel + tgt * (1.0 - alpha_rel)
            smooth_red[i] = curr
        if lookahead > 0:
            smooth_red = np.pad(smooth_red[lookahead:], (0, lookahead), mode='edge')
        limited = audio * smooth_red
        return np.clip(limited, -self.ceiling_linear, self.ceiling_linear)

    def normalize_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Normalizes audio array to target LUFS.
        Returns (normalized_audio, applied_gain_db, measured_speech_lufs, final_peak_dbfs).
        """
        meas_lufs = self.measure_active_speech_lufs(audio)
        if meas_lufs < -65.0:
            gain_db = 0.0
        else:
            gain_db = float(np.clip(self.target_lufs - meas_lufs, self.min_gain_db, self.max_gain_db))
        gain_lin = 10.0 ** (gain_db / 20.0)
        norm_audio = audio * gain_lin
        limited = self.limit_peaks(norm_audio)
        final_peak = float(np.max(np.abs(limited)))
        final_peak_db = 20.0 * np.log10(max(1e-9, final_peak))
        return limited.astype(np.float32), gain_db, meas_lufs, final_peak_db


def get_rnnoise_dll_path() -> Optional[str]:
    """Locate rnnoise.dll across PyInstaller bundle, pyrnnoise package, and local directory."""
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, "rnnoise.dll"))
        candidates.append(os.path.join(sys._MEIPASS, "pyrnnoise", "rnnoise.dll"))
    try:
        import pyrnnoise
        candidates.append(os.path.join(os.path.dirname(pyrnnoise.__file__), "rnnoise.dll"))
    except Exception:
        pass
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rnnoise.dll"))
    candidates.append(os.path.join(os.getcwd(), "rnnoise.dll"))

    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


class AIRNNoiseSuppressor:
    """
    Neural Network Audio Noise Suppressor (RNNoise).
    Uses a Deep Recurrent Neural Network (GRU) to isolate human speech and
    suppress ambient room noise, fan whine, computer hum, and background hiss.
    """
    FRAME_SIZE = 480  # 10ms at 48kHz
    SAMPLE_RATE = 48000

    def __init__(self, sr: int = 48000, strength: float = 1.0):
        self.sr = sr
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self._dll_path = get_rnnoise_dll_path()
        self._lib = None
        self._state = None

        if self._dll_path and os.path.isfile(self._dll_path):
            try:
                import ctypes
                self._lib = ctypes.CDLL(self._dll_path)
                self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]
                self._lib.rnnoise_create.restype = ctypes.c_void_p
                self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
                self._lib.rnnoise_process_frame.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                ]
                self._lib.rnnoise_process_frame.restype = ctypes.c_float
                self._state = self._lib.rnnoise_create(None)
            except Exception as e:
                self._lib = None
                self._state = None
                print(f"[-] Warning: Failed to load RNNoise: {e}")

    def __del__(self):
        if self._lib is not None and self._state is not None:
            try:
                self._lib.rnnoise_destroy(self._state)
            except Exception:
                pass
            self._state = None

    @property
    def is_available(self) -> bool:
        return self._state is not None

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a 1D float32 audio chunk through the recurrent neural network.
        Applies wet/dry strength blending.
        """
        if self._state is None or self.strength <= 0.0 or len(chunk) == 0:
            return chunk

        orig_len = len(chunk)
        if self.sr != self.SAMPLE_RATE:
            # resample_poly uses polyphase integer-ratio filtering — ~3× faster than
            # signal.resample (FFT-based) and uses far less memory on long chunks.
            from math import gcd
            g   = gcd(self.SAMPLE_RATE, self.sr)
            up  = self.SAMPLE_RATE // g
            dn  = self.sr          // g
            x   = resample_poly(chunk, up, dn).astype(np.float32)
        else:
            x = chunk.astype(np.float32)

        n_samples = len(x)
        pad_needed = (self.FRAME_SIZE - (n_samples % self.FRAME_SIZE)) % self.FRAME_SIZE
        if pad_needed > 0:
            x_padded = np.pad(x, (0, pad_needed), mode='constant')
        else:
            x_padded = x.copy()

        num_frames = len(x_padded) // self.FRAME_SIZE
        out_padded = np.empty_like(x_padded)

        import ctypes
        c_float_p = ctypes.POINTER(ctypes.c_float)

        for i in range(num_frames):
            frame = x_padded[i * self.FRAME_SIZE : (i + 1) * self.FRAME_SIZE].copy()
            frame *= 32767.0
            ptr = frame.ctypes.data_as(c_float_p)
            self._lib.rnnoise_process_frame(self._state, ptr, ptr)
            out_padded[i * self.FRAME_SIZE : (i + 1) * self.FRAME_SIZE] = (frame / 32767.0)

        clean = out_padded[:n_samples]

        if self.sr != self.SAMPLE_RATE:
            from math import gcd
            g   = gcd(self.sr, self.SAMPLE_RATE)
            up  = self.sr          // g
            dn  = self.SAMPLE_RATE // g
            clean = resample_poly(clean, up, dn).astype(np.float32)
            # Trim or pad back to original length (polyphase can add/remove 1-2 samples)
            if len(clean) > orig_len:
                clean = clean[:orig_len]
            elif len(clean) < orig_len:
                clean = np.pad(clean, (0, orig_len - len(clean)))

        if self.strength < 1.0:
            return (1.0 - self.strength) * chunk + self.strength * clean
        return clean


class EnsembleVocalRestorer:
    """
    Studio Dialogue Restorer with profiles calibrated to match the Track 3 reference standard.
    """
    def __init__(self, profile: str = "t1_room_echo", sr: int = 48000, use_ai_denoise: bool = False, ai_denoise_strength: float = 1.0):
        self.sr = sr
        self.profile = profile
        self.target_peak = 10.0 ** (-1.0 / 20.0)
        self.ai_denoiser = None

        if use_ai_denoise or profile == "ai_rnnoise":
            self.ai_denoiser = AIRNNoiseSuppressor(sr=sr, strength=ai_denoise_strength)

        if profile == "t1_room_echo":
            self.b_hp, self.a_hp = signal.butter(2, 75.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_w, self.a_w = biquad_shelf(+3.5, 180.0, True, sr)
            self.zi_w = signal.lfilter_zi(self.b_w, self.a_w)
            self.b_p, self.a_p = biquad_peaking(-2.2, 2200.0, 1.4, sr)
            self.zi_p = signal.lfilter_zi(self.b_p, self.a_p)
            self.b_a, self.a_a = biquad_shelf(+3.0, 7500.0, False, sr)
            self.zi_a = signal.lfilter_zi(self.b_a, self.a_a)
            self.tail_g = 1.0
        elif profile == "t2_muffled":
            self.b_hp, self.a_hp = signal.butter(2, 80.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_m, self.a_m = biquad_peaking(-6.0, 340.0, 1.3, sr)
            self.zi_m = signal.lfilter_zi(self.b_m, self.a_m)
            self.b_p, self.a_p = biquad_peaking(+3.8, 2200.0, 1.2, sr)
            self.zi_p = signal.lfilter_zi(self.b_p, self.a_p)
            self.b_a, self.a_a = biquad_shelf(+9.5, 4800.0, False, sr)
            self.zi_a = signal.lfilter_zi(self.b_a, self.a_a)
        elif profile == "t4_megaphone":
            self.b_hp, self.a_hp = signal.butter(3, 85.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_w, self.a_w = biquad_shelf(+5.5, 160.0, True, sr)
            self.zi_w = signal.lfilter_zi(self.b_w, self.a_w)
            self.b_m, self.a_m = biquad_peaking(-6.5, 515.0, 1.6, sr)
            self.zi_m = signal.lfilter_zi(self.b_m, self.a_m)
            self.b_p, self.a_p = biquad_peaking(+4.2, 2000.0, 1.3, sr)
            self.zi_p = signal.lfilter_zi(self.b_p, self.a_p)
            self.b_s, self.a_s = biquad_peaking(-3.5, 6200.0, 2.5, sr)
            self.zi_s = signal.lfilter_zi(self.b_s, self.a_s)
            self.b_a, self.a_a = biquad_shelf(+3.0, 8000.0, False, sr)
            self.zi_a = signal.lfilter_zi(self.b_a, self.a_a)
        elif profile in ("t5_headset", "t5_bleed_gate"):
            self.b_hp, self.a_hp = signal.butter(2, 75.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_w, self.a_w = biquad_shelf(+4.5, 145.0, True, sr)
            self.zi_w = signal.lfilter_zi(self.b_w, self.a_w)
            self.b_body, self.a_body = biquad_peaking(+2.0, 185.0, 1.2, sr)
            self.zi_body = signal.lfilter_zi(self.b_body, self.a_body)
            self.b_m, self.a_m = biquad_peaking(-4.5, 450.0, 1.5, sr)
            self.zi_m = signal.lfilter_zi(self.b_m, self.a_m)
            self.b_h, self.a_h = biquad_peaking(-2.5, 3300.0, 2.0, sr)
            self.zi_h = signal.lfilter_zi(self.b_h, self.a_h)
            self.b_a, self.a_a = biquad_shelf(+2.0, 6500.0, False, sr)
            self.zi_a = signal.lfilter_zi(self.b_a, self.a_a)
        elif profile in ("t7_rati_clarity", "t7_rati", "rati_clarity"):
            # Rati (Umbra): High-pass 75 Hz, 185 Hz de-boom cut (-5.0 dB), 420 Hz de-box cut (-4.5 dB), 2400 Hz consonant presence lift (+5.5 dB), 6500 Hz air shelf (+4.0 dB)
            self.b_hp, self.a_hp = signal.butter(2, 75.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_boom, self.a_boom = biquad_peaking(-5.0, 185.0, 1.5, sr)
            self.zi_boom = signal.lfilter_zi(self.b_boom, self.a_boom)
            self.b_box, self.a_box = biquad_peaking(-4.5, 420.0, 1.4, sr)
            self.zi_box = signal.lfilter_zi(self.b_box, self.a_box)
            self.b_pres, self.a_pres = biquad_peaking(+5.5, 2400.0, 1.1, sr)
            self.zi_pres = signal.lfilter_zi(self.b_pres, self.a_pres)
            self.b_air, self.a_air = biquad_shelf(+4.0, 6500.0, False, sr)
            self.zi_air = signal.lfilter_zi(self.b_air, self.a_air)
        elif profile in ("t3_reference", "t3_reference_standard"):
            # Natural broadcast pass: 60 Hz 2nd-order high-pass for sub-rumble + neutral dynamics
            self.b_hp, self.a_hp = signal.butter(2, 60.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_w, self.a_w = None, None
        elif profile == "silence_only":
            self.b_hp, self.a_hp = None, None
            self.b_w, self.a_w = None, None
        elif profile == "ai_rnnoise":
            self.b_hp, self.a_hp = signal.butter(2, 60.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_w, self.a_w = None, None
        else:
            self.b_hp, self.a_hp = signal.butter(2, 75.0 / (sr/2), btype='highpass')
            self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp)
            self.b_w, self.a_w = biquad_shelf(+1.5, 180.0, True, sr)
            self.zi_w = signal.lfilter_zi(self.b_w, self.a_w)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return chunk

        if self.profile == "t1_room_echo":
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y, self.zi_w = signal.lfilter(self.b_w, self.a_w, y, zi=self.zi_w)
            y, self.zi_p = signal.lfilter(self.b_p, self.a_p, y, zi=self.zi_p)
            y, self.zi_a = signal.lfilter(self.b_a, self.a_a, y, zi=self.zi_a)
            
            hop_g = int(0.005 * self.sr)
            win_g = int(0.020 * self.sr)
            # frame_rate: how many grain frames per second
            frame_rate = self.sr / max(1, hop_g)
            n_fr = (len(y) - win_g) // hop_g
            if n_fr > 0:
                rms_fr = np.array([np.sqrt(np.mean(y[i*hop_g:i*hop_g+win_g]**2)) for i in range(n_fr)])
                rms_db = 20.0 * np.log10(rms_fr + 1e-9)
                # Only cut room-echo tails that are WELL below speech (-44 dBFS floor, not -24).
                # Speech consonants regularly dip to -28..-35 dBFS — the old -24 threshold was
                # carving into live speech and causing audible gain pumping / garbling.
                tail_thresh_db = -44.0
                tail_range_db  = 20.0   # fade from -44 to -64 dBFS → 0-to-1 ratio
                tail_max_cut_db = -12.0 # max attenuation applied to pure silence/echo tails
                ratio = np.clip((tail_thresh_db - rms_db) / tail_range_db, 0.0, 1.0)
                gain_db = tail_max_cut_db * ratio
                gain_lin = 10.0 ** (gain_db / 20.0)
                gain_smooth = np.zeros_like(gain_lin)
                # Attack 350 ms: only engage after sustained silence (genuine echo tail).
                # Word gaps in conversation (~100-200 ms) are shorter than this TC so the
                # compressor ignores them completely — fixes the "pumping on every pause" bug.
                # Release 20 ms: snap back to unity gain the instant speech returns.
                a_att = np.exp(-1.0 / (0.350 * frame_rate))
                a_rel = np.exp(-1.0 / (0.020 * frame_rate))
                for i in range(n_fr):
                    t_g = gain_lin[i]
                    if t_g < self.tail_g:
                        # Gain decreasing: use slow attack so we don't snap into speech gaps
                        self.tail_g = a_att * self.tail_g + (1 - a_att) * t_g
                    else:
                        # Gain recovering: use faster release so speech onset isn't clipped
                        self.tail_g = a_rel * self.tail_g + (1 - a_rel) * t_g
                    gain_smooth[i] = self.tail_g
                sample_g = np.interp(np.arange(len(y)), np.arange(n_fr)*hop_g + win_g//2, gain_smooth)
                y = y * np.clip(sample_g, 10.0 ** (tail_max_cut_db / 20.0), 1.0)
            y = np.tanh(y * 1.10) / 1.10

        elif self.profile == "t2_muffled":
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y, self.zi_m = signal.lfilter(self.b_m, self.a_m, y, zi=self.zi_m)
            y, self.zi_p = signal.lfilter(self.b_p, self.a_p, y, zi=self.zi_p)
            y, self.zi_a = signal.lfilter(self.b_a, self.a_a, y, zi=self.zi_a)
            y = np.tanh(y * 1.08) / 1.08

        elif self.profile == "t4_megaphone":
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y, self.zi_w = signal.lfilter(self.b_w, self.a_w, y, zi=self.zi_w)
            y, self.zi_m = signal.lfilter(self.b_m, self.a_m, y, zi=self.zi_m)
            y, self.zi_p = signal.lfilter(self.b_p, self.a_p, y, zi=self.zi_p)
            y, self.zi_s = signal.lfilter(self.b_s, self.a_s, y, zi=self.zi_s)
            y, self.zi_a = signal.lfilter(self.b_a, self.a_a, y, zi=self.zi_a)
            y = np.tanh(y * 1.10) / 1.10

        elif self.profile in ("t5_headset", "t5_bleed_gate"):
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y, self.zi_w = signal.lfilter(self.b_w, self.a_w, y, zi=self.zi_w)
            y, self.zi_body = signal.lfilter(self.b_body, self.a_body, y, zi=self.zi_body)
            y, self.zi_m = signal.lfilter(self.b_m, self.a_m, y, zi=self.zi_m)
            y, self.zi_h = signal.lfilter(self.b_h, self.a_h, y, zi=self.zi_h)
            y, self.zi_a = signal.lfilter(self.b_a, self.a_a, y, zi=self.zi_a)
            y = np.tanh(y * 1.15) / 1.15

        elif self.profile in ("t7_rati_clarity", "t7_rati", "rati_clarity"):
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y, self.zi_boom = signal.lfilter(self.b_boom, self.a_boom, y, zi=self.zi_boom)
            y, self.zi_box = signal.lfilter(self.b_box, self.a_box, y, zi=self.zi_box)
            y, self.zi_pres = signal.lfilter(self.b_pres, self.a_pres, y, zi=self.zi_pres)
            y, self.zi_air = signal.lfilter(self.b_air, self.a_air, y, zi=self.zi_air)
            y = np.tanh(y * 1.10) / 1.10

        elif self.profile in ("t3_reference", "t3_reference_standard", "ai_rnnoise"):
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y = np.tanh(y * 1.05) / 1.05

        elif self.profile == "silence_only":
            y = chunk.copy()

        else:
            y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
            y, self.zi_w = signal.lfilter(self.b_w, self.a_w, y, zi=self.zi_w)

        # Neural AI Noise Suppression applied at the very END of the voice restoration chain
        if self.ai_denoiser is not None:
            y = self.ai_denoiser.process_chunk(y)

        peak = np.max(np.abs(y))
        if peak > 0.01:
            scale = min(1.8, self.target_peak / peak)
            y = y * scale
        return np.clip(y, -self.target_peak, self.target_peak).astype(np.float32)


def export_as_mp3(wav_path: str, mp3_path: Optional[str] = None, bitrate: str = "320k") -> str:
    """
    Convert WAV audio file to high-quality broadcast MP3 using ffmpeg.
    Returns path to created MP3 file.
    """
    if mp3_path is None:
        base, _ = os.path.splitext(wav_path)
        mp3_path = f"{base}.mp3"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg MP3 export failed: {res.stderr}")
    return mp3_path


def process_vocal_restoration_file(
    input_path: str,
    output_path: str,
    profile: str = "t1_room_echo",
    apply_silence_gate: bool = True,
    apply_ai_denoise: bool = False,
    ai_denoise_strength: float = 1.0,
    silence_floor_db: float = -120.0,
    open_thresh_db: float = -34.0,
    hold_ms: float = 280.0,
    preview_sec: Optional[float] = None,
    chunk_sec: float = 30.0,
    export_format: str = "wav",
    mp3_bitrate: str = "320k",
    progress_callback=None,
    cancel_check=None,
    log_func=print
) -> bool:
    """
    Process an audio file with the Dialogue Restorer & Voice Activity Digital Silence Gate.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log_func(f"[*] Opening dialogue track: {input_path}")
    src = sf.SoundFile(input_path, mode='r')
    sr = src.samplerate
    tot_samples = src.frames
    total_sec = tot_samples / sr

    if preview_sec and preview_sec > 0:
        total_sec = min(total_sec, float(preview_sec))
        max_samples = int(total_sec * sr)
        log_func(f"[*] Preview mode: processing first {preview_sec:.0f}s ({format_time(total_sec)})")
    else:
        max_samples = tot_samples

    log_func(f"[*] Audio specs: {sr} Hz | Channels: {src.channels} | Duration: {format_time(total_sec)}")
    log_func(f"[*] Selected Profile: {profile.upper()} (Tuned to Track 3 Reference)")
    if apply_ai_denoise or profile == "ai_rnnoise":
        log_func(f"[*] Neural AI Noise Suppression: ENABLED (Deep RNNoise @ {int(ai_denoise_strength*100)}% strength)")
    else:
        log_func("[*] Neural AI Noise Suppression: DISABLED")

    if apply_silence_gate:
        floor_label = "Pure Digital Silence (-inf dB)" if silence_floor_db <= -90.0 else f"{silence_floor_db:.0f} dBFS floor"
        log_func(f"[*] Voice Activity Gate: ENABLED ({floor_label}, open @ {open_thresh_db:.0f} dBFS, hold {hold_ms:.0f}ms)")
    else:
        log_func("[*] Voice Activity Gate: DISABLED (continuous audio)")

    restorer = EnsembleVocalRestorer(
        profile=profile,
        sr=sr,
        use_ai_denoise=False
    )
    ai_suppressor = AIRNNoiseSuppressor(sr=sr, strength=ai_denoise_strength) if apply_ai_denoise else None

    _LONG_HOLD_PROFILES = {"t6_laptop_fan", "t1_room_echo"}
    gate_hold_ms = 500.0 if profile in _LONG_HOLD_PROFILES else hold_ms
    silence_gate = DialogueSafeSilenceGate(
        sr=sr,
        open_thresh_db=open_thresh_db,
        close_thresh_db=open_thresh_db - 10.0,
        hold_ms=gate_hold_ms,
        rel_ms=120.0,
        lookahead_ms=30.0,
        floor_db=silence_floor_db
    ) if apply_silence_gate else None

    is_mp3 = export_format.lower().endswith("mp3") or output_path.lower().endswith(".mp3")
    final_output_path = output_path
    if is_mp3:
        if not final_output_path.lower().endswith(".mp3"):
            base, _ = os.path.splitext(final_output_path)
            final_output_path = f"{base}.mp3"
        temp_wav_path = final_output_path + ".temp.wav"
        wav_dest = temp_wav_path
    else:
        temp_wav_path = None
        wav_dest = final_output_path

    out_dir = os.path.dirname(os.path.abspath(wav_dest))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dst = sf.SoundFile(
        wav_dest,
        mode='w',
        samplerate=sr,
        channels=1,
        subtype='PCM_24' if sr <= 48000 else 'FLOAT'
    )

    chunk_samples = int(chunk_sec * sr)
    processed = 0
    t0 = time.time()

    try:
        while processed < max_samples:
            if cancel_check and cancel_check():
                log_func("[-] Processing cancelled by user.")
                return False

            to_read = min(chunk_samples, max_samples - processed)
            chunk = src.read(to_read, dtype='float32')
            if len(chunk) == 0:
                break
            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)

            is_last = (processed + len(chunk) >= max_samples)

            # 1. Restorative EQ + Soft Tanh Limiter
            y_tuned = restorer.process_chunk(chunk)

            # 2. Dialogue Silence Gate (removes pauses/reflections before AI denoise)
            if silence_gate is not None:
                y_gated = silence_gate.process_chunk(y_tuned, is_last=is_last)
            else:
                y_gated = y_tuned

            # 3. RNNoise AI Denoise [END of chunk DSP]
            if ai_suppressor is not None:
                y_out = ai_suppressor.process_chunk(y_gated)
            else:
                y_out = y_gated

            dst.write(y_out)


            processed += len(chunk)
            pct = (processed / max_samples) * 100.0
            elapsed = time.time() - t0
            speed = (processed / sr) / max(0.001, elapsed)
            eta = (max_samples - processed) / (sr * max(0.1, speed))

            if progress_callback:
                progress_callback(pct, speed, eta, elapsed)

            if int(pct) % 5 == 0 or processed == max_samples:
                log_func(f"[*] Progress: {pct:5.1f}% | Speed: {speed:5.1f}x | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")

    finally:
        src.close()
        dst.close()

    if is_mp3 and temp_wav_path:
        log_func(f"[*] Encoding to high-quality {mp3_bitrate} broadcast MP3 via FFmpeg...")
        try:
            export_as_mp3(temp_wav_path, final_output_path, bitrate=mp3_bitrate)
            log_func(f"[+] MP3 export complete: {final_output_path}")
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    total_time = time.time() - t0
    log_func(f"[+] Dialogue processing finished in {format_time(total_time)} ({total_sec/max(0.001, total_time):.1f}x real-time)!")
    log_func(f"[+] Output file saved: {final_output_path}")
    return True


def calibrate_laptop_fan_noise(file_path: str, sr: int = 48000, scan_sec: float = 60.0) -> np.ndarray:
    """Profile laptop fan motor & chassis noise from the quietest pause segments.

    Uses a robust multi-window median approach: scans the first scan_sec seconds
    in 0.5-second hops, ranks all windows by RMS, takes the 10 quietest ones
    (true pause/fan-only moments), and median-averages their STFT magnitude spectra.
    This avoids the old single-best-block method which could accidentally land on a
    talking segment in a session where Magnus speaks through most of the recording.
    """
    try:
        data, file_sr = sf.read(file_path, frames=int(scan_sec * sr), dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]

        b_hp, a_hp = signal.butter(3, 85.0 / (sr / 2), btype='highpass')
        data_hp = signal.lfilter(b_hp, a_hp, data)

        win_len  = int(0.5 * sr)   # 0.5-second analysis windows
        step_len = int(0.25 * sr)  # 0.25-second hop (50% overlap)
        n_steps  = max(1, (len(data_hp) - win_len) // step_len)

        rms_list = []
        for i in range(n_steps):
            seg = data_hp[i * step_len : i * step_len + win_len]
            rms_list.append((np.sqrt(np.mean(seg**2)), i * step_len))

        # Sort by RMS, take up to 10 quietest windows that are also below -40 dBFS
        rms_list.sort(key=lambda x: x[0])
        quiet_specs = []
        for rms_val, start in rms_list:
            if len(quiet_specs) >= 10:
                break
            if rms_val < 1e-4:  # very nearly digital silence → skip (no useful info)
                continue
            if 20.0 * np.log10(rms_val + 1e-9) > -28.0:
                # This window is too loud — likely a talking segment; stop here.
                # (List is sorted ascending so all remaining will be louder too.)
                break
            seg = data_hp[start : start + win_len]
            _, _, z_seg = signal.stft(seg, fs=sr, nperseg=1024, noverlap=768)
            quiet_specs.append(np.abs(z_seg))

        if not quiet_specs:
            # Fallback: use the single quietest window regardless
            _, start = rms_list[0]
            seg = data_hp[start : start + win_len]
            _, _, z_seg = signal.stft(seg, fs=sr, nperseg=1024, noverlap=768)
            return np.mean(np.abs(z_seg), axis=1, keepdims=True)

        # Median across all collected quiet spectra (robust against outliers/voice leakage)
        stacked = np.stack([np.mean(s, axis=1) for s in quiet_specs], axis=0)  # (n, freq_bins)
        return np.median(stacked, axis=0, keepdims=False).reshape(-1, 1).astype(np.float32)

    except Exception:
        return np.zeros((513, 1), dtype=np.float32)


def process_laptop_mic_file(
    input_path: str,
    output_path: str,
    preview_sec: Optional[float] = None,
    alpha: float = 1.2,
    floor_db: float = -24.0,
    desk_cut_db: float = -5.0,
    hollow_cut_db: float = -3.5,
    warmth_db: float = 2.5,
    presence_db: float = 4.5,
    apply_ai_denoise: bool = False,
    ai_denoise_strength: float = 1.0,
    progress_callback=None,
    cancel_check=None,
    chunk_sec: float = 30.0,
    log_func=print
) -> bool:
    """
    Process a single audio file through the Laptop Mic Restoration Pipeline.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log_func(f"[*] Opening laptop mic file: {input_path}")
    src = sf.SoundFile(input_path, mode='r')
    sr = src.samplerate
    tot_samples = src.frames
    total_sec = tot_samples / sr

    if preview_sec and preview_sec > 0:
        total_sec = min(total_sec, float(preview_sec))
        max_samples = int(total_sec * sr)
        log_func(f"[*] Preview mode: processing first {preview_sec:.0f}s ({format_time(total_sec)})")
    else:
        max_samples = tot_samples

    log_func(f"[*] Audio specs: {sr} Hz | Channels: {src.channels} | Duration: {format_time(total_sec)}")
    log_func("[*] Profiling laptop fan motor & chassis noise...")
    noise_mag = calibrate_laptop_fan_noise(input_path, sr=sr)

    enhancer = LaptopMicEnhancer(
        sr=sr,
        noise_mag=noise_mag,
        alpha=alpha,
        floor_db=floor_db,
        desk_cut_db=desk_cut_db,
        hollow_cut_db=hollow_cut_db,
        warmth_db=warmth_db,
        presence_db=presence_db
    )

    ai_suppressor = AIRNNoiseSuppressor(sr=sr, strength=ai_denoise_strength) if apply_ai_denoise else None
    if ai_suppressor and ai_suppressor.is_available:
        log_func(f"[*] Neural AI Noise Suppression: ENABLED (Deep RNNoise @ {int(ai_denoise_strength*100)}% strength)")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dst = sf.SoundFile(
        output_path,
        mode='w',
        samplerate=sr,
        channels=1,
        subtype='PCM_24' if sr <= 48000 else 'FLOAT'
    )

    chunk_samples = int(chunk_sec * sr)
    processed = 0
    t0 = time.time()

    try:
        while processed < max_samples:
            if cancel_check and cancel_check():
                log_func("[-] Processing cancelled by user.")
                return False

            to_read = min(chunk_samples, max_samples - processed)
            chunk = src.read(to_read, dtype='float32')
            if len(chunk) == 0:
                break
            if chunk.ndim > 1:
                chunk = chunk[:, 0]

            is_last = (processed + len(chunk) >= max_samples)
            enhanced = enhancer.process_chunk(chunk, is_last=is_last)
            if ai_suppressor:
                enhanced = ai_suppressor.process_chunk(enhanced)
            dst.write(enhanced)

            processed += len(chunk)
            pct = (processed / max_samples) * 100.0
            curr_sec = processed / sr
            elapsed = time.time() - t0
            speed = curr_sec / max(0.001, elapsed)
            eta_sec = (max_samples - processed) / (sr * max(0.1, speed))

            if progress_callback:
                progress_callback(pct, curr_sec, total_sec, speed, eta_sec)

    finally:
        src.close()
        dst.close()

    total_time = time.time() - t0
    log_func(f"[+] Complete! Processed {format_time(processed/sr)} in {total_time:.1f}s ({processed/sr/max(0.001, total_time):.1f}x real-time)")
    log_func(f"[+] Output saved: {output_path}")
    return True


CAMPAIGN_CONFIGS = {
    "sw5e": {
        "id": "sw5e",
        "name": "Star Wars 5e (SW5E)",
        "icon": "🌌",
        "num_tracks": 6,
        "slots": {
            1: {
                "player": "Robin",
                "character": "Cratebreaker",
                "label": "Slot 1 (Robin):",
                "speaker": "Robin (Cratebreaker)",
                "profile_id": "t1_room_echo",
                "profile_name": "Robin: Room Echo & Slapback Suppressor",
                "active": True,
                "aliases": ("robin", "cratebreaker")
            },
            2: {
                "player": "Tino",
                "character": "It'Mir",
                "label": "Slot 2 (Tino):",
                "speaker": "Tino (It'Mir)",
                "profile_id": "t2_muffled",
                "profile_name": "Tino: Blanket De-Muffler (Air & Clarity)",
                "active": True,
                "aliases": ("tino", "itmir", "it'mir")
            },
            3: {
                "player": "Blu",
                "character": "Caelen",
                "label": "Slot 3 (Blu):",
                "speaker": "Blu (Caelen)",
                "profile_id": "t3_reference",
                "profile_name": "Blu: Reference Standard (Natural Dialogue)",
                "active": True,
                "aliases": ("blu", "caelen")
            },
            4: {
                "player": "Marc",
                "character": "GM",
                "label": "Slot 4 (Marc - GM):",
                "speaker": "Marc (GM)",
                "profile_id": "t4_megaphone",
                "profile_name": "Marc: Megaphone & Plosive De-Resonator",
                "active": True,
                "aliases": ("marc", "marcus", "gm")
            },
            5: {
                "player": "Mathew",
                "character": "Salova",
                "label": "Slot 5 (Mathew):",
                "speaker": "Mathew (Salova)",
                "profile_id": "t5_bleed_gate",
                "profile_name": "Mathew: Multitrack Bleed Gate + Headset Polish",
                "active": True,
                "aliases": ("mathew", "matthew", "salova")
            },
            6: {
                "player": "Timmy",
                "character": "Belial",
                "label": "Slot 6 (Timmy):",
                "speaker": "Timmy (Belial)",
                "profile_id": "t6_laptop_fan",
                "profile_name": "Timmy: Laptop Mic Fan Whine & Desk Filter",
                "active": True,
                "aliases": ("timmy", "belial")
            },
        },
        "prompt": "Star Wars 5e tabletop RPG session. Characters: Caelen, Cratebreaker, Belial, It'Mir, Salova. GM: Marc. Setting: Kashyyyk, Coruscant, Nar Shaddaa, Sith, Jedi, Lord Kyrix, Xavier, Wookiee, Ewok, Jawa, Devaronian, lightsaber, vibrodagger, d20, Nat 20."
    },
    "red": {
        "id": "red",
        "name": "Cyberpunk RED",
        "icon": "🦾",
        "num_tracks": 5,
        "slots": {
            1: {
                "player": "Robin",
                "character": "QBall",
                "label": "Slot 1 (Robin):",
                "speaker": "Robin (QBall)",
                "profile_id": "t1_room_echo",
                "profile_name": "Robin: Room Echo & Slapback Suppressor",
                "active": True,
                "aliases": ("robin", "qball", "q-ball")
            },
            2: {
                "player": "Blu",
                "character": "GM",
                "label": "Slot 2 (Blu - GM):",
                "speaker": "Blu (GM)",
                "profile_id": "t3_reference",
                "profile_name": "Blu: Reference Standard (Natural Dialogue)",
                "active": True,
                "aliases": ("blu", "gm")
            },
            3: {
                "player": "Marc",
                "character": "Ryu",
                "label": "Slot 3 (Marc):",
                "speaker": "Marc (Ryu)",
                "profile_id": "t4_megaphone",
                "profile_name": "Marc: Megaphone & Plosive De-Resonator",
                "active": True,
                "aliases": ("marc", "marcus", "ryu")
            },
            4: {
                "player": "Rati",
                "character": "Umbra",
                "label": "Slot 4 (Rati):",
                "speaker": "Rati (Umbra)",
                "profile_id": "t7_rati_clarity",
                "profile_name": "Rati: Low-Mid De-Boom & Consonant Clarity",
                "active": True,
                "aliases": ("rati", "umbra")
            },
            5: {
                "player": "Timmy",
                "character": "Magnus",
                "label": "Slot 5 (Timmy):",
                "speaker": "Timmy (Magnus)",
                "profile_id": "t6_laptop_fan",
                "profile_name": "Timmy: Laptop Mic Fan Whine & Desk Filter",
                "active": True,
                "aliases": ("timmy", "magnus")
            },
            6: {
                "player": "-",
                "character": "-",
                "label": "Slot 6 (Inactive):",
                "speaker": "Slot 6 (Inactive)",
                "profile_id": "skip",
                "profile_name": "Skip Track (Do Not Process)",
                "active": False,
                "aliases": ()
            },
        },
        "prompt": "Cyberpunk RED tabletop RPG session. Characters: QBall, Ryu, Umbra, Magnus. GM: Blu. Setting: Night City, Edgerunners, Arasaka, Militech, Trauma Team, Fixer, Solo, Netrunner, Tech, Rockerboy, Nomad, cyberware, chrome, cyberdeck, eurodollars, eddies, d10, Critical Success."
    }
}

DEFAULT_SLOT_PROFILES = CAMPAIGN_CONFIGS["sw5e"]["slots"]


def get_campaign_config(mode: str = "sw5e") -> Dict[str, Any]:
    """Retrieve slot presets and metadata for sw5e or red campaign."""
    clean = mode.lower().strip() if mode else "sw5e"
    if "red" in clean or "cyber" in clean:
        return CAMPAIGN_CONFIGS["red"]
    return CAMPAIGN_CONFIGS["sw5e"]


PROFILE_CHOICES = [
    "Robin: Room Echo & Slapback Suppressor",
    "Tino: Blanket De-Muffler (Air & Clarity)",
    "Blu: Reference Standard (Natural Dialogue)",
    "Marc: Megaphone & Plosive De-Resonator",
    "Mathew: Multitrack Bleed Gate + Headset Polish",
    "Timmy: Laptop Mic Fan Whine & Desk Filter",
    "Rati: Low-Mid De-Boom & Consonant Clarity",
    "🧠 AI Neural Speech Denoise (RNNoise Voice Isolator)",
    "Digital Silence Gate Only (Transparent EQ)",
    "Skip Track (Do Not Process)"
]


def profile_str_to_id(choice_str: str) -> str:
    s = choice_str.lower()
    if "rati" in s or "umbra" in s or "de-boom" in s or "track 7" in s:
        return "t7_rati_clarity"
    elif "robin" in s or "track 1" in s or "room echo" in s or "slapback" in s:
        return "t1_room_echo"
    elif "tino" in s or "track 2" in s or "blanket" in s or "muffled" in s:
        return "t2_muffled"
    elif "blu" in s or "track 3" in s or "reference" in s:
        return "t3_reference"
    elif "marc" in s or "marcus" in s or "track 4" in s or "megaphone" in s or "plosive" in s:
        return "t4_megaphone"
    elif "mathew" in s or "matthew" in s or "track 5" in s or "bleed" in s:
        return "t5_bleed_gate"
    elif "timmy" in s or "track 6" in s or "laptop" in s or "fan" in s:
        return "t6_laptop_fan"
    elif "ai neural" in s or "rnnoise" in s or "voice isolator" in s:
        return "ai_rnnoise"
    elif "silence" in s:
        return "silence_only"
    elif "skip" in s:
        return "skip"
    return "t3_reference"


def profile_id_to_str(prof_id: str) -> str:
    mapping = {
        "t1_room_echo": "Robin: Room Echo & Slapback Suppressor",
        "t2_muffled": "Tino: Blanket De-Muffler (Air & Clarity)",
        "t3_reference": "Blu: Reference Standard (Natural Dialogue)",
        "t3_reference_standard": "Blu: Reference Standard (Natural Dialogue)",
        "t4_megaphone": "Marc: Megaphone & Plosive De-Resonator",
        "t5_bleed_gate": "Mathew: Multitrack Bleed Gate + Headset Polish",
        "t6_laptop_fan": "Timmy: Laptop Mic Fan Whine & Desk Filter",
        "t7_rati_clarity": "Rati: Low-Mid De-Boom & Consonant Clarity",
        "t7_rati": "Rati: Low-Mid De-Boom & Consonant Clarity",
        "rati_clarity": "Rati: Low-Mid De-Boom & Consonant Clarity",
        "ai_rnnoise": "🧠 AI Neural Speech Denoise (RNNoise Voice Isolator)",
        "silence_only": "Digital Silence Gate Only (Transparent EQ)",
        "skip": "Skip Track (Do Not Process)"
    }
    return mapping.get(prof_id, "Blu: Reference Standard (Natural Dialogue)")


def auto_detect_session_tracks(source: Union[str, List[str]], mode: str = "sw5e") -> Dict[int, str]:
    """
    Detect session audio tracks and map them to slots 1-6 according to the active campaign mode (sw5e or red).
    'source' can be a folder path or a list of file paths.
    """
    audio_exts = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.wma')
    video_exts = ('.mkv', '.mp4', '.mov', '.m4v', '.avi', '.webm')
    candidate_paths = []

    clean_mode = mode.lower().strip() if mode else "sw5e"
    is_red = ("red" in clean_mode or "cyber" in clean_mode)

    # Check if a single video container was passed directly
    if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(video_exts):
        try:
            from video_ingest import extract_video_audio_stems
            return extract_video_audio_stems(source, campaign_mode=clean_mode)
        except Exception as e:
            print(f"[-] Video extraction error: {e}")

    if isinstance(source, str):
        if os.path.isdir(source):
            for entry in os.listdir(source):
                full_p = os.path.join(source, entry)
                if os.path.isfile(full_p) and entry.lower().endswith(audio_exts):
                    candidate_paths.append(full_p)
            # If no audio files found in directory, check for video containers
            if not candidate_paths:
                vids = [os.path.join(source, e) for e in os.listdir(source) if os.path.isfile(os.path.join(source, e)) and e.lower().endswith(video_exts)]
                if vids:
                    try:
                        from video_ingest import extract_video_audio_stems
                        return extract_video_audio_stems(vids[0], campaign_mode=clean_mode)
                    except Exception as e:
                        print(f"[-] Video extraction error: {e}")
        elif os.path.isfile(source):
            candidate_paths = [source]
    elif isinstance(source, (list, tuple)):
        for item in source:
            if isinstance(item, str):
                if item.lower().endswith(video_exts) and os.path.isfile(item):
                    try:
                        from video_ingest import extract_video_audio_stems
                        return extract_video_audio_stems(item, campaign_mode=clean_mode)
                    except Exception as e:
                        print(f"[-] Video extraction error: {e}")
                elif item.lower().endswith(audio_exts):
                    candidate_paths.append(item)

    slots: Dict[int, str] = {}
    ignored_keywords = ('cleaned', 'enhanced', 'mastered', 'preview', 'temp', 'tuned', 'digital_silence', 'digitalsilence')

    if is_red:
        name_map = {
            1: ('robin', 'qball', 'q-ball'),
            2: ('blu',),
            3: ('marc', 'marcus', 'ryu'),
            4: ('rati', 'umbra'),
            5: ('timmy', 'magnus')
        }
        max_slots = 5
    else:
        name_map = {
            1: ('robin', 'cratebreaker'),
            2: ('tino', 'itmir', "it'mir"),
            3: ('blu', 'caelen'),
            4: ('marc', 'marcus'),
            5: ('mathew', 'matthew', 'salova'),
            6: ('timmy', 'belial')
        }
        max_slots = 6

    for p in candidate_paths:
        fname = os.path.basename(p)
        fname_lower = fname.lower()
        if any(kw in fname_lower for kw in ignored_keywords):
            continue

        slot_num = None

        # Check speaker name / character alias matches first
        for s_idx, aliases in name_map.items():
            if any(re.search(rf'\b{re.escape(alias)}\b', fname_lower) or alias in fname_lower for alias in aliases):
                slot_num = s_idx
                break

        # If not matched by name, match by standard track index patterns
        if slot_num is None:
            patterns = [
                r'(?:track|a|spk|speaker|ch|mic|player)[_\-\s]*0?([1-6])(?!\d)',
                r'[_\-\s]0?([1-6])[_\-\.\s]',
                r'\b0?([1-6])\b',
                r'0?([1-6])(?=\.[^.]+$)'
            ]
            for pat in patterns:
                m = re.search(pat, fname, re.IGNORECASE)
                if m:
                    val = int(m.group(1))
                    if 1 <= val <= max_slots:
                        slot_num = val
                        break
        
        if slot_num and slot_num not in slots:
            slots[slot_num] = p

    return slots


def process_automated_session(
    slots: Dict[int, Union[str, Dict[str, any]]],
    output_dir: str,
    export_format: str = "mp3",
    mp3_bitrate: str = "320k",
    apply_silence_gate: bool = True,
    apply_ai_denoise: bool = False,
    ai_denoise_strength: float = 1.0,
    apply_speech_normalization: bool = True,
    target_lufs: float = -18.0,
    peak_ceiling_db: float = -1.0,
    silence_floor_db: float = -120.0,
    open_thresh_db: float = -34.0,
    hold_ms: float = 280.0,
    preview_sec: Optional[float] = None,
    chunk_sec: float = 30.0,
    progress_callback=None,
    cancel_check=None,
    log_func=print
) -> Dict[str, str]:
    """
    Automated multitrack session mastering orchestrator.
    Processes each loaded track according to its assigned profile:
    - Slot 5: Multitrack Bleed Gate with cross-track reference matrix + headset polish
    - Slot 1: Room Echo & Slapback Suppressor
    - Slot 2: Blanket De-Muffler
    - Slot 3: Reference Standard
    - Slot 4: Megaphone & Plosive De-Resonator
    - Slot 6: Laptop Mic Fan Whine & Desk Filter
    All tracks receive DialogueSafeSilenceGate for zero vocal cutoff and pure digital silence in pauses.
    Optionally applies Deep Neural AI Noise Suppression (RNNoise) across all tracks.
    Optionally normalizes active speech loudness across all tracks (ITU-R BS.1770-4) with True Peak limiting.
    """
    active_slots = {}
    for k, v in slots.items():
        if isinstance(v, dict):
            p = v.get("path")
            prof = v.get("profile", DEFAULT_SLOT_PROFILES.get(k, {}).get("id", "t3_reference"))
        else:
            p = v
            prof = DEFAULT_SLOT_PROFILES.get(k, {}).get("id", "t3_reference")
        if p and os.path.isfile(p) and prof != "skip":
            active_slots[k] = {"path": p, "profile": prof}

    if not active_slots:
        log_func("[-] No valid audio tracks found to process.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    is_mp3 = export_format.lower().endswith("mp3")
    ext = "mp3" if is_mp3 else "wav"
    out_results = {}

    ai_label = f"ENABLED ({int(ai_denoise_strength*100)}% RNNoise)" if apply_ai_denoise else "DISABLED"
    norm_label = f"ENABLED (Target: {target_lufs:.1f} LUFS, Ceiling: {peak_ceiling_db:.1f} dBFS)" if apply_speech_normalization else "DISABLED"

    log_func("=" * 70)
    log_func(f"🚀 STARTING AUTOMATED SESSION MASTERING: {len(active_slots)} TRACK(S)")
    log_func(f"Output Directory: {output_dir}")
    log_func(f"Format: {export_format.upper()} ({mp3_bitrate} CBR if MP3) | Silence Gate: {'ENABLED' if apply_silence_gate else 'DISABLED'} | AI Denoise: {ai_label} | Speech Match: {norm_label}")
    if preview_sec and preview_sec > 0:
        log_func(f"⚡ PREVIEW MODE: Processing first {preview_sec:.0f}s ({format_time(preview_sec)})")
    log_func("=" * 70)

    # Order slots: bleed gate (needs all files, must run first) then the rest
    bleed_gate_slots  = [s for s in sorted(active_slots.keys()) if active_slots[s]["profile"] == "t5_bleed_gate"]
    parallel_slots    = [s for s in sorted(active_slots.keys()) if active_slots[s]["profile"] != "t5_bleed_gate"]
    sorted_slots      = bleed_gate_slots + parallel_slots
    num_tracks        = len(sorted_slots)
    t_session_start   = time.time()

    # Determine reference files for multitrack bleed gating
    all_raw_files = [active_slots[k]["path"] for k in sorted(active_slots.keys())]

    # --- Thread-safe logging ---
    import threading, queue as _queue
    _log_queue: "queue.Queue[Optional[str]]" = _queue.Queue()
    _log_lock = threading.Lock()

    def _safe_log(msg: str):
        with _log_lock:
            log_func(msg)

    # Cancelled flag (threading.Event is thread-safe)
    _cancelled = threading.Event()

    def _cancel_check_thread() -> bool:
        if _cancelled.is_set():
            return True
        if cancel_check and cancel_check():
            _cancelled.set()
            return True
        return False

    # Step 1: bleed-gate slots sequentially (they share the AudioSource)
    for track_idx_0, slot_num in enumerate(bleed_gate_slots):
        if _cancel_check_thread():
            _safe_log("[-] Session processing cancelled by user.")
            return out_results

        slot_info = active_slots[slot_num]
        in_path = slot_info["path"]
        profile_id = slot_info["profile"]
        base_name = os.path.splitext(os.path.basename(in_path))[0]

        if preview_sec and preview_sec > 0:
            out_filename = f"{base_name}_Mastered_Preview.{ext}"
        else:
            out_filename = f"{base_name}_Mastered.{ext}"
        final_out_path = os.path.join(output_dir, out_filename)

        log_func(f"\n[{track_idx_0 + 1}/{num_tracks}] Processing Slot {slot_num}: {os.path.basename(in_path)}")
        log_func(f"    Profile: {profile_id_to_str(profile_id)}")
        log_func(f"    Target:  {out_filename}")

        # Open input to get specs
        src_info = sf.SoundFile(in_path, mode='r')
        sr = src_info.samplerate
        tot_samples = src_info.frames
        src_info.close()

        max_samples = min(tot_samples, int(preview_sec * sr)) if (preview_sec and preview_sec > 0) else tot_samples
        total_sec = max_samples / sr

        # Setup audio sink (Direct ffmpeg pipe or SoundFile)
        # Setup audio sink (Direct ffmpeg pipe or SoundFile, or stage1 intermediate WAV if normalizing)
        proc = None
        dst = None
        temp_wav = None
        stage1_wav = None

        if apply_speech_normalization:
            stage1_wav = final_out_path + ".stage1.wav"
            dst = sf.SoundFile(stage1_wav, mode='w', samplerate=sr, channels=1, subtype='PCM_24' if sr <= 48000 else 'FLOAT')
        elif is_mp3:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'f32le',
                '-ar', str(sr),
                '-ac', '1',
                '-i', 'pipe:0',
                '-codec:a', 'libmp3lame',
                '-b:a', mp3_bitrate,
                final_out_path
            ]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log_func(f"[-] Note: Direct FFmpeg pipe failed ({e}). Writing temporary WAV instead.")
                temp_wav = final_out_path + ".temp.wav"
                dst = sf.SoundFile(temp_wav, mode='w', samplerate=sr, channels=1, subtype='PCM_24' if sr <= 48000 else 'FLOAT')
        else:
            dst = sf.SoundFile(final_out_path, mode='w', samplerate=sr, channels=1, subtype='PCM_24' if sr <= 48000 else 'FLOAT')

        def write_sink(audio_chunk: np.ndarray):
            if proc is not None:
                proc.stdin.write(audio_chunk.astype(np.float32).tobytes())
            elif dst is not None:
                dst.write(audio_chunk)

        def close_sink():
            nonlocal proc, dst, temp_wav
            if proc is not None:
                try:
                    proc.stdin.close()
                    proc.wait()
                except Exception:
                    pass
                proc = None
            if dst is not None:
                try:
                    dst.close()
                except Exception:
                    pass
                dst = None
            if temp_wav and os.path.isfile(temp_wav):
                try:
                    export_as_mp3(temp_wav, final_out_path, bitrate=mp3_bitrate)
                finally:
                    if os.path.isfile(temp_wav):
                        try:
                            os.remove(temp_wav)
                        except Exception:
                            pass
                temp_wav = None

        _LONG_HOLD_PROFILES = {"t6_laptop_fan", "t1_room_echo"}
        gate_hold_ms = 500.0 if profile_id in _LONG_HOLD_PROFILES else hold_ms
        silence_gate = DialogueSafeSilenceGate(
            sr=sr,
            open_thresh_db=open_thresh_db,
            close_thresh_db=open_thresh_db - 10.0,
            hold_ms=gate_hold_ms,
            rel_ms=120.0,
            lookahead_ms=30.0,
            floor_db=silence_floor_db
        ) if apply_silence_gate else None

        chunk_samples = int(chunk_sec * sr)
        processed = 0
        t_track_start = time.time()

        ai_suppressor = AIRNNoiseSuppressor(sr=sr, strength=ai_denoise_strength) if apply_ai_denoise else None
        if ai_suppressor and ai_suppressor.is_available:
            log_func(f"    🧠 Neural AI Speech Denoise: ACTIVE ({int(ai_denoise_strength*100)}% strength)")

        try:
            if profile_id == "t5_bleed_gate" and len(all_raw_files) > 1:
                target_0idx = all_raw_files.index(in_path)
                m_src = AudioSource(all_raw_files)
                log_func("    🔍 Auto-calibrating multitrack bleed & vocal levels from audio...")
                calibrator = BleedCalibrator(target_idx=target_0idx, n_tracks=m_src.n_tracks, sr=sr)
                ref_thresh, tgt_thresh, noise_floor, speech_level = calibrator.calibrate(
                    m_src,
                    max_scan_sec=120.0,
                    sensitivity=1.0
                )
                log_func(f"    📊 Auto-Bleed Calibrated: Target Speech {20*math.log10(max(1e-9, speech_level)):.1f}dBFS, Noise {20*math.log10(max(1e-9, noise_floor)):.1f}dBFS")
                gate_engine = StreamingMultitrackBleedGate(
                    sr=sr,
                    target_idx=target_0idx,
                    n_tracks=m_src.n_tracks,
                    match_cutoff=0.42,
                    sensitivity=1.0,
                    floor_db=-60.0,
                    ref_speech_thresh=ref_thresh,
                    tgt_speech_thresh=tgt_thresh,
                    enhance=True
                )
                try:
                    while processed < max_samples:
                        if cancel_check and cancel_check():
                            log_func("[-] Processing cancelled by user.")
                            break
                        to_read = min(chunk_samples, max_samples - processed)
                        raw_chunk = m_src.read_chunk(to_read)
                        if raw_chunk is None or raw_chunk.shape[1] == 0:
                            break
                        processed_chunk, _ = gate_engine.process_chunk(raw_chunk)
                        is_last = (processed + len(processed_chunk) >= max_samples)
                        if silence_gate:
                            gated_chunk = silence_gate.process_chunk(processed_chunk, is_last=is_last)
                        else:
                            gated_chunk = processed_chunk
                        if ai_suppressor:
                            final_chunk = ai_suppressor.process_chunk(gated_chunk)
                        else:
                            final_chunk = gated_chunk
                        write_sink(final_chunk)

                        processed += len(final_chunk)
                        pct = (processed / max_samples) * 100.0
                        total_pct = ((track_idx_0 + (processed / max_samples)) / num_tracks) * 100.0
                        elapsed = time.time() - t_track_start
                        speed = (processed / sr) / max(0.001, elapsed)
                        eta = (max_samples - processed) / (sr * max(0.1, speed))
                        if progress_callback:
                            progress_callback(track_idx_0 + 1, num_tracks, out_filename, pct, total_pct, speed, format_time(eta), f"Processing Slot {slot_num}: {out_filename}")
                finally:
                    m_src.close()

            elif profile_id == "t6_laptop_fan":
                log_func("    Profiling laptop fan motor & chassis noise...")
                noise_mag = calibrate_laptop_fan_noise(in_path, sr=sr)
                enhancer = LaptopMicEnhancer(
                    sr=sr,
                    noise_mag=noise_mag,
                    alpha=1.2,          # was 1.6 — reduced to cut musical noise artefacts
                    floor_db=-24.0,
                    desk_cut_db=-5.0,
                    hollow_cut_db=-3.5,
                    warmth_db=2.5,
                    presence_db=4.5
                )
                src = sf.SoundFile(in_path, mode='r')
                try:
                    while processed < max_samples:
                        if cancel_check and cancel_check():
                            log_func("[-] Processing cancelled by user.")
                            break
                        to_read = min(chunk_samples, max_samples - processed)
                        chunk = src.read(to_read, dtype='float32')
                        if len(chunk) == 0:
                            break
                        if chunk.ndim > 1:
                            chunk = chunk[:, 0]
                        is_last = (processed + len(chunk) >= max_samples)
                        enhanced = enhancer.process_chunk(chunk, is_last=is_last)
                        if silence_gate:
                            gated_chunk = silence_gate.process_chunk(enhanced, is_last=is_last)
                        else:
                            gated_chunk = enhanced
                        if ai_suppressor:
                            final_chunk = ai_suppressor.process_chunk(gated_chunk)
                        else:
                            final_chunk = gated_chunk
                        write_sink(final_chunk)

                        processed += len(chunk)
                        pct = (processed / max_samples) * 100.0
                        total_pct = ((track_idx_0 + (processed / max_samples)) / num_tracks) * 100.0
                        elapsed = time.time() - t_track_start
                        speed = (processed / sr) / max(0.001, elapsed)
                        eta = (max_samples - processed) / (sr * max(0.1, speed))
                        if progress_callback:
                            progress_callback(track_idx_0 + 1, num_tracks, out_filename, pct, total_pct, speed, format_time(eta), f"Processing Slot {slot_num}: {out_filename}")
                finally:
                    src.close()

            else:
                restorer = EnsembleVocalRestorer(
                    profile=profile_id,
                    sr=sr,
                    use_ai_denoise=False
                )
                src = sf.SoundFile(in_path, mode='r')
                try:
                    while processed < max_samples:
                        if cancel_check and cancel_check():
                            log_func("[-] Processing cancelled by user.")
                            break
                        to_read = min(chunk_samples, max_samples - processed)
                        chunk = src.read(to_read, dtype='float32')
                        if len(chunk) == 0:
                            break
                        if chunk.ndim > 1:
                            chunk = np.mean(chunk, axis=1)
                        is_last = (processed + len(chunk) >= max_samples)
                        tuned = restorer.process_chunk(chunk)
                        if silence_gate:
                            gated_chunk = silence_gate.process_chunk(tuned, is_last=is_last)
                        else:
                            gated_chunk = tuned
                        if ai_suppressor:
                            final_chunk = ai_suppressor.process_chunk(gated_chunk)
                        else:
                            final_chunk = gated_chunk
                        write_sink(final_chunk)


                        processed += len(chunk)
                        pct = (processed / max_samples) * 100.0
                        total_pct = ((track_idx_0 + (processed / max_samples)) / num_tracks) * 100.0
                        elapsed = time.time() - t_track_start
                        speed = (processed / sr) / max(0.001, elapsed)
                        eta = (max_samples - processed) / (sr * max(0.1, speed))
                        if progress_callback:
                            progress_callback(track_idx_0 + 1, num_tracks, out_filename, pct, total_pct, speed, format_time(eta), f"Processing Slot {slot_num}: {out_filename}")
                finally:
                    src.close()

        finally:
            close_sink()

        # Active Speech Loudness Normalization & True Peak Limiter pass
        if apply_speech_normalization and stage1_wav and os.path.isfile(stage1_wav):
            if cancel_check and cancel_check():
                if os.path.isfile(stage1_wav):
                    try:
                        os.remove(stage1_wav)
                    except Exception:
                        pass
                break
            try:
                log_func(f"    ⚖️ Normalizing Speech Loudness (Target: {target_lufs:.1f} LUFS, Ceiling: {peak_ceiling_db:.1f} dBFS)...")
                stage1_data, s1_sr = sf.read(stage1_wav, dtype='float32')
                normalizer = BroadcastSpeechNormalizer(
                    sr=s1_sr,
                    target_lufs=target_lufs,
                    peak_ceiling_db=peak_ceiling_db
                )
                norm_audio, gain_db, meas_lufs, final_peak = normalizer.normalize_audio(stage1_data)
                log_func(f"    [+] Speech Level: Measured {meas_lufs:.1f} LUFS -> Applied {gain_db:+.1f} dB (Target: {target_lufs:.1f} LUFS | Peak: {final_peak:.1f} dBFS)")

                if is_mp3:
                    temp_norm_wav = final_out_path + ".norm.wav"
                    sf.write(temp_norm_wav, norm_audio, s1_sr, subtype='PCM_24' if s1_sr <= 48000 else 'FLOAT')
                    export_as_mp3(temp_norm_wav, final_out_path, bitrate=mp3_bitrate)
                    if os.path.isfile(temp_norm_wav):
                        try:
                            os.remove(temp_norm_wav)
                        except Exception:
                            pass
                else:
                    sf.write(final_out_path, norm_audio, s1_sr, subtype='PCM_24' if s1_sr <= 48000 else 'FLOAT')
            finally:
                if os.path.isfile(stage1_wav):
                    try:
                        os.remove(stage1_wav)
                    except Exception:
                        pass

        track_dur = processed / sr
        t_track_total = time.time() - t_track_start
        sz_mb = os.path.getsize(final_out_path) / (1024**2) if os.path.isfile(final_out_path) else 0.0
        _safe_log(f"[+] Slot {slot_num} completed in {t_track_total:.1f}s ({track_dur / max(0.001, t_track_total):.1f}x real-time) | Size: {sz_mb:.1f} MB")
        out_results[str(slot_num)] = final_out_path

    # Step 2: All remaining independent tracks processed concurrently.
    # ThreadPoolExecutor shares the CUDA context (unlike ProcessPoolExecutor) so all
    # threads can use the RTX 3060 GPU simultaneously without re-initialising CUDA.
    # Max 4 workers: enough to saturate CPU IIR chains and GPU STFT work on 6 tracks
    # without risking VRAM overflow on the 12 GB RTX 3060.
    _results_lock = threading.Lock()

    def _process_parallel_slot(p_slot_num: int, p_track_idx_0: int) -> None:
        """Worker: process one independent (non-bleed-gate) slot end-to-end."""
        if _cancel_check_thread():
            return

        slot_info    = active_slots[p_slot_num]
        in_path      = slot_info["path"]
        profile_id   = slot_info["profile"]
        base_name    = os.path.splitext(os.path.basename(in_path))[0]

        out_filename = (
            f"{base_name}_Mastered_Preview.{ext}"
            if (preview_sec and preview_sec > 0)
            else f"{base_name}_Mastered.{ext}"
        )
        final_out_path = os.path.join(output_dir, out_filename)

        _safe_log(f"\n[{p_track_idx_0 + 1}/{num_tracks}] Processing Slot {p_slot_num}: {os.path.basename(in_path)}")
        _safe_log(f"    Profile: {profile_id_to_str(profile_id)}")
        _safe_log(f"    Target:  {out_filename}")

        src_info    = sf.SoundFile(in_path, mode='r')
        sr          = src_info.samplerate
        tot_samples = src_info.frames
        src_info.close()

        max_samples   = min(tot_samples, int(preview_sec * sr)) if (preview_sec and preview_sec > 0) else tot_samples
        chunk_samples = int(chunk_sec * sr)
        processed_s   = 0
        t_track_start = time.time()

        # Silence gate — per-thread instance (stateful, cannot be shared).
        # Profiles with short burst speaking patterns get a longer hold to prevent
        # the gate cycling on normal conversational pauses between bursts.
        _LONG_HOLD_PROFILES = {"t6_laptop_fan", "t1_room_echo"}
        gate_hold_ms = 500.0 if profile_id in _LONG_HOLD_PROFILES else hold_ms
        silence_gate = DialogueSafeSilenceGate(
            sr=sr,
            open_thresh_db=open_thresh_db,
            close_thresh_db=open_thresh_db - 10.0,
            hold_ms=gate_hold_ms,
            rel_ms=120.0,
            lookahead_ms=30.0,
            floor_db=silence_floor_db
        ) if apply_silence_gate else None

        ai_suppressor = AIRNNoiseSuppressor(sr=sr, strength=ai_denoise_strength) if apply_ai_denoise else None

        # Output sink — stage1 WAV if normalising, else direct MP3/WAV
        stage1_wav_p = None
        proc_p = None; dst_p = None; temp_wav_p = None

        if apply_speech_normalization:
            stage1_wav_p = final_out_path + ".stage1.wav"
            dst_p = sf.SoundFile(stage1_wav_p, mode='w', samplerate=sr, channels=1,
                                 subtype='PCM_24' if sr <= 48000 else 'FLOAT')
        elif is_mp3:
            cmd_p = ['ffmpeg', '-y', '-f', 'f32le', '-ar', str(sr), '-ac', '1',
                     '-i', 'pipe:0', '-codec:a', 'libmp3lame', '-b:a', mp3_bitrate, final_out_path]
            try:
                proc_p = subprocess.Popen(cmd_p, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                temp_wav_p = final_out_path + ".temp.wav"
                dst_p = sf.SoundFile(temp_wav_p, mode='w', samplerate=sr, channels=1,
                                     subtype='PCM_24' if sr <= 48000 else 'FLOAT')
        else:
            dst_p = sf.SoundFile(final_out_path, mode='w', samplerate=sr, channels=1,
                                 subtype='PCM_24' if sr <= 48000 else 'FLOAT')

        def write_p(chunk_audio):
            if proc_p: proc_p.stdin.write(chunk_audio.astype(np.float32).tobytes())
            elif dst_p: dst_p.write(chunk_audio)

        def close_p():
            if proc_p:
                try: proc_p.stdin.close(); proc_p.wait()
                except Exception: pass
            if dst_p:
                try: dst_p.close()
                except Exception: pass

        try:
            if profile_id == "t6_laptop_fan":
                _safe_log("    Profiling laptop fan motor & chassis noise...")
                noise_mag = calibrate_laptop_fan_noise(in_path, sr=sr)
                enhancer  = LaptopMicEnhancer(sr=sr, noise_mag=noise_mag, alpha=1.2,
                                              floor_db=-24.0, desk_cut_db=-5.0,
                                              hollow_cut_db=-3.5, warmth_db=2.5, presence_db=4.5)
                src_p = sf.SoundFile(in_path, mode='r')
                try:
                    while processed_s < max_samples:
                        if _cancel_check_thread(): break
                        to_read = min(chunk_samples, max_samples - processed_s)
                        chunk = src_p.read(to_read, dtype='float32')
                        if len(chunk) == 0: break
                        if chunk.ndim > 1: chunk = chunk[:, 0]
                        is_last = (processed_s + len(chunk) >= max_samples)
                        enhanced = enhancer.process_chunk(chunk, is_last=is_last)
                        gated = silence_gate.process_chunk(enhanced, is_last=is_last) if silence_gate else enhanced
                        final_chunk = ai_suppressor.process_chunk(gated) if ai_suppressor else gated
                        write_p(final_chunk)
                        processed_s += len(chunk)
                        if progress_callback:
                            pct = (processed_s / max_samples) * 100.0
                            total_pct = ((p_track_idx_0 + processed_s / max_samples) / num_tracks) * 100.0
                            elapsed = time.time() - t_track_start
                            speed = (processed_s / sr) / max(0.001, elapsed)
                            eta = (max_samples - processed_s) / (sr * max(0.1, speed))
                            progress_callback(p_track_idx_0 + 1, num_tracks, out_filename, pct, total_pct, speed, format_time(eta), f"Processing Slot {p_slot_num}: {out_filename}")
                finally:
                    src_p.close()
            else:
                restorer = EnsembleVocalRestorer(profile=profile_id, sr=sr,
                                                use_ai_denoise=False)
                src_p = sf.SoundFile(in_path, mode='r')
                try:
                    while processed_s < max_samples:
                        if _cancel_check_thread(): break
                        to_read = min(chunk_samples, max_samples - processed_s)
                        chunk = src_p.read(to_read, dtype='float32')
                        if len(chunk) == 0: break
                        if chunk.ndim > 1: chunk = np.mean(chunk, axis=1)
                        is_last = (processed_s + len(chunk) >= max_samples)
                        tuned = restorer.process_chunk(chunk)
                        gated = silence_gate.process_chunk(tuned, is_last=is_last) if silence_gate else tuned
                        final_chunk = ai_suppressor.process_chunk(gated) if ai_suppressor else gated
                        write_p(final_chunk)

                        processed_s += len(chunk)
                        if progress_callback:
                            pct = (processed_s / max_samples) * 100.0
                            total_pct = ((p_track_idx_0 + processed_s / max_samples) / num_tracks) * 100.0
                            elapsed = time.time() - t_track_start
                            speed = (processed_s / sr) / max(0.001, elapsed)
                            eta = (max_samples - processed_s) / (sr * max(0.1, speed))
                            progress_callback(p_track_idx_0 + 1, num_tracks, out_filename, pct, total_pct, speed, format_time(eta), f"Processing Slot {p_slot_num}: {out_filename}")
                finally:
                    src_p.close()
        finally:
            close_p()

        # Loudness normalisation pass
        if apply_speech_normalization and stage1_wav_p and os.path.isfile(stage1_wav_p):
            if not _cancel_check_thread():
                try:
                    _safe_log(f"    ⚖️ Normalizing Speech Loudness (Target: {target_lufs:.1f} LUFS)...")
                    stage1_data, s1_sr = sf.read(stage1_wav_p, dtype='float32')
                    normalizer = BroadcastSpeechNormalizer(sr=s1_sr, target_lufs=target_lufs,
                                                          peak_ceiling_db=peak_ceiling_db)
                    norm_audio, gain_db, meas_lufs, final_peak = normalizer.normalize_audio(stage1_data)
                    _safe_log(f"    [+] Speech Level: {meas_lufs:.1f} LUFS -> {gain_db:+.1f} dB | Peak: {final_peak:.1f} dBFS")
                    if is_mp3:
                        tnw = final_out_path + ".norm.wav"
                        sf.write(tnw, norm_audio, s1_sr, subtype='PCM_24' if s1_sr <= 48000 else 'FLOAT')
                        export_as_mp3(tnw, final_out_path, bitrate=mp3_bitrate)
                        try: os.remove(tnw)
                        except Exception: pass
                    else:
                        sf.write(final_out_path, norm_audio, s1_sr, subtype='PCM_24' if s1_sr <= 48000 else 'FLOAT')
                finally:
                    try: os.remove(stage1_wav_p)
                    except Exception: pass

        track_dur     = processed_s / sr
        t_track_total = time.time() - t_track_start
        sz_mb = os.path.getsize(final_out_path) / (1024**2) if os.path.isfile(final_out_path) else 0.0
        _safe_log(f"[+] Slot {p_slot_num} completed in {t_track_total:.1f}s ({track_dur / max(0.001, t_track_total):.1f}x real-time) | Size: {sz_mb:.1f} MB")
        with _results_lock:
            out_results[str(p_slot_num)] = final_out_path

    # Dispatch independent tracks concurrently (max 4 workers to avoid VRAM saturation)
    n_workers = min(len(parallel_slots), 4)
    if n_workers > 0:
        _safe_log(f"\n⚡ Launching {len(parallel_slots)} independent tracks across {n_workers} parallel workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_parallel_slot, slot_num, len(bleed_gate_slots) + idx): slot_num
                for idx, slot_num in enumerate(parallel_slots)
            }
            for future in concurrent.futures.as_completed(futures):
                slot_num = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    _safe_log(f"[-] Slot {slot_num} raised an error: {exc}")

    total_session_time = time.time() - t_session_start
    log_func("\n" + "=" * 70)
    log_func(f"✅ ALL {len(out_results)} SESSION TRACKS MASTERED SUCCESSFULLY IN {format_time(total_session_time)}!")
    log_func(f"Master Directory: {output_dir}")
    log_func("=" * 70)
    return out_results


# Aliases for complete backwards compatibility
WaveformMatchBleedGate = StreamingMultitrackBleedGate
AdaptiveBleedGate = StreamingMultitrackBleedGate


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_multitrack_session(args):
    start_wall_time = time.time()
    
    # 1. Open Audio Input
    if args.multichannel:
        input_files = [args.multichannel]
    elif args.inputs:
        input_files = args.inputs
    else:
        print("[-] Error: You must provide either --inputs with track files or --multichannel with a multichannel file.")
        sys.exit(1)
        
    print(f"[*] Opening audio source ({len(input_files)} file(s))...")
    src = AudioSource(input_files)
    
    target_0idx = args.target - 1
    if target_0idx < 0 or target_0idx >= src.n_tracks:
        print(f"[-] Error: Target track {args.target} is out of range. Input has {src.n_tracks} tracks (1 to {src.n_tracks}).")
        src.close()
        sys.exit(1)
        
    total_sec = src.duration_sec
    if args.preview and args.preview > 0:
        total_sec = min(total_sec, float(args.preview))
        print(f"[*] Preview mode active: processing first {args.preview:.0f} seconds ({format_time(total_sec)}).")
        
    print(f"[*] Session Specs:")
    print(f"    - Total Tracks: {src.n_tracks}")
    print(f"    - Sample Rate:  {src.samplerate} Hz")
    print(f"    - Duration:     {format_time(src.duration_sec)} ({src.duration_sec:.2f}s, {src.total_samples:,} samples)")
    print(f"    - Target Track: Track {args.target}")

    # 2. Calibrate Thresholds
    calibrator = BleedCalibrator(target_idx=target_0idx, n_tracks=src.n_tracks, sr=src.samplerate)
    ref_thresh, tgt_thresh, noise_floor, speech_level = calibrator.calibrate(
        src,
        max_scan_sec=args.calibrate_sec,
        sensitivity=args.sensitivity
    )

    print(f"[*] Dynamic Calibration:")
    print(f"    - Target Noise Floor:       {20*math.log10(max(1e-9, noise_floor)):.1f} dBFS")
    print(f"    - Target Speech Level:      {20*math.log10(max(1e-9, speech_level)):.1f} dBFS")
    print(f"    - Target Speech Threshold:  {20*math.log10(max(1e-9, tgt_thresh)):.1f} dBFS")
    print(f"    - Reference Active Thresh:  {20*math.log10(max(1e-9, ref_thresh)):.1f} dBFS")

    # 3. Setup Output Destination
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(input_files[0]))[0]
        output_path = f"{base_name}_track{args.target}_cleaned.wav"
        
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_handle = sf.SoundFile(
        output_path,
        mode='w',
        samplerate=src.samplerate,
        channels=1,
        subtype='PCM_24' if src.samplerate <= 48000 else 'FLOAT'
    )
    print(f"[*] Output destination: {output_path}")

    # 4. Initialize Gating Engine
    gate = StreamingMultitrackBleedGate(
        sr=src.samplerate,
        target_idx=target_0idx,
        n_tracks=src.n_tracks,
        match_cutoff=args.match_cutoff,
        sensitivity=args.sensitivity,
        floor_db=args.floor,
        attack_ms=args.attack,
        hold_ms=args.hold,
        release_ms=args.release,
        ref_speech_thresh=ref_thresh,
        tgt_speech_thresh=tgt_thresh,
        enhance=args.enhance,
    )

    # 5. Streaming Chunk Processing
    chunk_samples = int(args.chunk_sec * src.samplerate)
    total_processed_samples = 0
    max_process_samples = int(total_sec * src.samplerate)
    total_unmuted_samples = 0
    t_start_dsp = time.time()
    
    print("\n" + "="*70)
    print("PROGRESS: Streaming & Processing multitrack audio...")
    print("="*70)

    try:
        while total_processed_samples < max_process_samples:
            to_read = min(chunk_samples, max_process_samples - total_processed_samples)
            raw_chunk = src.read_chunk(to_read)
            if raw_chunk is None or raw_chunk.shape[1] == 0:
                break
                
            processed_chunk, stats = gate.process_chunk(raw_chunk)
            out_handle.write(processed_chunk)
            
            n_chunk = len(processed_chunk)
            total_processed_samples += n_chunk
            total_unmuted_samples += int(stats["unmuted_ratio"] * n_chunk)
            
            # Progress display
            curr_sec = total_processed_samples / src.samplerate
            pct = (total_processed_samples / max_process_samples) * 100.0
            elapsed = time.time() - t_start_dsp
            speed = curr_sec / max(0.001, elapsed)
            eta_sec = (max_process_samples - total_processed_samples) / (src.samplerate * max(0.1, speed))
            
            bar_width = 30
            filled = int(bar_width * (pct / 100.0))
            bar = "=" * filled + "-" * (bar_width - filled)
            
            sys.stdout.write(
                f"\r[{bar}] {pct:5.1f}% | {format_time(curr_sec)} / {format_time(total_sec)} | "
                f"Speed: {speed:5.1f}x | ETA: {format_time(eta_sec)} "
            )
            sys.stdout.flush()
            
    finally:
        out_handle.close()
        src.close()

    total_time = time.time() - start_wall_time
    proc_duration = total_processed_samples / src.samplerate
    unmuted_duration = total_unmuted_samples / src.samplerate
    muted_duration = proc_duration - unmuted_duration
    muted_pct = (muted_duration / max(0.1, proc_duration)) * 100.0

    print("\n" + "="*70)
    print("PROCESSING COMPLETE - SUMMARY REPORT")
    print("="*70)
    print(f"Total Audio Processed:   {format_time(proc_duration)} ({proc_duration:.2f} seconds)")
    print(f"Target Active Time:      {format_time(unmuted_duration)} ({100.0 - muted_pct:.1f}%)")
    print(f"Target Muted Time:       {format_time(muted_duration)} ({muted_pct:.1f}%)")
    print(f"Total Wall Clock Time:   {total_time:.2f} seconds ({proc_duration / max(0.001, total_time):.1f}x real-time)")
    print(f"Cleaned Output Audio:    {output_path}")
    print("="*70 + "\n")


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Multitrack Audio Bleed Suppressor and Adaptive Spectral Gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    input_group = parser.add_argument_group("Audio Inputs")
    input_group.add_argument(
        "--inputs", "-i", nargs="+",
        help="Paths to individual audio track files (e.g. mic1.mp3 mic2.mp3 ... mic6.mp3)"
    )
    input_group.add_argument(
        "--multichannel", "-m",
        help="Path to a single multichannel audio file (e.g. 6-channel WAV)"
    )
    input_group.add_argument(
        "--target", "-t", type=int, default=5,
        help="Index of the target track with severe bleed (1-based index, e.g. 5 for Track 5)"
    )
    input_group.add_argument(
        "--output", "-o",
        help="Path to write the cleaned output track (defaults to [name]_track[target]_cleaned.wav)"
    )

    gate_group = parser.add_argument_group("Gate & Bleed Parameters")
    gate_group.add_argument(
        "--match-cutoff", type=float, default=0.48,
        help="Spectral match cutoff (0.35 to 0.70). Audio matching other tracks >= this is muted as bleed"
    )
    gate_group.add_argument(
        "--sensitivity", type=float, default=1.0,
        help="Vocal sensitivity multiplier (higher = opens more easily for quiet whispers; lower = more strict)"
    )
    gate_group.add_argument(
        "--attack", type=float, default=10.0,
        help="Gate attack time in milliseconds (fast opening for word onsets)"
    )
    gate_group.add_argument(
        "--hold", type=float, default=160.0,
        help="Gate hold time in milliseconds (bridges syllable and word pauses)"
    )
    gate_group.add_argument(
        "--release", type=float, default=85.0,
        help="Gate release time in milliseconds (smooth fade-out to prevent clicks)"
    )
    gate_group.add_argument(
        "--floor", type=float, default=-60.0,
        help="Mute attenuation floor in dB (-60 for near silence, -99 or lower for -inf digital mute, -36 for subtle room tone)"
    )
    gate_group.add_argument(
        "--enhance", action="store_true",
        help="Apply broadcast vocal tuning (rumble cut, anti-box EQ, warmth saturation, dynamic speech leveler)"
    )

    perf_group = parser.add_argument_group("Performance & Workflow")
    perf_group.add_argument(
        "--chunk-sec", type=float, default=30.0,
        help="Streaming chunk size in seconds (ensures minimal memory usage for long recordings)"
    )
    perf_group.add_argument(
        "--calibrate-sec", type=float, default=120.0,
        help="Maximum duration in seconds to scan for initial level calibration"
    )
    perf_group.add_argument(
        "--preview", "-p", type=float, default=None,
        help="Quick preview mode: only process the first N seconds (e.g. --preview 120 for 2 minutes)"
    )

    laptop_group = parser.add_argument_group("Laptop Mic Restoration Mode")
    laptop_group.add_argument(
        "--denoise-laptop", action="store_true",
        help="Run laptop mic restoration mode (profiles & removes fan whine + proximity warmth & anti-box EQ)"
    )
    laptop_group.add_argument(
        "--laptop-file", type=str, default=None,
        help="Path to audio file to restore (can also pass via --inputs)"
    )

    auto_group = parser.add_argument_group("Automated Session Mastering Mode")
    auto_group.add_argument(
        "--auto-session", type=str, default=None,
        help="Path to folder or directory containing session audio files (Track 1-6 auto-assigned)"
    )
    auto_group.add_argument(
        "--format", type=str, default="mp3", choices=["mp3", "wav"],
        help="Export format for session stems (mp3 = 320 kbps broadcast CBR, wav = 24-bit PCM)"
    )
    auto_group.add_argument(
        "--silence-gate", action="store_true", default=True,
        help="Apply dialogue-safe digital silence gate to all non-speaking pauses"
    )
    auto_group.add_argument(
        "--ai-denoise", action="store_true", default=False,
        help="Apply Deep Neural AI Noise Suppression (RNNoise) across all speech"
    )
    auto_group.add_argument(
        "--ai-strength", type=float, default=1.0,
        help="Neural AI noise suppression strength (0.0 to 1.0, default: 1.0)"
    )

    return parser


if __name__ == "__main__":
    parser = build_cli_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    if args.auto_session:
        sess_dir = os.path.abspath(args.auto_session)
        if not os.path.isdir(sess_dir):
            print(f"[-] Error: Session directory not found: {sess_dir}")
            sys.exit(1)
        detected_slots = auto_detect_session_tracks(sess_dir)
        if not detected_slots:
            print(f"[-] Error: No audio tracks detected in: {sess_dir}")
            sys.exit(1)
        out_master_dir = args.output or os.path.join(sess_dir, "Mastered")
        res = process_automated_session(
            slots=detected_slots,
            output_dir=out_master_dir,
            export_format=args.format,
            apply_silence_gate=args.silence_gate,
            apply_ai_denoise=args.ai_denoise,
            ai_denoise_strength=args.ai_strength,
            preview_sec=args.preview,
            chunk_sec=args.chunk_sec
        )
        sys.exit(0 if res else 1)

    if args.denoise_laptop or args.laptop_file:
        in_file = args.laptop_file or (args.inputs[0] if args.inputs else None)
        if not in_file:
            print("[-] Error: Specify an input file using --laptop-file <file> or --inputs <file>.")
            sys.exit(1)
        out_file = args.output
        if not out_file:
            base, ext = os.path.splitext(in_file)
            out_file = f"{base}_cleaned_enhanced.wav"
        success = process_laptop_mic_file(
            input_path=in_file,
            output_path=out_file,
            apply_ai_denoise=args.ai_denoise,
            ai_denoise_strength=args.ai_strength,
            preview_sec=args.preview,
            chunk_sec=args.chunk_sec
        )
        sys.exit(0 if success else 1)

    process_multitrack_session(args)
