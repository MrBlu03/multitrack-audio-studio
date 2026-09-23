"""
test_all_chains_v43.py
======================
Full regression suite for Multitrack Audio Studio v4.3
Tests every DSP profile, the GPU STFT paths, parallel processing,
DialogueSafeSilenceGate, BroadcastSpeechNormalizer, and the Whisper model cache.

Run from the workspace root:
    python test_all_chains_v43.py
"""
import sys, os, time, tempfile, traceback
import numpy as np
import soundfile as sf

# Allow importing from workspace root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multitrack_bleed_gate as mb

SR = 48000
CHUNK_SEC = 2.0           # short chunks to keep tests fast
DURATION_SEC = 4.0        # 4 seconds of synthetic audio per test


def _make_speech(sr=SR, duration=DURATION_SEC, amp=0.25) -> np.ndarray:
    """Voiced speech simulator: 120 Hz fundamental + harmonics + noise burst."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    sig  = 0.5 * np.sin(2 * np.pi * 120 * t)
    sig += 0.3 * np.sin(2 * np.pi * 240 * t)
    sig += 0.2 * np.sin(2 * np.pi * 480 * t)
    sig += 0.1 * np.sin(2 * np.pi * 960 * t)
    sig += 0.05 * np.random.randn(len(t)).astype(np.float32)
    sig *= amp
    # Envelope: 0.1s fade-in / fade-out to avoid clicks
    fade = int(0.1 * sr)
    sig[:fade]  *= np.linspace(0, 1, fade, dtype=np.float32)
    sig[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return sig


def _make_silence(sr=SR, duration=DURATION_SEC) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _make_fan_noise(sr=SR, duration=DURATION_SEC, fundamental=90.5) -> np.ndarray:
    """Laptop fan: narrow-band tones at fundamental + harmonics."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 4]:
        sig += 0.02 * np.sin(2 * np.pi * fundamental * h * t)
    sig += 0.005 * np.random.randn(len(t)).astype(np.float32)
    return sig


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0

def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        print(f"  ✅ PASS  {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {name}" + (f"  →  {detail}" if detail else ""))
        FAIL += 1


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. GPU detection
# ---------------------------------------------------------------------------
section("1. GPU / CUDA detection")
check("_CUDA_AVAILABLE flag set",   isinstance(mb._CUDA_AVAILABLE, bool))
if mb._CUDA_AVAILABLE:
    check("_torch module present when CUDA available", mb._torch is not None)
    check("CUDA device object valid", mb._CUDA_DEVICE is not None)
    print(f"     → CUDA detected: {mb._torch.cuda.get_device_name(0)}")
else:
    print("     → No CUDA / torch not installed — GPU paths will use scipy fallback (correct for non-CUDA machine)")
    check("CPU fallback active (_torch None or no CUDA)",
          mb._torch is None or not mb._CUDA_AVAILABLE)


# ---------------------------------------------------------------------------
# 2. _gpu_spectral_subtract (GPU or CPU fallback)
# ---------------------------------------------------------------------------
section("2. _gpu_spectral_subtract (STFT spectral subtraction)")
speech = _make_speech()
fan    = _make_fan_noise()
mixed  = np.clip(speech + fan, -1.0, 1.0)

nperseg  = 1024
noverlap = 768
noise_mag_shape = (nperseg // 2 + 1, 1)
noise_profile = np.abs(np.random.randn(*noise_mag_shape).astype(np.float32)) * 0.01

try:
    t0  = time.perf_counter()
    out = mb._gpu_spectral_subtract(mixed, noise_profile, alpha=1.6,
                                    nperseg=nperseg, noverlap=noverlap, sr=SR)
    dt  = time.perf_counter() - t0
    check("Output same length as input", len(out) == len(mixed),
          f"got {len(out)}, expected {len(mixed)}")
    check("Output dtype is float32",     out.dtype == np.float32)
    check("Output contains no NaN/Inf",  np.isfinite(out).all())
    check("Output not all zeros",        np.abs(out).max() > 1e-6)
    print(f"     → Processed {DURATION_SEC}s in {dt*1000:.1f}ms "
          f"({'GPU' if mb._CUDA_AVAILABLE else 'CPU fallback'})")
except Exception as e:
    check("_gpu_spectral_subtract raised no exception", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 3. _gpu_bleed_cosine_batch
# ---------------------------------------------------------------------------
section("3. _gpu_bleed_cosine_batch (cosine similarity batch)")
from scipy import signal as _signal
sr_s = SR
nperseg_b, noverlap_b = 1024, 768
audio_a = _make_speech(amp=0.3)
audio_b = _make_speech(amp=0.2)   # simulated bleed

_, _, Za = _signal.stft(audio_a, fs=sr_s, nperseg=nperseg_b, noverlap=noverlap_b)
_, _, Zb = _signal.stft(audio_b, fs=sr_s, nperseg=nperseg_b, noverlap=noverlap_b)

n_stft = Za.shape[1]
m5     = np.abs(Za).astype(np.float32)
norm5  = (np.sqrt(np.sum(m5**2, axis=0)) + 1e-12).astype(np.float32)
db5    = (20.0 * np.log10(norm5 + 1e-9)).astype(np.float32)
mk     = np.abs(Zb).astype(np.float32)
norm_k = (np.sqrt(np.sum(mk**2, axis=0)) + 1e-12).astype(np.float32)
db_k   = (20.0 * np.log10(norm_k + 1e-9)).astype(np.float32)
rms_k  = np.sqrt(np.mean(audio_b**2)) * np.ones(n_stft, dtype=np.float32)
shifts = np.array([0, 2, 5, 10, 20], dtype=np.int32)

try:
    t0 = time.perf_counter()
    sim_out, _ = mb._gpu_bleed_cosine_batch(m5, norm5, db5, mk, norm_k, db_k,
                                             rms_k, shifts, n_stft)
    dt = time.perf_counter() - t0
    check("Output length == n_stft",    len(sim_out) == n_stft,
          f"got {len(sim_out)}")
    check("Output dtype float32",       sim_out.dtype == np.float32)
    check("Similarities in [0, 1+ε]",   sim_out.max() <= 1.1)
    check("No NaN/Inf",                 np.isfinite(sim_out).all())
    print(f"     → {len(shifts)} shifts × {n_stft} frames in {dt*1000:.1f}ms "
          f"({'GPU' if mb._CUDA_AVAILABLE else 'CPU fallback'})")
except Exception as e:
    check("_gpu_bleed_cosine_batch raised no exception", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. EnsembleVocalRestorer — all profiles
# ---------------------------------------------------------------------------
section("4. EnsembleVocalRestorer — all DSP profiles")
PROFILES = ["t1_room_echo", "t2_muffled", "t3_reference",
            "t4_megaphone", "t7_rati_clarity"]

speech_chunk = _make_speech(duration=CHUNK_SEC)

for prof in PROFILES:
    try:
        restorer = mb.EnsembleVocalRestorer(profile=prof, sr=SR,
                                            use_ai_denoise=False)
        out = restorer.process_chunk(speech_chunk.copy())
        ok_len   = len(out) == len(speech_chunk)
        ok_dtype = out.dtype == np.float32
        ok_finite = np.isfinite(out).all()
        ok_nonzero = np.abs(out).max() > 1e-6
        ok = ok_len and ok_dtype and ok_finite and ok_nonzero
        detail = (
            f"len={len(out)}(want {len(speech_chunk)}), "
            f"dtype={out.dtype}, finite={ok_finite}, peak={np.abs(out).max():.4f}"
        )
        check(f"EnsembleVocalRestorer [{prof}]", ok, detail if not ok else "")
    except Exception as e:
        check(f"EnsembleVocalRestorer [{prof}] raised no exception", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 5. LaptopMicEnhancer (t6_laptop_fan) — with and without noise profile
# ---------------------------------------------------------------------------
section("5. LaptopMicEnhancer (t6_laptop_fan)")

fan_speech = np.clip(_make_speech(duration=CHUNK_SEC) + _make_fan_noise(duration=CHUNK_SEC), -1, 1)

# Without noise profile
try:
    enh = mb.LaptopMicEnhancer(sr=SR, noise_mag=None, alpha=1.6,
                                floor_db=-40.0, desk_cut_db=-5.0,
                                hollow_cut_db=-3.5, warmth_db=2.5, presence_db=4.5)
    out = enh.process_chunk(fan_speech.copy())
    ok = len(out) == len(fan_speech) and out.dtype == np.float32 and np.isfinite(out).all()
    check("LaptopMicEnhancer (no noise profile)", ok)
except Exception as e:
    check("LaptopMicEnhancer (no noise profile)", False, str(e))
    traceback.print_exc()

# With synthetic noise profile (same shape as real calibrate output)
try:
    noise_mag = (np.abs(np.random.randn(513, 1)) * 0.01).astype(np.float32)
    enh2 = mb.LaptopMicEnhancer(sr=SR, noise_mag=noise_mag, alpha=1.6,
                                 floor_db=-40.0, desk_cut_db=-5.0,
                                 hollow_cut_db=-3.5, warmth_db=2.5, presence_db=4.5)
    out2 = enh2.process_chunk(fan_speech.copy())
    ok2 = len(out2) == len(fan_speech) and out2.dtype == np.float32 and np.isfinite(out2).all()
    check("LaptopMicEnhancer (with noise profile, GPU STFT path)", ok2)
    if ok2:
        # Also test is_last=True (final chunk path)
        out3 = enh2.process_chunk(fan_speech.copy(), is_last=True)
        check("LaptopMicEnhancer is_last=True", np.isfinite(out3).all())
except Exception as e:
    check("LaptopMicEnhancer (with noise profile)", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 6. DialogueSafeSilenceGate
# ---------------------------------------------------------------------------
section("6. DialogueSafeSilenceGate")

gate = mb.DialogueSafeSilenceGate(sr=SR, open_thresh_db=-34.0,
                                   close_thresh_db=-44.0, hold_ms=280.0,
                                   rel_ms=120.0, lookahead_ms=30.0,
                                   floor_db=-120.0)

# (a) Speech chunk should come through (not all-zero)
speech_g = _make_speech(duration=CHUNK_SEC, amp=0.3)
out_speech = gate.process_chunk(speech_g.copy())
check("Gate passes speech (not all-zero output)",
      np.abs(out_speech).max() > 1e-4,
      f"max={np.abs(out_speech).max():.6f}")

# (b) Deep silence should be zeroed out (after hold time)
gate2 = mb.DialogueSafeSilenceGate(sr=SR, open_thresh_db=-34.0,
                                    close_thresh_db=-44.0, hold_ms=50.0,
                                    rel_ms=50.0, lookahead_ms=10.0,
                                    floor_db=-120.0)
# Feed a full silent chunk (gate never opens)
silence_g = _make_silence(duration=CHUNK_SEC)
out_silence = gate2.process_chunk(silence_g.copy())
max_sil = np.abs(out_silence).max()
check("Gate zeros out pure silence",
      max_sil < 1e-9,
      f"max={max_sil:.2e}")

# (c) Output length preserved
check("Gate output length == input length",
      len(out_speech) == len(speech_g))

# (d) Multi-chunk stateful streaming
gate3 = mb.DialogueSafeSilenceGate(sr=SR, open_thresh_db=-34.0,
                                    close_thresh_db=-44.0, hold_ms=280.0)
results = []
for i in range(4):
    is_last = (i == 3)
    c = _make_speech(duration=CHUNK_SEC, amp=0.3)
    results.append(gate3.process_chunk(c, is_last=is_last))
check("Gate multi-chunk: no NaN/Inf across 4 chunks",
      all(np.isfinite(r).all() for r in results))


# ---------------------------------------------------------------------------
# 7. BroadcastSpeechNormalizer
# ---------------------------------------------------------------------------
section("7. BroadcastSpeechNormalizer (LUFS + peak limiter)")

norm = mb.BroadcastSpeechNormalizer(sr=SR, target_lufs=-18.0, peak_ceiling_db=-1.0)

# (a) LUFS measurement on speech signal (longer for stable measurement)
long_speech = _make_speech(duration=10.0, amp=0.15)
lufs_measured = norm.measure_active_speech_lufs(long_speech)
check("LUFS measurement returns finite value",
      np.isfinite(lufs_measured) and lufs_measured > -70.0,
      f"measured={lufs_measured:.1f} LUFS")

# (b) Normalise and check output LUFS close to target
norm_audio, gain_db, meas_lufs, final_peak = norm.normalize_audio(long_speech)
check("normalize_audio output length == input",
      len(norm_audio) == len(long_speech))
check("normalize_audio dtype == float32",
      norm_audio.dtype == np.float32)
check("normalize_audio no NaN/Inf",
      np.isfinite(norm_audio).all())
check("Peak does not exceed ceiling (-1.0 dBFS)",
      np.abs(norm_audio).max() <= 10 ** (-1.0 / 20.0) + 1e-5,
      f"peak={20*np.log10(np.abs(norm_audio).max()):.2f} dBFS")

# Re-measure output LUFS (should be near -18)
lufs_after = norm.measure_active_speech_lufs(norm_audio)
check("Post-normalise LUFS within ±3 dB of target (-18 LUFS)",
      abs(lufs_after - (-18.0)) < 3.0,
      f"measured {lufs_after:.1f} LUFS after normalise")

print(f"     → Input: {meas_lufs:.1f} LUFS  → Applied: {gain_db:+.1f} dB → Output: {lufs_after:.1f} LUFS | Peak: {final_peak:.1f} dBFS")


# ---------------------------------------------------------------------------
# 8. calibrate_laptop_fan_noise
# ---------------------------------------------------------------------------
section("8. calibrate_laptop_fan_noise (robust multi-window median)")

with tempfile.TemporaryDirectory() as tmp:
    fan_file = os.path.join(tmp, "fan_test.wav")
    # 30s of fan noise with a short speech burst in the middle
    full = np.concatenate([
        _make_fan_noise(duration=10.0),
        _make_speech(duration=5.0, amp=0.4),
        _make_fan_noise(duration=10.0),
        _make_speech(duration=3.0, amp=0.3),
        _make_fan_noise(duration=2.0),
    ])
    sf.write(fan_file, full.astype(np.float32), SR)

    try:
        t0 = time.perf_counter()
        noise_mag = mb.calibrate_laptop_fan_noise(fan_file, sr=SR)
        dt = time.perf_counter() - t0
        check("calibrate returns ndarray",          isinstance(noise_mag, np.ndarray))
        check("Shape is (513, 1)",                  noise_mag.shape == (513, 1),
              f"got {noise_mag.shape}")
        check("dtype float32",                      noise_mag.dtype == np.float32)
        check("No NaN/Inf in noise profile",        np.isfinite(noise_mag).all())
        check("Profile has non-trivial energy",     noise_mag.max() > 0.0)
        print(f"     → Calibrated in {dt*1000:.0f}ms | Peak bin: {noise_mag.max():.6f}")
    except Exception as e:
        check("calibrate_laptop_fan_noise raised no exception", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 9. process_automated_session — parallel processing across all profiles
# ---------------------------------------------------------------------------
section("9. process_automated_session — all profiles, parallel execution")

with tempfile.TemporaryDirectory() as sess_dir:
    out_dir = os.path.join(sess_dir, "Mastered")

    # Build 5 synthetic WAV tracks (slot 6 inactive, slot 5 = t5_bleed_gate needs all files)
    track_files = {}
    for slot in range(1, 6):
        path = os.path.join(sess_dir, f"Final_Audio_A{slot:02d}.wav")
        sig = _make_speech(duration=6.0, amp=0.2 + 0.02 * slot)
        sf.write(path, sig, SR)
        track_files[slot] = path

    # Deliberately create a profile mix: t1, t2, t3, t4 in parallel + t5 bleed-gate first
    slots = {
        1: {"path": track_files[1], "profile": "t1_room_echo"},
        2: {"path": track_files[2], "profile": "t2_muffled"},
        3: {"path": track_files[3], "profile": "t3_reference"},
        4: {"path": track_files[4], "profile": "t4_megaphone"},
        5: {"path": track_files[5], "profile": "t5_bleed_gate"},
    }

    log_lines = []
    t0 = time.perf_counter()
    results = mb.process_automated_session(
        slots=slots,
        output_dir=out_dir,
        export_format="wav",
        apply_silence_gate=True,
        apply_ai_denoise=False,
        apply_speech_normalization=True,
        target_lufs=-18.0,
        peak_ceiling_db=-1.0,
        preview_sec=4.0,           # 4-second preview to keep test fast
        chunk_sec=2.0,
        log_func=log_lines.append,
    )
    dt = time.perf_counter() - t0

    check("process_automated_session returns dict",     isinstance(results, dict))
    check("All 5 slots produced output files",          len(results) == 5,
          f"got {len(results)} results")
    for slot_k, out_path in results.items():
        exists = os.path.isfile(out_path)
        check(f"  Slot {slot_k} output file exists",   exists, out_path)
        if exists:
            data, file_sr = sf.read(out_path, dtype='float32')
            check(f"  Slot {slot_k} output readable",  len(data) > 0)
            check(f"  Slot {slot_k} no NaN/Inf",       np.isfinite(data).all())
            check(f"  Slot {slot_k} peak < 0 dBFS",    np.abs(data).max() <= 1.0 + 1e-4,
                  f"peak={np.abs(data).max():.4f}")

    parallel_msg = any("parallel workers" in l for l in log_lines)
    check("Parallel workers log message present",       parallel_msg)
    print(f"     → 5 tracks processed in {dt:.2f}s (4s preview each)")


# ---------------------------------------------------------------------------
# 10. Campaign config sanity
# ---------------------------------------------------------------------------
section("10. Campaign configs & slot-profile mapping")

try:
    sw5e = mb.get_campaign_config("sw5e")
    red  = mb.get_campaign_config("red")
    sw5e_slots = sw5e["slots"]
    red_slots  = red["slots"]
    check("SW5E config has 6 slots",    len(sw5e_slots) == 6, f"got {len(sw5e_slots)}")
    check("RED config has 6 slots",     len(red_slots)  == 6, f"got {len(red_slots)}")
    check("SW5E slot 5 is t5_bleed_gate",
          sw5e_slots[5].get("profile_id") == "t5_bleed_gate",
          f"got {sw5e_slots[5].get('profile_id')}")
    check("RED slot 4 is t7_rati_clarity",
          red_slots[4].get("profile_id") == "t7_rati_clarity",
          f"got {red_slots[4].get('profile_id')}")
    check("RED slot 6 inactive",
          red_slots[6].get("profile_id") == "skip" or not red_slots[6].get("active", True),
          f"active={red_slots[6].get('active')}, profile={red_slots[6].get('profile_id')}")
    check("SW5E slot 1 is t1_room_echo",
          sw5e_slots[1].get("profile_id") == "t1_room_echo")
    check("RED slot 2 is t3_reference (Blu GM)",
          red_slots[2].get("profile_id") == "t3_reference")
except Exception as e:
    check("Campaign config access raised no exception", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 11. t7_rati_clarity profile via EnsembleVocalRestorer
# ---------------------------------------------------------------------------
section("11. t7_rati_clarity (Rati/Umbra) DSP profile")

try:
    rati = mb.EnsembleVocalRestorer(profile="t7_rati_clarity", sr=SR, use_ai_denoise=False)
    chunk = _make_speech(duration=CHUNK_SEC, amp=0.2)
    out = rati.process_chunk(chunk)
    check("Output length correct",        len(out) == len(chunk))
    check("No NaN/Inf",                   np.isfinite(out).all())
    check("Peak within headroom",         np.abs(out).max() <= 1.5)
    print(f"     → Peak: {np.abs(out).max():.4f}")
except Exception as e:
    check("t7_rati_clarity raised no exception", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 12. AIRNNoiseSuppressor (only if DLL is present — skip gracefully)
# ---------------------------------------------------------------------------
section("12. AIRNNoiseSuppressor (RNNoise DLL)")

try:
    dll_path = mb.get_rnnoise_dll_path()
    if dll_path and os.path.isfile(dll_path):
        suppressor = mb.AIRNNoiseSuppressor(sr=SR, strength=1.0)
        # _state is set when DLL loaded successfully; is_available() is a classmethod
        rnn_ready = suppressor._state is not None
        check("RNNoise DLL loaded (_state set)", rnn_ready)
        if rnn_ready:
            chunk = _make_speech(duration=CHUNK_SEC, amp=0.15) + _make_fan_noise(duration=CHUNK_SEC) * 0.3
            out = suppressor.process_chunk(chunk)
            check("RNNoise output length correct",  len(out) == len(chunk))
            check("RNNoise no NaN/Inf",             np.isfinite(out).all())
            # Test resample_poly path: 44100 Hz input
            suppressor_441 = mb.AIRNNoiseSuppressor(sr=44100, strength=0.8)
            if suppressor_441._state is not None:
                chunk_441 = _make_speech(sr=44100, duration=CHUNK_SEC).astype(np.float32)
                out_441 = suppressor_441.process_chunk(chunk_441)
                check("RNNoise resample_poly path (44.1→48kHz)",
                      len(out_441) == len(chunk_441) and np.isfinite(out_441).all())
    else:
        # DLL is only present inside the PyInstaller EXE bundle — this is expected
        print("     → RNNoise DLL not found at module level (only present in EXE build) — skipping")
        check("RNNoise DLL absent — skip is valid", True)
except Exception as e:
    check("AIRNNoiseSuppressor raised no exception", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 13. Direct 4K Raw Video Ingestion & 8-Track MeldStudio / OBS Stem Extraction
# ---------------------------------------------------------------------------
section("13. Direct 4K Raw Video Ingestion & Multi-Track Extraction")

try:
    import video_ingest
    import subprocess
    import soundfile as sf

    # 1. Test video mapping configs
    sw_map = video_ingest.get_default_video_track_mapping("sw5e", num_streams=8)
    check("SW5E 8-stream mapping has 8 entries", len(sw_map) == 8)
    check("SW5E T1 is mixdown (slot None)",       sw_map[0]["slot"] is None and sw_map[0]["role"] == "mixdown")
    check("SW5E T2 is music (slot None)",         sw_map[1]["slot"] is None and sw_map[1]["role"] == "music")
    check("SW5E T3 is Robin (slot 1)",            sw_map[2]["slot"] == 1 and sw_map[2]["player"] == "Robin")
    check("SW5E T7 is Mathew (slot 5)",           sw_map[6]["slot"] == 5 and sw_map[6]["player"] == "Mathew")
    check("SW5E T8 is Timmy (slot 6)",            sw_map[7]["slot"] == 6 and sw_map[7]["player"] == "Timmy")

    red_map = video_ingest.get_default_video_track_mapping("red", num_streams=8)
    check("RED 8-stream mapping has 8 entries",   len(red_map) == 8)
    check("RED T3 is Robin (slot 1)",             red_map[2]["slot"] == 1 and red_map[2]["player"] == "Robin")
    check("RED T4 is Blu GM (slot 2)",            red_map[3]["slot"] == 2 and red_map[3]["player"] == "Blu")
    check("RED T5 is Rati (slot 4)",              red_map[4]["slot"] == 4 and red_map[4]["player"] == "Rati")
    check("RED T6 is Marc (slot 3)",              red_map[5]["slot"] == 3 and red_map[5]["player"] == "Marc")
    check("RED T7 is Timmy (slot 5)",             red_map[6]["slot"] == 5 and red_map[6]["player"] == "Timmy")
    check("RED T8 is empty (slot 6 inactive)",    red_map[7]["active"] is False)

    # 2. Synthetic multi-track video generation & single-pass extraction
    with tempfile.TemporaryDirectory() as td:
        syn_vid = os.path.join(td, "synthetic_meld_session.mp4")
        
        # Build 8 audio streams + 1 video stream synthetic test file
        cmd_gen = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2:r=24",
            "-f", "lavfi", "-i", "sine=f=110:d=2",
            "-f", "lavfi", "-i", "sine=f=220:d=2",
            "-f", "lavfi", "-i", "sine=f=330:d=2",
            "-f", "lavfi", "-i", "sine=f=440:d=2",
            "-f", "lavfi", "-i", "sine=f=550:d=2",
            "-f", "lavfi", "-i", "sine=f=660:d=2",
            "-f", "lavfi", "-i", "sine=f=770:d=2",
            "-f", "lavfi", "-i", "sine=f=880:d=2",
            "-map", "0:v",
            "-map", "1:a", "-map", "2:a", "-map", "3:a", "-map", "4:a",
            "-map", "5:a", "-map", "6:a", "-map", "7:a", "-map", "8:a",
            "-c:v", "libx264", "-c:a", "aac",
            syn_vid
        ]
        
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        subprocess.run(cmd_gen, capture_output=True, check=True, startupinfo=startupinfo)
        check("Synthetic 8-stream video container generated", os.path.isfile(syn_vid))

        # Probe video
        probe = video_ingest.probe_video_streams(syn_vid)
        check("probe_video_streams succeeded", probe.get("success") is True)
        check("probe detected 8 audio streams", probe.get("num_audio_streams") == 8)

        # Single-pass extraction (SW5E)
        stems_dir = os.path.join(td, "stems_sw5e")
        extracted = video_ingest.extract_video_audio_stems(
            syn_vid,
            output_dir=stems_dir,
            campaign_mode="sw5e",
            log_func=lambda s: None
        )

        check("SW5E extracted 6 vocal slots", len(extracted) == 6)
        check("Music stem Track_Music.wav exported", os.path.isfile(os.path.join(stems_dir, "Track_Music.wav")))

        all_valid = True
        for slot_idx, wav_p in extracted.items():
            if not os.path.isfile(wav_p):
                all_valid = False
                break
            info = sf.info(wav_p)
            if info.samplerate != 48000 or "24" not in info.subtype:
                all_valid = False
                break
            data, _ = sf.read(wav_p, dtype="float32")
            if len(data) == 0 or np.isnan(data).any() or np.isinf(data).any():
                all_valid = False
                break

        check("All extracted stems are 24-bit PCM WAV (48 kHz) without NaN/Inf", all_valid)

        # Test auto_detect_session_tracks integration
        detected = mb.auto_detect_session_tracks(syn_vid, mode="sw5e")
        check("mb.auto_detect_session_tracks handles video container directly", len(detected) == 6)

except Exception as e:
    check("Video ingestion tests raised no exception", False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESULTS:  {PASS}/{total} PASSED  |  {FAIL} FAILED")
print(f"{'='*60}")
if FAIL == 0:
    print("  🎉 All chains verified — v4.3 is clean.")
else:
    print("  ⚠️  Some tests failed — review output above.")
    sys.exit(1)
