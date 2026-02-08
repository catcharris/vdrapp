import librosa
import numpy as np
import scipy.stats
import scipy.io.wavfile as wavfile
import io
from typing import Tuple, List, Optional, Dict
from src.utils import PASSAGGIO_CRITERIA

def generate_tone(frequency: float, duration_sec: float = 2.0, sample_rate: int = 44100) -> io.BytesIO:
    """Generates a sine wave tone for the given frequency."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate sine wave
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    # Convert to 16-bit PCM
    waveform_int16 = (waveform * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, waveform_int16)
    buf.seek(0)
    return buf

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Loads audio file."""
        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.array([]), self.sr

    def extract_features(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts pitch (F0) and energy (RMS) from audio.
        Returns: times, f0, rms, voiced_probs
        """
        if len(y) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Pitch Tracking using pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.sr,
            frame_length=2048,
            hop_length=512
        )
        
        # Energy (RMS)
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=self.sr, hop_length=hop_length)
        
        # Align lengths
        min_len = min(len(f0), len(rms), len(times))
        f0 = f0[:min_len]
        rms = rms[:min_len]
        times = times[:min_len]
        voiced_probs = voiced_probs[:min_len]
        
        # Octave Jump Suppression (>700 cents frame-to-frame)
        # Simple iterative filter: if jump is too large, invalidate current frame or smooth
        # Here we invalidate to be safe (treat as unvoiced/noise)
        cents_jump_threshold = 700
        
        cleaned_f0 = f0.copy()
        
        # Vectorized check is hard because it depends on previous valid value. 
        # Iterative approach is safer for this specific logic.
        last_valid = 0.0
        for i in range(len(cleaned_f0)):
            if np.isnan(cleaned_f0[i]):
                continue
            
            if last_valid > 0:
                # Calc diff in cents
                diff = 1200 * np.log2(cleaned_f0[i] / last_valid)
                if abs(diff) > cents_jump_threshold:
                    # Jump detected. Invalidate this frame.
                    cleaned_f0[i] = np.nan
                    continue 
            
            last_valid = cleaned_f0[i]
            
        return times, cleaned_f0, rms, voiced_probs

    def calculate_metrics(self, f0: np.ndarray, rms: np.ndarray, voiced_probs: np.ndarray, 
                          target_note: Optional[str] = None, voice_part: str = "Soprano") -> dict:
        """
        Calculates vocal metrics with strict validation.
        """
        metrics = {
            "accuracy": 999.0,   # Default to Bad (High Error)
            "stability": 999.0,  # Default to Unstable
            "drift": 999.0,      # Default to High Drift
            "overshoot": 0.0,
            "mean_pitch_hz": 0.0,
            "voiced_ratio": 0.0,
            "confidence": "High",
            "on_target_ratio": 0.0 # Default to 0%
        }
        
        # 1. Filter by Confidence & Voice Range
        confidence_threshold = 0.3 # stricter than default
        voice_limits = PASSAGGIO_CRITERIA.get(voice_part, {}).get("range_hz", [50, 2000])
        min_hz, max_hz = voice_limits[0], voice_limits[1]

        # Valid mask: Not NaN, Confidence high, Within Hz range
        valid_mask = (
            (~np.isnan(f0)) & 
            (voiced_probs > confidence_threshold) & 
            (f0 >= min_hz) & 
            (f0 <= max_hz)
        )
        
        voiced_f0 = f0[valid_mask]
        
        # Voiced Ratio
        total_frames = len(f0)
        if total_frames > 0:
            metrics["voiced_ratio"] = len(voiced_f0) / total_frames
        
        if len(voiced_f0) == 0:
            metrics["confidence"] = "Low"
            # Metrics remain at default (999.0 = Bad)
            return metrics

        if metrics["voiced_ratio"] < 0.7:  # User constraint: < 70% is low confidence
            metrics["confidence"] = "Low"

        metrics["mean_pitch_hz"] = float(np.mean(voiced_f0)) # Just for info

        # 2. Pitch Accuracy (Median, Octave Corrected, Clamped)
        if target_note:
            target_hz = librosa.note_to_hz(target_note)
            
            # Strict Accuracy: NO Octave Correction
            # User must sing the exact target note.
            
            # Calculate cents error directly against target_hz
            err = 1200 * np.log2(voiced_f0 / target_hz)
            min_cents_error = np.abs(err)
            
            # Metric 1: Accuracy (Mean Error) - Penalize drops significantly
            # Previously used Median, which hid short drops.
            metrics["accuracy"] = float(np.mean(min_cents_error))
            
            # Metric 2: On-Target Ratio
            # Percentage of frames within +/- 50 cents of target
            on_target_mask = min_cents_error <= 50.0
            metrics["on_target_ratio"] = float(np.mean(on_target_mask))
            
            # Clamp accuracy to 0-200 for sanity
            metrics["accuracy"] = min(200.0, metrics["accuracy"])
        else:
             metrics["on_target_ratio"] = 0.0
        
        # 3. Pitch Stability (Pre-Passaggio Primary)
        # Determine Pre/Post regions
        passaggio_hzs = PASSAGGIO_CRITERIA.get(voice_part, {}).get("passaggio_hz", [])
        primary_passaggio = passaggio_hzs[0] if passaggio_hzs else 300.0
        
        # Pre-Passaggio frames (< primary_passaggio)
        pre_mask = voiced_f0 < primary_passaggio
        post_mask = voiced_f0 >= primary_passaggio
        
        pre_f0 = voiced_f0[pre_mask]
        
        if len(pre_f0) > 10:
            # Calculate stability on Pre-region
            # Std dev in cents relative to *its own mean* (or target if available?)
            # Usually stability is fluctuation around the sung note. 
            # If it's a scale, std dev is huge. 
            # Assuming Sustained Note tests (T1, T6) mostly.
            # For scales, this metric might need separate logic (e.g. smoothness).
            # User said "Use Pre-passaggio stability".
            
            ref = np.mean(pre_f0)
            cents_dev = 1200 * np.log2(pre_f0 / ref)
            metrics["stability"] = float(np.std(cents_dev))
        else:
            # Fallback to whole if no pre-passaggio content (e.g. high soprano singing high)
            ref = np.mean(voiced_f0)
            cents_dev = 1200 * np.log2(voiced_f0 / ref)
            metrics["stability"] = float(np.std(cents_dev))

        # 4. Pitch Drift (End - Start)
        window = 20
        if len(voiced_f0) > window * 2:
            # Median of start/end to be robust to outliers
            start_pitch = np.median(voiced_f0[:window])
            end_pitch = np.median(voiced_f0[-window:])
            if start_pitch > 0 and end_pitch > 0:
                metrics["drift"] = 1200 * np.log2(end_pitch / start_pitch)

        return metrics
