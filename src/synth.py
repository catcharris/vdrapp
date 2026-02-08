import numpy as np
import soundfile as sf
import io

def generate_piano_note(freq, duration, sr=44100):
    """
    Generates a single piano-like note using additive synthesis and ADSR envelope.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Piano Physics Approximation:
    # 1. Harmonics: Fundamental is strong, overtones decay faster.
    # 2. Inharmonicity: Stiff strings (skip for MVP reliability).
    # 3. Envelope: Percussive attack, long decay.
    
    # 1. Base Waveform (Sum of Sines)
    # Fundamental
    wave = 1.0 * np.sin(2 * np.pi * freq * t)
    # 2nd Harmonic
    wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    # 3rd Harmonic
    wave += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    # 4th Harmonic
    wave += 0.12 * np.sin(2 * np.pi * freq * 4 * t)
    
    # 2. Amplitude Envelope (ADSR)
    # Attack: 0.01s (Fast hammer strike)
    # Decay: Exponential decay based on duration
    # Sustain/Release: Merged into decay for simplicity (piano style)
    
    attack_time = 0.01
    attack_samples = int(attack_time * sr)
    
    envelope = np.ones_like(t)
    
    # Attack Phase
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay Phase (Exponential)
    # Decay rate depends on frequency (Higher notes decay faster)
    decay_rate = 3.0 # Tuning parameter
    decay_curve = np.exp(-decay_rate * t)
    
    # Apply Decay from end of attack
    if attack_samples < len(t):
        envelope[attack_samples:] = decay_curve[attack_samples:]
        
    final_wave = wave * envelope
    return final_wave

def synthesize_midi_with_piano(midi_data, track_index, sr=44100):
    """
    Synthesizes a MIDI track using the custom piano generator.
    Handles polyphony by mixing arrays.
    """
    # 1. Determine total duration
    length_sec = midi_data.get_end_time() + 1.0 # Add tail
    total_samples = int(sr * length_sec)
    
    # Main mix buffer
    mix_buffer = np.zeros(total_samples)
    
    # Get track
    if not (0 <= track_index < len(midi_data.instruments)):
        return None
    track = midi_data.instruments[track_index]
    
    # Iterate Notes
    for note in track.notes:
        start_time = note.start
        end_time = note.end
        duration = end_time - start_time
        freq = 440 * (2 ** ((note.pitch - 69) / 12))
        
        # Generate Note Audio
        # Limit note duration to avoid huge arrays if MIDI is weird, but usually fine.
        # Add a bit of release time?
        actual_duration = duration + 0.5 # Let it ring a bit
        note_audio = generate_piano_note(freq, actual_duration, sr)
        
        # Calculate start index
        start_idx = int(start_time * sr)
        end_idx = start_idx + len(note_audio)
        
        # Add to mix (Handle bounds)
        if end_idx > total_samples:
            # Crop note or extend buffer? Crop for now.
            valid_len = total_samples - start_idx
            mix_buffer[start_idx:total_samples] += note_audio[:valid_len]
        else:
            mix_buffer[start_idx:end_idx] += note_audio
            
    # Normalize
    max_val = np.max(np.abs(mix_buffer))
    if max_val > 0:
        mix_buffer = mix_buffer / max_val
        
    # soundfile expects float32/float64 [-1, 1] usually, or int16.
    # Let's stick to float32 for soundfile
    waveform_float = mix_buffer.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, waveform_float, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf
