import pretty_midi
import numpy as np
import io
import scipy.io.wavfile as wavfile

def synthesis_midi_track(midi_data, track_index: int, fs=44100) -> io.BytesIO:
    """
    Synthesizes a specific MIDI track to audio (sine wave).
    Returns wav bytes buffer.
    """
    # Create a new MIDI object with only the selected instrument
    temp_midi = pretty_midi.PrettyMIDI()
    
    # Copy the selected instrument
    if 0 <= track_index < len(midi_data.instruments):
        inst = midi_data.instruments[track_index]
        temp_midi.instruments.append(inst)
    else:
        return None

    # Synthesize
    # sine wave synthesis is supported by pretty_midi via synthesize() function (uses sine wave by default if no fluid)
    # properly it uses 'fluidsynth' if installed, or simple synthesized wave.
    # verify if synthesize() works without fluidsynth. 
    # pretty_midi.synthesize() doc: "Synthesize the MIDI using a sine wave... if fs is not None"
    
    audio_data = temp_midi.synthesize(fs=fs)
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
    
    # Convert to 16-bit PCM
    waveform_int16 = (audio_data * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    wavfile.write(buf, fs, waveform_int16)
    buf.seek(0)
    return buf

def get_midi_tracks(midi_file) -> list:
    """Returns list of (index, name, program) for tracks."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        tracks = []
        for i, inst in enumerate(midi_data.instruments):
            name = inst.name if inst.name else f"Track {i+1}"
            tracks.append({"index": i, "name": name, "program": inst.program})
        return tracks, midi_data
    except Exception as e:
        print(f"Error parsing MIDI: {e}")
        return [], None
