# Vocal Diagnostic Report - Utilities & Constants

# Part Definitions
PARTS = ["Soprano", "Alto", "Tenor", "Baritone", "Bass"]

# Passaggio Definitions (Note name to Frequency mapping can be done via librosa)
# Format: [Lower_Passaggio, Upper_Passaggio] (Approximate)
# We will use Note Names for display and Frequency for analysis logic if needed.
# For MVP, we store the text descriptions.
PASSAGGIO_CRITERIA = {
    # Approx frequencies: F4=349Hz, G4=392Hz, E4=330Hz, D4=293Hz, Eb4=311Hz
    # C2=65Hz, C6=1046Hz (General limits)
    
    "Soprano": {
        "range_hz": [220, 1200], # A3 to D6 approx
        "passaggio_hz": [349.23, 392.00], # F4, G4
        "desc": "Primo (F4) / Secondo (G4)"
    },
    "Alto": {
        "range_hz": [174, 700], # F3 to F5 approx
        "passaggio_hz": [329.63, 349.23], # E4, F4
        "desc": "Primo (E4) / Secondo (F4)"
    },
    "Tenor": {
        "range_hz": [130, 523], # C3 to C5 approx
        "passaggio_hz": [349.23], # F4
        "desc": "Secondo (F4)"
    },
    "Baritone": {
        "range_hz": [98, 392], # G2 to G4 approx
        "passaggio_hz": [329.63], # E4
        "desc": "Secondo (E4)"
    },
    "Bass": {
        "range_hz": [65, 330], # C2 to E4 approx
        "passaggio_hz": [293.66, 311.13], # D4, Eb4
        "desc": "Primo (D4) / Secondo (Eb4)"
    },
}

# Test Definitions
TESTS = [
    {
        "id": "T1",
        "name": "Sustained Note (Before)",
        "description": "기준음 5초 지속 (파트별 기준음)",
        "duration_guide": 5,
        "type": "audio"
    },
    {
        "id": "T2",
        "name": "Messa di Voce",
        "description": "작게 -> 크게 -> 작게 (6~8초)",
        "duration_guide": 8,
        "type": "audio"
    },
    {
        "id": "T3",
        "name": "Vowel Transition",
        "description": "아-에-이 또는 아-오-우 (모음 전환)",
        "duration_guide": 8,
        "type": "audio+video_optional"
    },
    {
        "id": "T4",
        "name": "Scale (Passaggio)",
        "description": "빠사지오 구간을 포함한 짧은 스케일",
        "duration_guide": 10,
        "type": "audio"
    },
    {
        "id": "T5",
        "name": "Choir Phrase",
        "description": "짧은 합창 프레이즈 (10~15초)",
        "duration_guide": 15,
        "type": "audio"
    },
    {
        "id": "T6",
        "name": "Sustained Note (After)",
        "description": "Test 1과 동일한 기준음 재측정 (교정 후)",
        "duration_guide": 5,
        "type": "audio"
    }
]

def get_frequency_from_note(note_name):
    # This can be implemented using librosa.note_to_hz later
    pass
