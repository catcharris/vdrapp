import streamlit as st
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.models import StudentSession, TestResult, TagInstance
from src.utils import PARTS, PASSAGGIO_CRITERIA, TESTS
from src.audio_processor import AudioProcessor
from src.pdf_generator import PDFGenerator

# Constants
RECORDINGS_DIR = "recordings"
REPORTS_DIR = "reports"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize Session State
if 'session' not in st.session_state:
    st.session_state['session'] = StudentSession()
if 'current_test_index' not in st.session_state:
    st.session_state['current_test_index'] = 0

def save_uploaded_file(uploaded_file, test_id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{st.session_state['session'].id}_{test_id}_{timestamp}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

def generate_diagnosis(result, part):
    """Generates meaningful diagnostic text based on metrics."""
    diagnosis = []
    
    # Accuracy
    if result.pitch_accuracy_cents > 50:
        diagnosis.append(f"피치가 목표음보다 평균 {result.pitch_accuracy_cents:.1f} cents 벗어났습니다. (정확도 주의)")
    elif result.pitch_accuracy_cents < 20:
        diagnosis.append("피치 정확도가 매우 우수합니다.")

    # Stability
    if result.pitch_stability_cents > 30:
        diagnosis.append("음의 흔들림(Vibrato/Tremolo)이 다소 큽니다. 호흡 지탱을 확인하세요.")
    elif result.pitch_stability_cents < 10:
        diagnosis.append("음이 매우 안정적입니다 (Straight Tone).")

    # Drift
    if result.pitch_drift_cents < -20:
        diagnosis.append("끝음이 처지는 경향(Flat Drift)이 있습니다.")
    elif result.pitch_drift_cents > 20:
        diagnosis.append("끝음이 샵되는 경향(Sharp Drift)이 있습니다.")

    # Voiced Ratio
    # We need to add voiced_ratio to TestResult model first, or just use it here if returned
    # Assuming result object has it or we add it to the model.
    # For now, simplistic check if accuracy is 0 (which might imply no voiced frames found)
    
    if not diagnosis:
        diagnosis.append("전반적으로 안정적인 발성입니다. 세부 지표인 비브라토 속도 등을 체크해보세요.")

    return " ".join(diagnosis)

def analyze_audio(filepath, test_id, target_note=None, voice_part="Soprano"):
    processor = AudioProcessor()
    y, sr = processor.load_audio(filepath)
    times, f0, rms, voiced_probs = processor.extract_features(y)
    
    # Calculate Metrics with new strict logic
    metrics = processor.calculate_metrics(f0, rms, voiced_probs, target_note, voice_part)
    
    # Validation for Test 6 (After) vs Test 1 (Before)
    if test_id == "T6":
         t1_result = st.session_state['session'].get_result("T1")
         if t1_result:
             # Check duration similarity? 
             duration = times[-1] if len(times) > 0 else 0
             t1_duration = t1_result.pitch_track_time[-1] if t1_result.pitch_track_time else 0
             if abs(duration - t1_duration) > 2.0:
                 st.warning(f"⚠️ Warning: Test 6 duration ({duration:.1f}s) differs significantly from Test 1 ({t1_duration:.1f}s). Comparability may be low.")

    result = TestResult(
        test_id=test_id,
        test_name=TESTS[st.session_state['current_test_index']]['name'],
        audio_file_path=filepath,
        pitch_track_time=times.tolist(),
        pitch_track_hz=f0.tolist(), # Contains nans now
        energy_track_time=times.tolist(),
        energy_track_rms=rms.tolist(),
        pitch_accuracy_cents=metrics['accuracy'],
        pitch_stability_cents=metrics['stability'],
        pitch_drift_cents=metrics['drift'],
        attack_overshoot_score=metrics['overshoot'],
        processed_at=datetime.datetime.now()
    )
    
    # Generate and attach diagnosis tag
    diag_text = generate_diagnosis(result, voice_part)
    result.tags.append(TagInstance(tag_type="Diagnosis", description=diag_text))
    
    return result

def main():
    st.set_page_config(page_title="Vocal Diagnostic Report", layout="wide")
    st.title("🎤 Vocal Diagnostic Report (VDR)")

    # Sidebar: Setup
    with st.sidebar:
        st.header("1. Student Info")
        st.session_state['session'].student_name = st.text_input("Name", st.session_state['session'].student_name)
        st.session_state['session'].part = st.selectbox("Part", PARTS, index=PARTS.index(st.session_state['session'].part))
        st.session_state['session'].coach_name = st.text_input("Coach", st.session_state['session'].coach_name)
        
        # Update Passaggio Info
        st.session_state['session'].passaggio_info = PASSAGGIO_CRITERIA[st.session_state['session'].part]
        st.info(f"Passaggio: {st.session_state['session'].passaggio_info['desc']}")
        
        if st.button("Reset Session"):
            st.session_state['session'] = StudentSession()
            st.session_state['current_test_index'] = 0
            st.rerun()

    # Main Area: Test Flow
    tests = TESTS
    current_index = st.session_state['current_test_index']
    
    if current_index < len(tests):
        test = tests[current_index]
        st.header(f"Test {current_index + 1}/{len(tests)}: {test['name']}")
        st.markdown(f"**Instructions**: {test['description']}")
        st.markdown(f"_Duration Guide: {test['duration_guide']} seconds_")
        
        # Recording / Upload
        audio_value = st.audio_input(f"Record {test['name']}")
        
        uploaded_file = None
        if audio_value:
            uploaded_file = audio_value
        else:
            with st.expander("Or upload a file"):
                uploaded_file = st.file_uploader(f"Upload Recording for {test['id']}", type=['wav', 'mp3', 'm4a'], key=f"uploader_{test['id']}")
        
        if uploaded_file is not None:
            if st.button(f"Analyze {test['name']}", key=f"analyze_{test['id']}"):
                with st.spinner("Analyzing Audio..."):
                    filepath = save_uploaded_file(uploaded_file, test['id'])
                    
                    # Determine target note for accuracy
                    # For MVP, we map Part to a default target note for Sustained tests
                    # Soprano/Tenor -> F4, Alto/Baritone -> E4, Bass -> D4
                    target_map = {
                        "Soprano": "F4", "Tenor": "F4", 
                        "Alto": "E4", "Baritone": "E4", 
                        "Bass": "D4"
                    }
                    part = st.session_state['session'].part
                    target = target_map.get(part, "C4") if "Sustained" in test['name'] else None
                    
                    result = analyze_audio(filepath, test['id'], target, part)
                    st.session_state['session'].add_result(result)
                    st.success("Analysis Complete!")
        
        # Display Current Results
        current_result = st.session_state['session'].get_result(test['id'])
        if current_result:
            col1, col2, col3 = st.columns(3)
            col1.metric("Stability (Std Cents)", f"{current_result.pitch_stability_cents:.2f}")
            col2.metric("Drift (Cents)", f"{current_result.pitch_drift_cents:.2f}")
            col3.metric("Accuracy (Error)", f"{current_result.pitch_accuracy_cents:.2f}")
            
            # Show Diagnosis
            for tag in current_result.tags:
                if tag.tag_type == "Diagnosis":
                    st.info(f"**Diagnosis**: {tag.description}")
            
            st.subheader("Pitch & Energy Track")
            # Simple Plot using Matplotlib
            
            # Font Config
            try:
                import matplotlib.font_manager as fm
                FONT_PATH = os.path.join(os.path.dirname(__file__), 'assets/fonts/NanumGothic.ttf')
                prop = fm.FontProperties(fname=FONT_PATH)
                plt.rcParams['font.family'] = prop.get_name()
            except:
                prop = None

            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
            times = np.array(current_result.pitch_track_time)
            pitch = np.array(current_result.pitch_track_hz)
            energy = np.array(current_result.energy_track_rms)
            
            # Mask unvoiced for pitch plot
            voiced_mask = pitch > 0 
            ax[0].plot(times[voiced_mask], pitch[voiced_mask], '.')
            ax[0].set_ylabel("Freq (Hz)")
            ax[0].set_title("Pitch")
            
            ax[1].plot(times, energy, color='orange')
            ax[1].set_ylabel("Energy")
            ax[1].set_xlabel("Time (s)")
            
            st.pyplot(fig)
            
            if st.button("Next Test ->"):
                st.session_state['current_test_index'] += 1
                st.rerun()
                
    else:
        # All Tests Completed
        st.header("🎉 Diagnosis Complete!")
        st.success("All tests recorded and analyzed.")
        
        st.subheader("Coach's Final Comment")
        st.session_state['session'].coach_comment = st.text_area("Diagnosis & Observations", st.session_state['session'].coach_comment)
        st.session_state['session'].routine_assignment = st.text_area("Prescribed Routine", st.session_state['session'].routine_assignment)
        
        if st.button("Generate PDF Report"):
            try:
                gen = PDFGenerator()
                filename = f"VDR_Report_{st.session_state['session'].student_name}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
                filepath = os.path.join(REPORTS_DIR, filename)
                gen.generate_report(st.session_state['session'], filepath)
                
                with open(filepath, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name=filename, mime="application/pdf")
                st.success(f"Report Generated: {filename}")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
