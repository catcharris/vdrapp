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
    if result.pitch_accuracy_cents >= 900.0:
        diagnosis.append("음정이 감지되지 않았거나 분석에 실패했습니다. (소음/무음)")
    elif result.pitch_accuracy_cents > 50:
        diagnosis.append(f"피치가 목표음보다 평균 {result.pitch_accuracy_cents:.1f} cents 벗어났습니다. (정확도 주의)")
    elif result.pitch_accuracy_cents < 20:
        diagnosis.append("피치 정확도가 매우 우수합니다.")

    # Stability
    if result.pitch_stability_cents >= 900.0:
        pass # Already handled by accuracy failure message
    elif result.pitch_stability_cents > 30:
        diagnosis.append("음의 흔들림(Vibrato/Tremolo)이 다소 큽니다. 호흡 지탱을 확인하세요.")
    elif result.pitch_stability_cents < 10 and result.pitch_stability_cents >= 0:
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
    
    # New Strict Check: On-Target Ratio
    if result.pitch_on_target_ratio < 0.6:
        diagnosis.append("음정이 불안정하여 목표음을 많이 벗어납니다. (정확도 < 60%)")
    elif result.pitch_on_target_ratio < 0.85:
        diagnosis.append("중간중간 음정이 흔들립니다. 호흡 지탱에 신경쓰세요.")
    
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
        pitch_on_target_ratio=metrics.get('on_target_ratio', 0.0),
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

    # Sidebar
    with st.sidebar:
        st.title("VDR Settings")
        
        # Version & Environment Check (v1.16 Stable)
        import sys
        current_py = sys.version.split()[0]
        # Check if Python is 3.11 (Good) or 3.13 (Bad)
        is_safe_version = sys.version_info < (3, 12) 
        
        if is_safe_version:
            st.caption(f"v1.16 (Python {current_py} OK) ✅")
        else:
            st.error(f"⚠️ Old Server Detected ({current_py})")
            st.caption("Please Delete & Re-deploy App")
        
        # User Profile
        st.subheader("Student Profile")
    
    # Main Page Setup
    if not is_safe_version:
         st.warning(f"⚠️ Server is updating... (Python {current_py})")
         st.markdown("---")
    with st.sidebar:
        st.session_state['session'].student_name = st.text_input("Name", st.session_state['session'].student_name)
        st.session_state['session'].part = st.selectbox("Part", PARTS, index=PARTS.index(st.session_state['session'].part))
        st.session_state['session'].coach_name = st.text_input("Coach", st.session_state['session'].coach_name)
        
        # Update Passaggio Info
        st.session_state['session'].passaggio_info = PASSAGGIO_CRITERIA[st.session_state['session'].part]
        st.info(f"Passaggio: {st.session_state['session'].passaggio_info['desc']}")
        
        # Debug Info
        with st.expander("🛠️ Debug Info (Show to Developer)", expanded=True):
            import sys
            import subprocess
            
            st.code(f"Python: {sys.version.split()[0]}")
            
            # Show pip freeze to debug installation
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
                installed_packages = result.stdout
                st.text("Installed Packages:")
                st.code(installed_packages, language="text", line_numbers=True)
            except Exception as e:
                st.error(f"Failed to list packages: {e}")

            # Specific Checks
            try:
                import mediapipe as mp
                st.success(f"MediaPipe: {mp.__version__}")
                st.write(f"Has solutions? {'✅' if hasattr(mp, 'solutions') else '❌'}")
            except ImportError:
                st.error("MediaPipe: Not Installed")
                
            try:
                import cv2
                st.success(f"OpenCV: {cv2.__version__}")
            except ImportError:
                st.error("OpenCV: Not Installed")
        
        st.markdown("---")
        
        # MIDI Part Player
        st.subheader("🎹 MIDI Part Practice")
        midi_file = st.file_uploader("Upload MIDI File", type=["mid", "midi"])
        
        if midi_file:
            from src.midi_handler import get_midi_tracks, synthesis_midi_track
            # Read file pointer compatible with pretty_midi
            # pretty_midi expects file path or file-like object
            tracks, midi_data = get_midi_tracks(midi_file)
            
            if tracks:
                track_names = [f"{t['index']}: {t['name']}" for t in tracks]
                selected_track_str = st.selectbox("Select Part to Play", track_names)
                selected_index = int(selected_track_str.split(":")[0])
                
                if st.button("Generate & Play Part"):
                    with st.spinner("Synthesizing Audio..."):
                        wav_bytes = synthesis_midi_track(midi_data, selected_index)
                        if wav_bytes:
                            st.audio(wav_bytes, format='audio/wav')
                        else:
                            st.error("Failed to synthesize track.")
            else:
                st.error("No tracks found or invalid MIDI.")
        
        st.markdown("---")

        if st.button("Reset Session"):
            st.session_state['session'] = StudentSession()
            st.session_state['current_test_index'] = 0
            st.rerun()

    # Main Area: Test Flow
    tests = TESTS
    current_index = st.session_state['current_test_index']
    
    if current_index < len(tests):
        test = tests[current_index]
        st.subheader(f"Test {current_index + 1}/{len(tests)}: {test['name']}")

    # Tabs for Audio / Video
    tab1, tab2 = st.tabs(["🎤 Audio Analysis", "📹 Video Analysis (Face Tension)"])

    with tab1:
        st.markdown(f"**Instructions**: {test['description']}")
        
        # Display Target Note for Sustained Tests
        if "Sustained" in test['name']:
            target_map = {
                "Soprano": "F5 (698 Hz)", 
                "Alto": "E5 (659 Hz)", 
                "Tenor": "F4 (349 Hz)", 
                "Baritone": "E4 (330 Hz)", 
                "Bass": "Eb4 (311 Hz)"
            }
            target_note = target_map.get(st.session_state['session'].part, "C4").split(" ")[0] # Extract "F4" from "F4 (349 Hz)"
            st.info(f"🎵 **Target Note (Passaggio Start)**: {target_map.get(st.session_state['session'].part)}")
            
            # Play Reference Pitch
            try:
                # Extract Hz from string "F4 (349 Hz)" -> 349
                target_str = target_map.get(st.session_state['session'].part, "C4 (261 Hz)")
                hz_str = target_str.split("(")[1].split(" ")[0]
                target_hz = float(hz_str)
                
                # Use Piano Synth for Reference Pitch
                from src.synth import generate_piano_note
                import soundfile as sf
                import io
                import numpy as np
                
                # Generate 2 seconds of Piano C4/F4/etc
                waveform = generate_piano_note(target_hz, duration=2.0)
                # Normalize
                waveform = waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform
                
                buf = io.BytesIO()
                sf.write(buf, waveform, 44100, format='WAV', subtype='PCM_16')
                buf.seek(0) 
                tone_bytes = buf
                
                st.audio(tone_bytes, format='audio/wav', start_time=0)
                st.caption("🎹 Play Reference Pitch (Piano)")
            except Exception as e:
                st.warning(f"Audio Playback Error: {e}")
        
        st.markdown(f"_Duration Guide: {test['duration_guide']} seconds_")
        
        # Recording / Upload
        audio_value = st.audio_input(f"Record {test['name']}")
        
        if audio_value:
            st.audio(audio_value, format='audio/wav')
            
            with st.spinner("Analyzing..."):
                # Save to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_value.read())
                    tmp_path = tmp_file.name
                
                # Analyze
                from src.audio_processor import AudioProcessor
                processor = AudioProcessor()
                y, sr = processor.load_audio(tmp_path)
                metrics = processor.calculate_metrics(y, sr, target_note=test.get('target_note'))
                
                st.success("Analysis Complete!")
                
                # Diagnosis
                diagnosis = generate_diagnosis(metrics, test['name'])
                st.session_state['session'].add_result(
                    test_name=test['name'],
                    metrics=metrics,
                    diagnosis=diagnosis
                )
                
                # Display Result
                col1, col2, col3 = st.columns(3)
                col1.metric("Pitch Accuracy (Cents)", f"{metrics['accuracy']:.1f}", delta_color="inverse")
                col2.metric("Stability (Std Dev)", f"{metrics['stability']:.1f}", delta_color="inverse")
                col3.metric("Drift (Slope)", f"{metrics['drift']:.2f}")
                
                 # Feedback based on On-Target Ratio
                if metrics['on_target_ratio'] < 0.6:
                    st.warning(f"⚠️ **Unstable Pitch**: Only {metrics['on_target_ratio']*100:.1f}% of frames were on target.")
                else:
                    st.success(f"✅ **Stable Pitch**: {metrics['on_target_ratio']*100:.1f}% on target.")

                # Diagnosis List
                st.write("### 🩺 Diagnosis")
                for item in diagnosis:
                    st.write(f"- {item}")
                
                # Graphs
                st.plotly_chart(generate_pitch_plot(y, sr, metrics['mean_pitch_hz']), use_container_width=True)

    with tab2:
        st.markdown("### 📹 Facial Tension Analysis")
        st.info("💡 **팁:** 모바일에서는 'Browse files' -> 'Camera'를 선택하여 바로 촬영할 수 있습니다!")
        
        with st.expander("📘 **[필독] 분석 결과 해석 및 촬영 가이드**"):
            st.markdown("""
            ### 1. 📉 그래프 보는 법 (구강 개방도 %)
            - **측정 방식**: 얼굴 전체 길이 대비 입이 세로로 얼마나 열렸는지 측정합니다. (Scale-Independent)
            - **수치 해석**:
                - **10% ~ 20%**: 안정적인 발성 (적당한 열림) ✅
                - **25% 이상**: 매우 크게 벌린 상태 (성량 확보/고음) 🔥
                - **10% 미만**: 입을 거의 안 벌림 (웅얼거림 주의) ⚠️
            
            ### 2. 🔄 영상 회전 기능
            - 만약 캡처된 사진이 **옆으로 누워 보인다면**, 위쪽의 **'영상 회전'** 체크박스를 켜고 다시 분석하세요!

            ### 3. 📸 촬영 팁
            - **'아~'** 발음으로 입을 위아래로 크게 벌려보세요.
            """)

        # Rotation Checkbox
        rotate_video = st.checkbox("🔄 영상 회전 (화면이 누워있으면 체크하세요)", value=False)
        
        video_file = st.file_uploader("Upload Video (or Record on Mobile)", type=["mp4", "mov", "avi"])
        
        if video_file:
            # Display Video
            st.video(video_file)
            
            if st.button("Analyze Face Tension"):
                with st.spinner("Processing Video (MediaPipe Face Mesh)..."):
                    try:
                        import tempfile
                        # Create temp file, write, and CLOSE it so other libs can read it
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                        tfile.write(video_file.read())
                        tfile.close() # Critical: Close before OpenCV opens it
                        
                        from src.video_processor import VideoProcessor
                        vp = VideoProcessor()
                        
                        # Pass rotation flag
                        df, max_frame = vp.process_video(tfile.name, rotate=rotate_video)
                        
                        # Cleanup
                        os.unlink(tfile.name)
                        
                        if df is not None and not df.empty:
                            st.success("분석 완료! (Analysis Complete)")
                            
                            # Show Max Opening Frame (Thumbnail)
                            if max_frame is not None:
                                st.write("### 📸 최대 개방 순간 (Best Open Mouth)")
                                st.image(max_frame, caption="가장 입을 크게 벌린 순간", use_container_width=True)
                            
                            # Show Chart
                            fig = vp.generate_tension_chart(df)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            avg_openness = df['openness'].max() # Use MAX openness, not mean, as per user feedback ("I opened wide")
                            # Or usually avg of top 10%? Let's show Max for "Capacity"
                            
                            st.metric("최대 구강 개방도 (Max Openness)", f"{avg_openness:.1f} %")
                            
                            if avg_openness < 10.0:
                                st.warning(f"⚠️ **입을 작게 벌리셨네요** (최대 {avg_openness:.1f}%)")
                                st.info("성악 발성에서는 '아' 발음 시 입을 세로로 충분히(20% 이상) 벌려야 소리가 멀리 나갑니다.")
                            elif avg_openness > 25.0:
                                st.success(f"🔥 **아주 크게 벌리셨습니다!** (최대 {avg_openness:.1f}%)")
                                st.write("시원시원한 발성입니다. 턱에 힘이 들어가지 않도록 주의하세요.")
                            else:
                                st.success(f"✅ **적당한 개방도입니다** (최대 {avg_openness:.1f}%)")
                                st.write("안정적인 발성 폼입니다.")
                                
                        else:
                            st.error("영상에서 얼굴을 찾을 수 없습니다. (회전 옵션을 바꿔보세요)")
                            
                    except Exception as e:
                        import traceback
                        st.error(f"Video Processing Error: {e}")
                        st.code(traceback.format_exc())

    st.markdown("---")
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
