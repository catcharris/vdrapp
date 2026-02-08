import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os

class VideoProcessor:
    def __init__(self):
        import mediapipe as mp
        import streamlit as st
        
        # Debugging MediaPipe Integrity
        try:
            st.toast(f"MP Version: {mp.__version__}")
            if hasattr(mp, 'solutions'):
                self.mp_face_mesh = mp.solutions.face_mesh
                st.toast("MediaPipe Solutions Loaded ✅")
            else:
                st.error("MediaPipe loaded but has no 'solutions' attribute!")
                # Attempt manual submodule import as last ditch
                import mediapipe.python.solutions.face_mesh as fm
                self.mp_face_mesh = fm
                st.warning("Force imported face_mesh from mediapipe.python.solutions")
        except Exception as e:
            st.error(f"MediaPipe Import Error: {e}")
            raise e

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_video(self, video_file_path, rotate=False):
        """
        Processes video to extract facial tension metrics.
        Returns (DataFrame, Max_Opening_Frame_RGB).
        """
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file_path}. Codec or path issue.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        data = []
        max_openness = -1
        max_frame_rgb = None
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Rotation Correction (Simple 90 deg clockwise if requested)
            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            timestamp = frame_count / fps
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, c = image.shape
                    
                    def get_coords(idx):
                        lm = face_landmarks.landmark[idx]
                        return np.array([lm.x * w, lm.y * h])
                    
                    # Landmarks
                    # 13: Inner Upper Lip, 14: Inner Lower Lip
                    # 10: Forehead Top, 152: Chin Bottom
                    
                    upper = get_coords(13)
                    lower = get_coords(14)
                    forehead = get_coords(10)
                    chin = get_coords(152)
                    
                    # Vertical Opening
                    vertical_dist = np.linalg.norm(upper - lower)
                    
                    # Face Height (Normalization Base)
                    face_height = np.linalg.norm(forehead - chin)
                    
                    # Normalized Openness (%)
                    # 0% = Closed
                    # 10% = Moderate
                    # 30%+ = Wide Open (Screaming/Singing High Note)
                    openness = (vertical_dist / (face_height + 1e-6)) * 100
                    
                    # Capture Max Opening Frame
                    if openness > max_openness:
                        max_openness = openness
                        max_frame_rgb = image_rgb.copy()
                    
                    data.append({
                        "time": timestamp,
                        "vertical_opening": vertical_dist,
                        "face_height": face_height,
                        "openness": openness
                    })
            
            frame_count += 1
            
        cap.release()
        
        if not data:
            return None, None
            
        return pd.DataFrame(data), max_frame_rgb

    def generate_tension_chart(self, df):
        import plotly.express as px
        if df.empty:
            return None
            
        # Chart: Openness %
        fig = px.line(df, x="time", y="openness", title="구강 개방도 (얼굴 길이 대비 입 벌림 %)")
        
        # Benchmarks
        fig.add_hline(y=10.0, line_dash="dot", line_color="green", annotation_text="적정 발성 (10%~)")
        fig.add_hline(y=25.0, line_dash="dash", line_color="red", annotation_text="최대 개방 (25%~)")
        
        fig.update_layout(
            yaxis_title="개방도 (%)", 
            xaxis_title="시간 (초)",
            yaxis_range=[0, 40.0] # Scale 0 to 40%
        )
        return fig
