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

    def process_video(self, video_file_path):
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
        max_vertical_dist = -1
        max_frame_rgb = None
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
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
                    
                    upper = get_coords(13)
                    lower = get_coords(14)
                    left = get_coords(61)
                    right = get_coords(291)
                    
                    # Vertical Opening
                    vertical_dist = np.linalg.norm(upper - lower)
                    
                    # Horizontal Spread
                    horizontal_dist = np.linalg.norm(left - right)
                    
                    # Capture Max Opening Frame (Thumbnail Candidate)
                    if vertical_dist > max_vertical_dist:
                        max_vertical_dist = vertical_dist
                        max_frame_rgb = image_rgb.copy()
                    
                    # Tension Metric: Width / Height
                    # If Ratio > 5.0 -> Likely Mouth Closed
                    # Normal Singing (Ah) -> Ratio 0.8 ~ 1.5
                    ratio = horizontal_dist / (vertical_dist + 1e-6)
                    
                    data.append({
                        "time": timestamp,
                        "vertical_opening": vertical_dist,
                        "horizontal_spread": horizontal_dist,
                        "tension_ratio": ratio
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
            
        # Filter insane ratios (e.g. mouth closed) for better chart scaling
        # Cap ratio at 5.0 for visualization
        df['display_ratio'] = df['tension_ratio'].clip(upper=5.0)
            
        fig = px.line(df, x="time", y="display_ratio", title="구강 긴장도 (가로/세로 비율)")
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="긴장 구간 (가로 벌어짐)")
        fig.add_hline(y=1.0, line_dash="dot", line_color="green", annotation_text="이상적 발성 (1.0 이하)")
        
        fig.update_layout(
            yaxis_title="비율 (낮을수록 좋음: 수직 개방)", 
            xaxis_title="시간 (초)",
            yaxis_range=[0, 5.0]
        )
        return fig
