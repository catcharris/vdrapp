import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os

class VideoProcessor:
    def __init__(self):
        # Fix for "module 'mediapipe' has no attribute 'solutions'"
        if not hasattr(mp, 'solutions'):
            try:
                import mediapipe.python.solutions as solutions
                mp.solutions = solutions
            except ImportError:
                pass

        self.mp_face_mesh = mp.solutions.face_mesh
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
        Returns a DataFrame with timestamps and metrics.
        """
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file_path}. Codec or path issue.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        data = []
        
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
                    # Extract Landmarks
                    # Mouth: 
                    # Upper Lip Top: 13
                    # Lower Lip Bottom: 14
                    # Left Corner: 61
                    # Right Corner: 291
                    
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
                    
                    # Mouth Aspect Ratio (MAR) -> Openness
                    # If Horizontal is huge compared to Vertical -> "Spread/Tight"
                    # If Vertical is huge -> "Open/Dropped Jaw" (Good for singing)
                    
                    # Tension Metric (heuristic): Horizontal / Vertical
                    # Normal singing (Ah): Ratio ~ 0.5 - 1.0?
                    # Tight smile (Ee): Ratio > 1.5?
                    
                    ratio = horizontal_dist / (vertical_dist + 1e-6) # Avoid div/0
                    
                    data.append({
                        "time": timestamp,
                        "vertical_opening": vertical_dist,
                        "horizontal_spread": horizontal_dist,
                        "tension_ratio": ratio
                    })
            
            frame_count += 1
            
        cap.release()
        return pd.DataFrame(data)

    def generate_tension_chart(self, df):
        import plotly.express as px
        if df.empty:
            return None
            
        fig = px.line(df, x="time", y="tension_ratio", title="Mouth Tension Analysis (Horizontal / Vertical Ratio)")
        # Add 'Good' zone?
        # Generally, we want vertical opening (lower ratio).
        # High ratio = Wide mouth / Closed jaw -> Bad tension?
        return fig
