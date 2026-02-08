import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from .models import StudentSession

# Register Font
FONT_PATH = os.path.join(os.path.dirname(__file__), '../../VDR_Project/assets/fonts/NanumGothic.ttf')
# Resolving absolute path relative to this file location
FONT_PATH = os.path.abspath(FONT_PATH)

try:
    pdfmetrics.registerFont(TTFont('NanumGothic', FONT_PATH))
    font_name = 'NanumGothic'
except Exception as e:
    print(f"Font Load Error: {e}")
    font_name = 'Helvetica' # Fallback

class PDFGenerator:
    def __init__(self):
        self.width, self.height = A4
        self.styles = getSampleStyleSheet()
        self.create_custom_styles()

    def create_custom_styles(self):
        # Override Normal style with Korean font
        self.styles['Normal'].fontName = font_name
        self.styles['Normal'].fontSize = 12
        self.styles['Normal'].leading = 16
        
        # Apply Korean Font to custom styles
        self.styles.add(ParagraphStyle(name='Header', fontName=font_name, fontSize=18, leading=22, spaceAfter=10, alignment=1)) # Center
        self.styles.add(ParagraphStyle(name='SubHeader', fontName=font_name, fontSize=14, leading=18, spaceAfter=6, textColor=colors.darkblue))
        # NormalSmall inherits from Normal if updated? No, new instance.
        self.styles.add(ParagraphStyle(name='NormalSmall', fontName=font_name, fontSize=10, leading=12))
        self.styles.add(ParagraphStyle(name='NormalKorean', fontName=font_name, fontSize=10, leading=12))

    def create_charts(self, session: StudentSession) -> BytesIO:
        """Generates charts for Before/After comparison."""
        # Configure Matplotlib Font
        try:
            prop = fm.FontProperties(fname=FONT_PATH)
            plt.rcParams['font.family'] = prop.get_name()
        except:
            pass
            
        # Create figure using OO Application interface
        fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        
        # Test 1 (Before)
        t1 = session.get_result("T1")
        if t1 and t1.pitch_track_hz:
            times = np.array(t1.pitch_track_time)
            pitch = np.array(t1.pitch_track_hz)
            voiced_mask = pitch > 0
            ax[0].plot(times[voiced_mask], pitch[voiced_mask], label='Before (Test 1)', color='blue')
            ax[0].set_ylabel("Freq (Hz)")
            ax[0].legend(prop=prop)
            ax[0].grid(True, alpha=0.3)

        # Test 6 (After)
        t6 = session.get_result("T6")
        if t6 and t6.pitch_track_hz:
            times = np.array(t6.pitch_track_time)
            pitch = np.array(t6.pitch_track_hz)
            voiced_mask = pitch > 0
            ax[0].plot(times[voiced_mask], pitch[voiced_mask], label='After (Test 6)', color='green', linestyle='--')
            ax[0].legend(prop=prop)

        ax[0].set_title("Pitch Stability Comparison", fontproperties=prop)

        # Energy Comparison
        if t1 and t1.energy_track_rms:
            times = np.array(t1.energy_track_time)
            energy = np.array(t1.energy_track_rms)
            ax[1].plot(times, energy, label='Before', color='orange')
        
        if t6 and t6.energy_track_rms:
            times = np.array(t6.energy_track_time)
            energy = np.array(t6.energy_track_rms)
            ax[1].plot(times, energy, label='After', color='red', linestyle='--')
            
        ax[1].set_ylabel("Energy (RMS)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend(prop=prop)
        ax[1].grid(True, alpha=0.3)
        ax[1].set_title("Energy/Breath Control Comparison", fontproperties=prop)

        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer

    def generate_report(self, session: StudentSession, filepath: str):
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                                rightMargin=30, leftMargin=30,
                                topMargin=30, bottomMargin=18)
        story = []

        # 1. Header
        header_text = f"Vocal Diagnostic Report - {session.student_name}"
        story.append(Paragraph(header_text, self.styles['Header']))
        
        info_data = [
            [f"Part: {session.part}", f"Date: {session.created_at.strftime('%Y-%m-%d')}"],
            [f"Coach: {session.coach_name}", f"Passaggio: {session.passaggio_info.get('range', 'N/A')}"]
        ]
        t = Table(info_data, colWidths=[250, 250])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), font_name), # Apply font to table content
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # 2. Before / After Graphs
        story.append(Paragraph("Change Overview (Before vs After)", self.styles['SubHeader']))
        chart_buffer = self.create_charts(session)
        im = Image(chart_buffer, width=500, height=250)
        story.append(im)
        story.append(Spacer(1, 12))

        # 3. Metrics Summary Table
        story.append(Paragraph("Vocal Metrics", self.styles['SubHeader']))
        
        # Helper to format metrics
        def fmt_metric(res, key):
            if not res: return "-"
            val = getattr(res, key, 0.0)
            return f"{val:.2f}"

        t1 = session.get_result("T1")
        t6 = session.get_result("T6")
        
        metrics_data = [
            ["Metric", "Before (Test 1)", "After (Test 6)", "Change"],
            ["Pitch Stability (std cents)", fmt_metric(t1, 'pitch_stability_cents'), fmt_metric(t6, 'pitch_stability_cents'), "->"],
            ["Pitch Drift (cents)", fmt_metric(t1, 'pitch_drift_cents'), fmt_metric(t6, 'pitch_drift_cents'), "->"],
            ["Accuracy (cents error)", fmt_metric(t1, 'pitch_accuracy_cents'), fmt_metric(t6, 'pitch_accuracy_cents'), "->"]
        ]
        
        table = Table(metrics_data, colWidths=[150, 100, 100, 100])
        table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), font_name), # Apply font to metrics table
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            #('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), # Removed Helvetica bold
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # 4. Top Tags & Coach Comment
        story.append(Paragraph("Diagnosis & Feedback", self.styles['SubHeader']))
        
        if session.summary_tags:
            for tag in session.summary_tags:
                story.append(Paragraph(f"• <b>{tag.tag_type}</b>: {tag.description}", self.styles['NormalSmall']))
        else:
             story.append(Paragraph("No specific issues detected.", self.styles['NormalSmall']))

        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Coach Comment:</b>", self.styles['Normal']))
        story.append(Paragraph(session.coach_comment or "(No comment)", self.styles['Normal']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Routine Assignment:</b>", self.styles['Normal']))
        story.append(Paragraph(session.routine_assignment or "(No assignment)", self.styles['Normal']))

        doc.build(story)
