import matplotlib.pyplot as plt
import io
import numpy as np

def draw_staff_and_note(note_name, clef_type="Treble"):
    """
    Draws a simple musical staff with a single note.
    note_name: "F4", "E4", "D4"
    clef_type: "Treble" or "Bass"
    """
    # Fix Dark Mode: Use White background (Paper style)
    fig, ax = plt.subplots(figsize=(3, 1.8), facecolor='white')
    ax.set_axis_off()
    
    # Draw Staff Lines (y=0, 2, 4, 6, 8)
    y_lines = [0, 2, 4, 6, 8]
    for y in y_lines:
        ax.hlines(y, xmin=0, xmax=10, colors='black', linewidth=1)
        
    # Draw Clef (Manual Shapes)
    if clef_type == "Treble":
        # Draw Simplified G-Clef (Spiral)
        # Center approx (1.5, 3) -> Line 2 (G)
        t = np.linspace(0, 10, 100)
        # Vertical line
        ax.plot([1, 1], [-2, 11], color='black', linewidth=1.5)
        # Top loop
        ax.plot(1 + 0.5*np.sin(t), 7 + 3*np.cos(t/2), color='black', linewidth=1) # Fake shape
        # Label
        ax.text(1, -4, "G-Clef", fontsize=8, ha='center')
        
        # Note Mapping (Treble)
        # E4 = Line 1 (y=0)
        base_y = 0 
        base_note_idx = 0 # E4 index in diatonic scale starting E4?
        # Let's map directly
        note_y_map = {
            "C4": -4, "D4": -2, "E4": 0, "F4": 2, "G4": 4, "A4": 6, "B4": 8, "C5": 10, "D5": 12, "E5": 14, "F5": 16
        }
        # My previous map was: E4=0, F4=1 (step=1). Lines at 0,2,4,6,8. Correct.
        # Wait, if step is 1 (line/space), then:
        # E4(Line 1)=0. F4(Space 1)=1. G4(Line 2)=2.
        pass
        
    elif clef_type == "Bass":
        # Draw Simplified F-Clef (Curve + Dots)
        # F3 is Line 4 (y=6)
        # Curve
        t = np.linspace(0, 3.14, 50)
        ax.plot(1 + 0.5*np.sin(t), 6 + 1.5*np.cos(t), color='black', linewidth=2) # Arc
        ax.plot([1, 1], [3, 7.5], color='black', linewidth=2) # Vertical
        # Dots around Line 4 (y=6) -> at y=6.5 and y=5.5?
        # Actually F-Clef dots are in Space 3 (y=5) and Space 4 (y=7)? No.
        # F-Clef dots flank the F-line (Line 4). So spaces above and below.
        ax.scatter([2], [7], s=20, color='black') # Space 4
        ax.scatter([2], [5], s=20, color='black') # Space 3
        ax.text(1, -4, "F-Clef", fontsize=8, ha='center')
        
        # Note Mapping (Bass)
        # G2(Line 1)=0. A2=1. B2=2. C3=3. D3=4. E3=5. F3(Line 4)=6. G3=7. A3(Line 5)=8. B3=9. C4=10. D4=11. E4=12. F4=13.
        # Note: Step=1.
    
    # Unified calculation
    # Define "Line 1" pitch
    if clef_type == "Treble":
        ref_note = "E4"; ref_y = 0
    else:
        ref_note = "G2"; ref_y = 0
        
    # Simple Diatonic Distance (ignoring accidentals for y-pos)
    scale = ["C", "D", "E", "F", "G", "A", "B"]
    
    def get_rank(n):
        # n = "F4" -> octave=4, note="F"
        note = n[0]
        octave = int(n[1])
        base_val = octave * 7 + scale.index(note)
        return base_val

    target_rank = get_rank(note_name)
    ref_rank = get_rank(ref_note)
    
    y_pos = (target_rank - ref_rank) + ref_y
    
    # Draw Note Head (Black)
    ax.scatter([5], [y_pos], s=400, color='black', marker='o')
    
    # Draw Ledger Lines
    # Staff Range: 0 to 8.
    # Below Staff: -2, -4 -> Draw lines
    # Above Staff: 10, 12, 14 -> Draw lines
    
    # Check bottom ledgers
    curr_y = -2
    while curr_y >= y_pos:
        ax.hlines(curr_y, 4, 6, colors='black', linewidth=1)
        curr_y -= 2
        
    # Check top ledgers
    curr_y = 10
    while curr_y <= y_pos:
        ax.hlines(curr_y, 4, 6, colors='black', linewidth=1)
        curr_y += 2
    
    # Label Note Name below
    ax.text(5, y_pos - 4.5, note_name, fontsize=12, ha='center', color='black', fontweight='bold')
    
    ax.set_ylim(-6, 16)
    ax.set_xlim(0, 8)
    
    # Save to buffer with White Background (NOT transparent)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf
