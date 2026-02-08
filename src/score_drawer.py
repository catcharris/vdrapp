import matplotlib.pyplot as plt
import io

def draw_staff_and_note(note_name):
    """
    Draws a simple musical staff with a single note.
    note_name: "F4", "E4", "D4"
    """
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.set_axis_off()
    
    # Draw Staff Lines (y=0, 1, 2, 3, 4) relative coords
    # 0 = Bottom Line (E4 usually in Treble)
    # Actually Treble Clef Lines: E4, G4, B4, D5, F5
    # y-coords: 0, 2, 4, 6, 8 (if step is 1 per note) or let's use standard spacing
    
    # Standard spacing: line every 2 units? No, let's say line every 1 unit is confusing.
    # Lines at y = 0, 2, 4, 6, 8.
    # E4 is Line 1 (y=0). F4 is Space 1 (y=1). G4 is Line 2 (y=2).
    
    y_lines = [0, 2, 4, 6, 8]
    for y in y_lines:
        ax.hlines(y, xmin=0, xmax=10, colors='black', linewidth=1)
        
    # Draw Clef (Simplified text for MVP)
    ax.text(0.5, 3, "𝄞", fontsize=30, ha='center', va='center') # Treble Clef Unicode
    
    # Note Position
    # D4: Below E4 -> y = -1
    # E4: Line 1 -> y = 0
    # F4: Space 1 -> y = 1
    
    note_y_map = {
        "D4": -1,
        "E4": 0,
        "F4": 1,
        "G4": 2,
        "A4": 3,
        "C4": -3 # Middle C (ledger line needed)
    }
    
    y_pos = note_y_map.get(note_name, 0)
    
    # Draw Note Head
    ax.scatter([5], [y_pos], s=300, color='black', marker='o')
    
    # Draw Ledger Line if needed (C4, D4?)
    # D4 is just below, usually doesn't need ledger line unless it's C4
    # But C4 needs one at -2.
    # D4 space is fine.
    
    # Label
    ax.text(5, y_pos - 2.5, note_name, fontsize=12, ha='center')
    
    ax.set_ylim(-4, 10)
    ax.set_xlim(0, 10)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf
